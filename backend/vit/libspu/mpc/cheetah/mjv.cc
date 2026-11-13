#include "libspu/core/cpp_prepare_observer.h"

// Copyright 2026

#include "libspu/mpc/cheetah/mjv.h"

#include "libspu/mpc/cheetah/mjv_linear.h"
#include "libspu/mpc/cheetah/mjv_object_context.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
namespace {

FieldType SecretField(const Value& value) {
  const auto* share_type = value.storage_type().as<AShrTy>();
  SPU_ENFORCE(share_type != nullptr, "MJV-CT expects arithmetic shares");
  return share_type->field();
}

void CheckRankTwo(const Value& value, std::string_view name) {
  SPU_ENFORCE(value.shape().ndim() == 2, "{} must be rank two, got {}", name,
              value.shape());
}

}  

void MJVContextState::RequireNewModel(int64_t model_id) const {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(used_model_ids_.find(model_id) == used_model_ids_.end(),
              "MJV model id {} was already used", model_id);
}

void MJVContextState::StoreModel(int64_t model_id, ModelContext model) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(used_model_ids_.find(model_id) == used_model_ids_.end(),
              "MJV model id {} was already used", model_id);
  stats_.model_setup_count += 1;
  stats_.right_operand_opened_elements += model.mask.numel();
  used_model_ids_.emplace(model_id);
  models_.emplace(model_id, std::move(model));
}

MJVContextState::ModelContext MJVContextState::GetModel(
    int64_t model_id) const {
  std::lock_guard guard(mutex_);
  const auto found = models_.find(model_id);
  SPU_ENFORCE(found != models_.end(), "MJV model id {} does not exist",
              model_id);
  return found->second;
}

void MJVContextState::ReleaseModelEpoch(const std::string& plan_sha256,
                                        int64_t model_epoch) {

spu::lifecycle::Scope scope_mjv_model_release("MJV.model-release", "A");
scope_mjv_model_release.Outcome("retired");

  std::lock_guard guard(mutex_);
  std::vector<int64_t> model_ids;
  for (const auto& [model_id, model] : models_) {
    if (model.runtime_family.plan_sha256 == plan_sha256 &&
        model.runtime_family.model_epoch == model_epoch) {
      model_ids.push_back(model_id);
    }
  }
  for (const auto& [request_id, request] : requests_) {
    (void)request_id;
    SPU_ENFORCE(
        std::find(model_ids.begin(), model_ids.end(), request.model_id) ==
            model_ids.end(),
        "model epoch has an unconsumed MJV request");
  }
  for (const auto model_id : model_ids) {
    models_.erase(model_id);
  }
}

void MJVContextState::RequireNewRequest(int64_t model_id,
                                        int64_t request_id) const {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(models_.find(model_id) != models_.end(),
              "MJV model id {} does not exist", model_id);
  SPU_ENFORCE(used_request_ids_.find(request_id) == used_request_ids_.end(),
              "MJV request id {} already exists", request_id);
}

void MJVContextState::StoreRequest(int64_t request_id,
                                   RequestContext request) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(used_request_ids_.find(request_id) == used_request_ids_.end(),
              "MJV request id {} already exists", request_id);
  const auto model = models_.find(request.model_id);
  SPU_ENFORCE(model != models_.end(), "MJV model id {} does not exist",
              request.model_id);
  stats_.request_preprocess_count += 1;
  stats_.correlation_elements += request.correlation.numel();
  used_request_ids_.emplace(request_id);
  requests_.emplace(request_id, std::move(request));
}

MJVContextState::RequestContext MJVContextState::ConsumeRequest(
    int64_t model_id, int64_t request_id, int64_t opened_elements) {
  std::lock_guard guard(mutex_);
  const auto found = requests_.find(request_id);
  if (found == requests_.end()) {
    SPU_ENFORCE(used_request_ids_.find(request_id) == used_request_ids_.end(),
                "MJV request id {} was already consumed", request_id);
    SPU_THROW("MJV request id {} does not exist", request_id);
  }
  SPU_ENFORCE(found->second.model_id == model_id,
              "MJV request id {} belongs to model {}, not {}", request_id,
              found->second.model_id, model_id);
  auto request = std::move(found->second);
  requests_.erase(found);
  stats_.online_count += 1;
  stats_.joint_view_opened_elements += opened_elements;
  return request;
}

int64_t MJVContextState::LiveRequestCount() const {
  std::lock_guard guard(mutex_);
  return static_cast<int64_t>(requests_.size());
}

MJVStats MJVContextState::GetStats() const {
  std::lock_guard guard(mutex_);
  return stats_;
}

void MJVModelSetup::evaluate(KernelEvalContext* ctx) const {

spu::lifecycle::Scope scope_mjv_model_mask_opening("MJV.model-mask-opening", "R-M", [&] { return static_cast<uint64_t>((ctx->getState<Communicator>()->lctx())->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_mjv_model_mask_opening.Outcome("initialized");

  const auto& weight = ctx->getParam<Value>(0);
  const auto model_id = ctx->getParam<int64_t>(1);
  const auto& output_widths = ctx->getParam<Sizes>(2);
  const auto sum_r = ctx->getParam<int64_t>(3);
  SPU_ENFORCE(ctx->numParams() == 4 || ctx->numParams() == 5 ||
                  ctx->numParams() == 15,
              "MJV model setup metadata is incomplete");
  const auto family = ctx->numParams() >= 5
                          ? static_cast<LinearKernelFamily>(
                                ctx->getParam<int64_t>(4))
                          : LinearKernelFamily::kMatMulAA;

  CheckRankTwo(weight, "MJV weight");
  const auto field = SecretField(weight);
  SPU_ENFORCE(!output_widths.empty(), "MJV consumer group must not be empty");
  SPU_ENFORCE(std::all_of(output_widths.begin(), output_widths.end(),
                          [](int64_t width) { return width > 0; }),
              "MJV output widths must be positive");
  const auto total_width = std::accumulate(
      output_widths.begin(), output_widths.end(), int64_t{0});
  SPU_ENFORCE(total_width == weight.shape()[1],
              "MJV output widths {} do not match weight width {}",
              total_width, weight.shape()[1]);
  const auto rho = weight.shape()[0];
  SPU_ENFORCE(sum_r >= rho, "MJV sum_r {} must be at least rho {}", sum_r,
              rho);
  SPU_ENFORCE(family == LinearKernelFamily::kMatMulAA ||
                  family == LinearKernelFamily::kMatMulAV ||
                  family == LinearKernelFamily::kMatMulVVS,
              "invalid static MJV linear family {}",
              static_cast<int64_t>(family));

  auto* state = ctx->getState<MJVContextState>();
  state->RequireNewModel(model_id);
  auto mask = ctx->getState<PrgState>()
                  ->genPriv(field, weight.shape())
                  .as(makeType<AShrTy>(field));
  const auto local_delta = ring_sub(weight.data(), mask);
  auto opened_delta =
      ctx->getState<Communicator>()->allReduce(
          ReduceOp::ADD, local_delta,
          "mjv-model-delta-" + std::to_string(model_id))
          .as(makeType<Pub2kTy>(field));

  MJVContextState::RuntimeFamilyBinding runtime_family;
  if (ctx->numParams() == 15) {
    runtime_family.plan_sha256 = ctx->getParam<std::string>(5);
    runtime_family.model_epoch = ctx->getParam<int64_t>(6);
    runtime_family.physical_family_id = ctx->getParam<std::string>(7);
    const auto lifetime = ctx->getParam<int64_t>(10);
    const auto operand = ctx->getParam<int64_t>(11);
    SPU_ENFORCE(lifetime >= 0 && lifetime <= 2 && operand >= 0 && operand <= 1,
                "invalid MJV runtime family enum");
    ctx->getState<MJVObjectContextState>()->RegisterModelFamily(
        runtime_family.plan_sha256, runtime_family.model_epoch,
        runtime_family.physical_family_id, ctx->getParam<std::string>(8),
        ctx->getParam<int64_t>(9), static_cast<ObjectLifetime>(lifetime),
        static_cast<ObjectOperand>(operand), ctx->getParam<Shape>(13),
        ctx->getParam<Shape>(14), ctx->getParam<std::string>(12));
  }

  state->StoreModel(model_id,
                    {std::move(mask), std::move(opened_delta), output_widths,
                     sum_r, rho, weight.dtype(), family,
                     std::move(runtime_family)});
  ctx->pushOutput(Value(weight.data(), weight.dtype()));
}

void MJVRequestPreprocess::evaluate(KernelEvalContext* ctx) const {

spu::lifecycle::Scope scope_mjv_request_correlation("MJV.request-correlation", "unclassified", [&] { return static_cast<uint64_t>((ctx->getState<Communicator>()->lctx())->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_mjv_request_correlation.Outcome("prepared");

  const auto& model_handle = ctx->getParam<Value>(0);
  const auto model_id = ctx->getParam<int64_t>(1);
  const auto request_id = ctx->getParam<int64_t>(2);
  const auto rows = ctx->getParam<int64_t>(3);
  SPU_ENFORCE(ctx->numParams() == 4 || ctx->numParams() == 8,
              "MJV request preprocessing metadata is incomplete");

  CheckRankTwo(model_handle, "MJV model handle");
  SecretField(model_handle);
  SPU_ENFORCE(rows > 0, "MJV request rows must be positive");
  auto* state = ctx->getState<MJVContextState>();
  state->RequireNewRequest(model_id, request_id);
  const auto model = state->GetModel(model_id);
  SPU_ENFORCE(model_handle.shape() == model.mask.shape(),
              "MJV model handle shape {} does not match {}",
              model_handle.shape(), model.mask.shape());
  SPU_ENFORCE(model_handle.dtype() == model.dtype,
              "MJV model handle dtype does not match setup");

  const auto field = model.mask.eltype().as<AShrTy>()->field();

spu::lifecycle::Scope scope_mjv_fresh_request_mask("MJV.fresh-request-mask", "F");

  auto request_mask = ctx->getState<PrgState>()
                          ->genPriv(field, {rows, model.rho})
                          .as(makeType<AShrTy>(field));

scope_mjv_fresh_request_mask.Stop();

spu::lifecycle::Scope scope_mjv_actual_correlation_generation("MJV.actual-correlation-generation", "F", [&] { return static_cast<uint64_t>((ctx->getState<Communicator>()->lctx())->GetStats()->sent_bytes); }, "yacl-shared-inclusive");

  auto correlation =
      dynDispatch<Value>(ctx->sctx(), "mmul_aa",
                         Value(request_mask, model.dtype),
                         Value(model.mask, model.dtype))
          .data();

scope_mjv_actual_correlation_generation.Stop();

  auto runtime_family = model.runtime_family;
  if (ctx->numParams() == 8) {
    SPU_ENFORCE(runtime_family.isBound(),
                "request runtime family has no model binding");
    SPU_ENFORCE(runtime_family.plan_sha256 ==
                        ctx->getParam<std::string>(4) &&
                    runtime_family.model_epoch == ctx->getParam<int64_t>(5) &&
                    runtime_family.physical_family_id ==
                        ctx->getParam<std::string>(7),
                "request runtime family differs from model setup");
    runtime_family.request_epoch = ctx->getParam<int64_t>(6);
    ctx->getState<MJVObjectContextState>()->RegisterRequestFamily(
        runtime_family.plan_sha256, runtime_family.model_epoch,
        runtime_family.request_epoch, runtime_family.physical_family_id);
  } else {
    runtime_family = {};
  }
  state->StoreRequest(
      request_id,
      {model_id, std::move(correlation), rows, std::move(runtime_family)});
  ctx->pushOutput(Value(std::move(request_mask), model.dtype));
}

void MJVGroupOnline::evaluate(KernelEvalContext* ctx) const {

spu::lifecycle::Scope scope_mjv_online_consume("MJV.online-consume", "F", [&] { return static_cast<uint64_t>((ctx->getState<Communicator>()->lctx())->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_mjv_online_consume.Outcome("consumed");

  const auto& request_handle = ctx->getParam<Value>(0);
  const auto& joint_activation = ctx->getParam<Value>(1);
  const auto model_id = ctx->getParam<int64_t>(2);
  const auto request_id = ctx->getParam<int64_t>(3);
  const auto logical_row_members =
      ctx->numParams() == 5 ? ctx->getParam<int64_t>(4) : int64_t{0};

  CheckRankTwo(request_handle, "MJV request handle");
  CheckRankTwo(joint_activation, "MJV joint activation");
  const auto field = SecretField(request_handle);
  SPU_ENFORCE(SecretField(joint_activation) == field,
              "MJV request and activation fields do not match");
  SPU_ENFORCE(request_handle.shape() == joint_activation.shape(),
              "MJV request shape {} does not match activation {}",
              request_handle.shape(), joint_activation.shape());
  SPU_ENFORCE(request_handle.dtype() == joint_activation.dtype(),
              "MJV request and activation dtypes do not match");

  auto* state = ctx->getState<MJVContextState>();
  const auto model = state->GetModel(model_id);
  SPU_ENFORCE(request_handle.shape()[1] == model.rho,
              "MJV request width {} does not match rho {}",
              request_handle.shape()[1], model.rho);
  SPU_ENFORCE(request_handle.dtype() == model.dtype,
              "MJV request dtype does not match model setup");
  const auto request = state->ConsumeRequest(
      model_id, request_id, joint_activation.numel());
  SPU_ENFORCE(request.rows == request_handle.shape()[0],
              "MJV request rows {} do not match handle {}", request.rows,
              request_handle.shape()[0]);

  auto* audit = ctx->getState<MJVLinearAuditState>();
  auto* scope = ctx->getState<MJVLinearScopeState>();
  if (logical_row_members > 0) {
    SPU_ENFORCE(model.output_widths.size() == 1,
                "logical row expansion requires one output width");
    SPU_ENFORCE_EQ(logical_row_members, request.rows,
                   "logical row count must equal physical batch rows");
  }
  const auto combined_width = std::accumulate(
      model.output_widths.begin(), model.output_widths.end(), int64_t{0});
  const Shape rhs_shape = {model.rho, combined_width};
  const Shape output_shape = {request.rows, combined_width};
  auto family = model.linear_family;
  if (scope->IsActive() &&
      scope->ActiveBackend() == LinearBackend::kOurs) {
    family = scope->ConsumeExpectedFamily(joint_activation.shape(), rhs_shape,
                                          output_shape);
  }
  audit->RecordHighLevel(family, joint_activation.shape(), rhs_shape,
                         output_shape);

  const auto local_gamma =
      ring_sub(joint_activation.data(), request_handle.data());
  const auto opened_gamma =
      ctx->getState<Communicator>()->allReduce(
          ReduceOp::ADD, local_gamma,
          "mjv-joint-view-" + std::to_string(request_id))
          .as(makeType<Pub2kTy>(field));

  auto output = request.correlation.clone();
  ring_add_(output, ring_mmul(opened_gamma, model.mask));
  ring_add_(output, ring_mmul(request_handle.data(), model.opened_delta));
  if (ctx->getState<Communicator>()->getRank() == 0) {
    ring_add_(output, ring_mmul(opened_gamma, model.opened_delta));
  }
  output = output.as(makeType<AShrTy>(field));

  std::vector<Value> consumers;
  consumers.reserve(model.output_widths.size());
  int64_t offset = 0;
  for (const auto width : model.output_widths) {
    const auto end = offset + width;
    consumers.emplace_back(
        output.slice({0, offset}, {request.rows, end}, {1, 1}), model.dtype);
    offset = end;
  }
  if (request.runtime_family.isBound()) {
    ctx->getState<MJVObjectContextState>()->RecordRuntimeFamily(
        request.runtime_family.plan_sha256,
        request.runtime_family.model_epoch,
        request.runtime_family.request_epoch,
        request.runtime_family.physical_family_id);
  }
  ctx->pushOutput(std::move(consumers));
}

}
