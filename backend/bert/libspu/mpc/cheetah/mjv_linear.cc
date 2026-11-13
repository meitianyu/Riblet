// Copyright 2026

#include "libspu/mpc/cheetah/mjv_linear.h"

#include "libspu/mpc/cheetah/mjv_object_context.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/prg_state.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

namespace {

const char* LinearKernelFamilyName(LinearKernelFamily family) {
  switch (family) {
    case LinearKernelFamily::kMatMulAA:
      return "MatMulAA";
    case LinearKernelFamily::kMatMulAV:
      return "MatMulAV";
    case LinearKernelFamily::kMatMulVVS:
      return "MatMulVVS";
    case LinearKernelFamily::kBatchMatMulAA:
      return "BatchMatMulAA";
    case LinearKernelFamily::kBatchMatMulAV:
      return "BatchMatMulAV";
  }
  return "Unknown";
}

void CheckDynamicShapes(const Shape& lhs, const Shape& rhs, bool batched) {
  const int64_t rank = batched ? 3 : 2;
  SPU_ENFORCE(lhs.ndim() == rank && rhs.ndim() == rank,
              "dynamic matrix ranks must be {}", rank);
  if (batched) {
    SPU_ENFORCE(lhs[0] == rhs[0] && lhs[2] == rhs[1],
                "invalid batched matrix shapes {} and {}", lhs, rhs);
  } else {
    SPU_ENFORCE(lhs[1] == rhs[0], "invalid matrix shapes {} and {}", lhs,
                rhs);
  }
}

Shape ProductShape(const Shape& lhs, const Shape& rhs, bool batched) {
  return batched ? Shape({lhs[0], lhs[1], rhs[2]})
                 : Shape({lhs[0], rhs[1]});
}

NdArrayRef LocalMatMul(const NdArrayRef& lhs, const NdArrayRef& rhs,
                       bool batched) {
  if (!batched) {
    return ring_mmul(lhs, rhs);
  }
  const auto batch = lhs.shape()[0];
  const auto rows = lhs.shape()[1];
  const auto width = rhs.shape()[2];
  NdArrayRef output(lhs.eltype(), {batch, rows, width});
  for (int64_t index = 0; index < batch; ++index) {
    auto lhs_slice = lhs.slice({index, 0, 0},
                               {index + 1, rows, lhs.shape()[2]}, {1, 1, 1})
                         .reshape({rows, lhs.shape()[2]});
    auto rhs_slice = rhs.slice({index, 0, 0},
                               {index + 1, rhs.shape()[1], width}, {1, 1, 1})
                         .reshape({rhs.shape()[1], width});
    auto output_slice = output.slice({index, 0, 0},
                                     {index + 1, rows, width}, {1, 1, 1})
                            .reshape({rows, width});
    ring_mmul_(output_slice, lhs_slice, rhs_slice);
  }
  return output;
}

FieldType SecretField(const Value& value) {
  const auto* share_type = value.data().eltype().as<AShrTy>();
  SPU_ENFORCE(share_type != nullptr,
              "dynamic matrix inputs must be arithmetic shares");
  return share_type->field();
}

Value ReconstructDynamicOnline(
    KernelEvalContext* ctx, const Value& lhs, const Value& rhs,
    const DynamicConsumerKey& key,
    const MJVDynamicLinearState::CorrelationContext& correlation,
    const NdArrayRef& lhs_mask, const NdArrayRef& rhs_mask,
    LinearKernelFamily family) {
  const auto field = SecretField(lhs);
  SPU_ENFORCE(SecretField(rhs) == field, "dynamic matrix fields must match");
  SPU_ENFORCE(lhs.dtype() == rhs.dtype(),
              "dynamic matrix input dtypes must match");
  const auto* lhs_mask_type = lhs_mask.eltype().as<AShrTy>();
  const auto* rhs_mask_type = rhs_mask.eltype().as<AShrTy>();
  SPU_ENFORCE(lhs_mask_type != nullptr && rhs_mask_type != nullptr &&
                  lhs_mask_type->field() == field &&
                  rhs_mask_type->field() == field,
              "dynamic matrix mask fields must match");
  SPU_ENFORCE(lhs.shape() == correlation.lhs_shape &&
                  rhs.shape() == correlation.rhs_shape &&
                  lhs_mask.shape() == correlation.lhs_shape &&
                  rhs_mask.shape() == correlation.rhs_shape,
              "dynamic matrix shapes do not match preprocessing");

  auto* audit = ctx->getState<MJVLinearAuditState>();
  auto* scope = ctx->getState<MJVLinearScopeState>();
  if (correlation.batched && family == LinearKernelFamily::kMatMulVVS) {
    const Shape lhs_slice = {lhs.shape()[1], lhs.shape()[2]};
    const Shape rhs_slice = {rhs.shape()[1], rhs.shape()[2]};
    const Shape output_slice = {lhs.shape()[1], rhs.shape()[2]};
    for (int64_t batch = 0; batch < lhs.shape()[0]; ++batch) {
      auto replacement_family = family;
      if (scope->IsActive() &&
          scope->ActiveBackend() == LinearBackend::kOurs) {
        replacement_family = scope->ConsumeExpectedFamily(
            lhs_slice, rhs_slice, output_slice);
        SPU_ENFORCE(replacement_family == family,
                    "dynamic linear family differs from baseline");
      }
      audit->RecordHighLevel(replacement_family, lhs_slice, rhs_slice,
                             output_slice);
    }
  } else {
    const auto output_shape =
        ProductShape(lhs.shape(), rhs.shape(), correlation.batched);
    auto replacement_family = family;
    if (scope->IsActive() &&
        scope->ActiveBackend() == LinearBackend::kOurs) {
      replacement_family = scope->ConsumeExpectedFamily(
          lhs.shape(), rhs.shape(), output_shape);
      SPU_ENFORCE(replacement_family == family,
                  "dynamic linear family differs from baseline");
    }
    audit->RecordHighLevel(replacement_family, lhs.shape(), rhs.shape(),
                           output_shape);
  }

  auto* communicator = ctx->getState<Communicator>();
  const auto tag = "mjv-consumer-" + key.plan_sha256 + "-" +
                   std::to_string(key.request_epoch) + "-" + key.consumer_id;
  auto opened_lhs = communicator->allReduce(
      ReduceOp::ADD, ring_sub(lhs.data(), lhs_mask), tag + "-lhs");
  auto opened_rhs = communicator->allReduce(
      ReduceOp::ADD, ring_sub(rhs.data(), rhs_mask), tag + "-rhs");

  auto output = correlation.correlation.clone();
  ring_add_(output,
            LocalMatMul(opened_lhs, correlation.rhs_mask,
                        correlation.batched));
  ring_add_(output,
            LocalMatMul(correlation.lhs_mask, opened_rhs,
                        correlation.batched));
  if (communicator->getRank() == 0) {
    ring_add_(output,
              LocalMatMul(opened_lhs, opened_rhs, correlation.batched));
  }
  if (!correlation.physical_family_id.empty()) {
    ctx->getState<MJVObjectContextState>()->RecordRuntimeFamily(
        key.plan_sha256, correlation.model_epoch, key.request_epoch,
        correlation.physical_family_id);
  }
  return Value(output.as(makeType<AShrTy>(field)), lhs.dtype());
}

}  

Value ConsumeNextMJVDynamicOnline(KernelEvalContext* ctx, const Value& lhs,
                                  const Value& rhs,
                                  LinearKernelFamily family) {
  const bool batched = lhs.shape().ndim() == 3;
  CheckDynamicShapes(lhs.shape(), rhs.shape(), batched);
  auto* state = ctx->getState<MJVDynamicLinearState>();
  const auto key =
      state->NextOnlineConsumer(lhs.shape(), rhs.shape(), batched);
  const auto correlation = state->Consume(key);
  return ReconstructDynamicOnline(ctx, lhs, rhs, key, correlation,
                                  correlation.lhs_mask,
                                  correlation.rhs_mask, family);
}

void MJVLinearAuditState::Begin(LinearPhase phase, LinearBackend backend) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!active_, "linear-path audit is already active");
  active_ = true;
  snapshot_ = {};
  snapshot_.phase = phase;
  snapshot_.backend = backend;
}

LinearAuditSnapshot MJVLinearAuditState::End() {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(active_, "linear-path audit is not active");
  active_ = false;
  return snapshot_;
}

void MJVLinearAuditState::RecordHighLevel(LinearKernelFamily family,
                                          const Shape& lhs_shape,
                                          const Shape& rhs_shape,
                                          const Shape& output_shape) {
  std::lock_guard guard(mutex_);
  if (!active_) {
    return;
  }
  snapshot_.high_level_calls.push_back(
      {static_cast<int64_t>(snapshot_.high_level_calls.size()), family,
       lhs_shape, rhs_shape, output_shape});
}

void MJVLinearAuditState::RecordHeSubcall(bool batched) {
  std::lock_guard guard(mutex_);
  if (!active_) {
    return;
  }
  SPU_ENFORCE(snapshot_.phase != LinearPhase::kOnline ||
                  snapshot_.backend != LinearBackend::kOurs,
              "Ours online linear path attempted an HE matrix subcall");
  if (batched) {
    snapshot_.batch_dot_ole_calls += 1;
  } else {
    snapshot_.dot_ole_calls += 1;
  }
}

void MJVLinearAuditState::Abort() {
  std::lock_guard guard(mutex_);
  active_ = false;
  snapshot_ = {};
}

void VerifyLinearPathParity(
    const std::vector<LinearCallFingerprint>& baseline,
    const std::vector<LinearCallFingerprint>& replacement) {
  SPU_ENFORCE_EQ(baseline.size(), replacement.size(),
                 "linear-path call count mismatch");
  for (size_t index = 0; index < baseline.size(); ++index) {
    SPU_ENFORCE(baseline[index] == replacement[index],
                "linear-path fingerprint mismatch at ordinal {}", index);
  }
}

void MJVDynamicLinearState::Store(DynamicConsumerKey key,
                                  CorrelationContext correlation) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!key.plan_sha256.empty() && key.request_epoch >= 0 &&
                  !key.consumer_id.empty(),
              "dynamic consumer context identity is incomplete");
  SPU_ENFORCE(used_correlation_keys_.find(key) ==
                  used_correlation_keys_.end(),
              "consumer correlation {} at epoch {} already exists",
              key.consumer_id, key.request_epoch);
  const DynamicRequestContextKey context{key.plan_sha256, key.request_epoch};
  used_correlation_keys_.emplace(key);
  prepared_order_[context].push_back(key);
  correlations_.emplace(std::move(key), std::move(correlation));
}

int64_t MJVDynamicLinearState::LiveCorrelationCount() const {
  std::lock_guard guard(mutex_);
  return static_cast<int64_t>(correlations_.size());
}

void MJVDynamicLinearState::BeginOnline(std::string plan_sha256,
                                        int64_t request_epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!online_active_, "dynamic online scope is already active");
  SPU_ENFORCE(!plan_sha256.empty() && request_epoch >= 0,
              "dynamic request context identity is incomplete");
  const DynamicRequestContextKey context{std::move(plan_sha256),
                                         request_epoch};
  const auto found = prepared_order_.find(context);
  online_order_.clear();
  if (found != prepared_order_.end()) {
    online_order_ = found->second;
  }
  for (const auto& key : online_order_) {
    SPU_ENFORCE(correlations_.find(key) != correlations_.end(),
                "consumer correlation {} was already consumed",
                key.consumer_id);
  }
  online_active_ = true;
  online_context_ = context;
  online_cursor_ = 0;
}

void MJVDynamicLinearState::EndOnline(const std::string& plan_sha256,
                                      int64_t request_epoch) {
  std::lock_guard guard(mutex_);
  const DynamicRequestContextKey context{plan_sha256, request_epoch};
  SPU_ENFORCE(online_active_ && online_context_ == context,
              "dynamic online request epoch {} is not active",
              request_epoch);
  const auto consumed = online_cursor_;
  const auto expected = static_cast<int64_t>(online_order_.size());
  online_active_ = false;
  prepared_order_.erase(context);
  online_context_ = {};
  online_order_.clear();
  online_cursor_ = 0;
  SPU_ENFORCE_EQ(consumed, expected,
                 "dynamic online correlation count mismatch");
}

DynamicConsumerKey MJVDynamicLinearState::NextOnlineConsumer(
    const Shape& lhs_shape, const Shape& rhs_shape, bool batched) const {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(online_active_, "dynamic online scope is not active");
  SPU_ENFORCE(online_cursor_ < static_cast<int64_t>(online_order_.size()),
              "dynamic online invoked more matrix operations than prepared");
  const auto& key = online_order_[online_cursor_];
  const auto found = correlations_.find(key);
  SPU_ENFORCE(found != correlations_.end(),
              "consumer correlation {} does not exist", key.consumer_id);
  SPU_ENFORCE(found->second.lhs_shape == lhs_shape &&
                  found->second.rhs_shape == rhs_shape &&
                  found->second.batched == batched,
              "dynamic online fingerprint mismatch for consumer {}",
              key.consumer_id);
  return key;
}

void MJVDynamicLinearState::AbortOnline() {
  std::lock_guard guard(mutex_);
  online_active_ = false;
  online_context_ = {};
  online_order_.clear();
  online_cursor_ = 0;
}

MJVDynamicLinearState::CorrelationContext MJVDynamicLinearState::Consume(
    const DynamicConsumerKey& key) {
  std::lock_guard guard(mutex_);
  const auto found = correlations_.find(key);
  if (found == correlations_.end()) {
    SPU_ENFORCE(used_correlation_keys_.find(key) ==
                    used_correlation_keys_.end(),
                "consumer correlation {} was already consumed",
                key.consumer_id);
    SPU_THROW("consumer correlation {} does not exist", key.consumer_id);
  }
  if (online_active_) {
    SPU_ENFORCE(online_cursor_ < static_cast<int64_t>(online_order_.size()) &&
                    key == online_order_[online_cursor_],
                "dynamic online operation order mismatch");
  }
  auto correlation = std::move(found->second);
  correlations_.erase(found);
  if (online_active_) {
    online_cursor_ += 1;
  }
  return correlation;
}

void MJVLinearScopeState::RegisterPublicPlan(
    const std::string& hash,const std::vector<LinearCallFingerprint>& calls) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!active_ && hash.size()==64 && !calls.empty(),"invalid public linear plan");
  for(size_t i=0;i<calls.size();++i){
    const auto& c=calls[i];
    const bool batched=c.family==LinearKernelFamily::kBatchMatMulAA || c.family==LinearKernelFamily::kBatchMatMulAV;
    CheckDynamicShapes(c.lhs_shape,c.rhs_shape,batched);
    SPU_ENFORCE(c.ordinal==static_cast<int64_t>(i) && c.output_shape==ProductShape(c.lhs_shape,c.rhs_shape,batched),"invalid public matrix shape");
  }
  const auto found=baselines_.find(hash);
  SPU_ENFORCE(found==baselines_.end() || found->second==calls,"public plan changed under same hash");
  baselines_[hash]=calls;
}

void MJVLinearScopeState::Begin(int64_t request_id, std::string manifest_hash,
                                LinearBackend backend) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!active_, "linear scope is already active");
  SPU_ENFORCE(request_id >= 0, "linear scope request ID must be nonnegative");
  SPU_ENFORCE(!manifest_hash.empty(), "linear scope manifest is empty");
  active_ = true;
  request_id_ = request_id;
  manifest_hash_ = std::move(manifest_hash);
  backend_ = backend;
  replacement_cursor_ = 0;
  if (backend == LinearBackend::kOurs) {
    SPU_ENFORCE(baselines_.find(manifest_hash_) != baselines_.end(),
                "linear baseline manifest {} does not exist", manifest_hash_);
  }
}

LinearBackend MJVLinearScopeState::ActiveBackend() const {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(active_, "linear scope is not active");
  return backend_;
}

std::string MJVLinearScopeState::ActiveManifestHash() const {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(active_, "linear scope is not active");
  return manifest_hash_;
}

bool MJVLinearScopeState::IsActive() const {
  std::lock_guard guard(mutex_);
  return active_;
}

LinearKernelFamily MJVLinearScopeState::ConsumeExpectedFamily(
    const Shape& lhs_shape, const Shape& rhs_shape,
    const Shape& output_shape) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(active_ && backend_ == LinearBackend::kOurs,
              "Ours linear replacement scope is not active");
  const auto found = baselines_.find(manifest_hash_);
  SPU_ENFORCE(found != baselines_.end(),
              "linear baseline manifest {} does not exist", manifest_hash_);
  SPU_ENFORCE(replacement_cursor_ <
                  static_cast<int64_t>(found->second.size()),
              "Ours emitted more linear replacements than the baseline");
  const auto& expected = found->second[replacement_cursor_];
  SPU_ENFORCE(expected.ordinal == replacement_cursor_ &&
                  expected.lhs_shape == lhs_shape &&
                  expected.rhs_shape == rhs_shape &&
                  expected.output_shape == output_shape,
              "linear replacement shape mismatch: manifest={} ordinal={} "
              "expected family={} lhs={} rhs={} output={}; observed lhs={} "
              "rhs={} output={}",
              manifest_hash_, replacement_cursor_,
              LinearKernelFamilyName(expected.family), expected.lhs_shape,
              expected.rhs_shape, expected.output_shape, lhs_shape, rhs_shape,
              output_shape);
  replacement_cursor_ += 1;
  return expected.family;
}

void MJVLinearScopeState::Finish(int64_t request_id,
                                 const LinearAuditSnapshot& snapshot) {
  std::vector<LinearCallFingerprint> baseline;
  std::string manifest;
  LinearBackend backend;
  {
    std::lock_guard guard(mutex_);
    SPU_ENFORCE(active_ && request_id_ == request_id,
                "linear scope request {} is not active", request_id);
    manifest = manifest_hash_;
    backend = backend_;
    active_ = false;
    manifest_hash_.clear();
    if (backend == LinearBackend::kBumbleBee) {
      baselines_[manifest] = snapshot.high_level_calls;
      return;
    }
    const auto found = baselines_.find(manifest);
    SPU_ENFORCE(found != baselines_.end(),
                "linear baseline manifest {} does not exist", manifest);
    baseline = found->second;
    SPU_ENFORCE_EQ(replacement_cursor_,
                   static_cast<int64_t>(baseline.size()),
                   "linear replacement count mismatch");
    replacement_cursor_ = 0;
  }
  SPU_ENFORCE_EQ(snapshot.dot_ole_calls, 0,
                 "Ours online DotOLE count must be zero");
  SPU_ENFORCE_EQ(snapshot.batch_dot_ole_calls, 0,
                 "Ours online BatchDotOLE count must be zero");
  VerifyLinearPathParity(baseline, snapshot.high_level_calls);
}

void MJVLinearScopeState::Abort() {
  std::lock_guard guard(mutex_);
  active_ = false;
  manifest_hash_.clear();
  replacement_cursor_ = 0;
}

void MJVConsumerPreprocess::evaluate(KernelEvalContext* ctx) const {
  DynamicConsumerKey key{ctx->getParam<std::string>(0),
                         ctx->getParam<int64_t>(1),
                         ctx->getParam<std::string>(2)};
  const auto& lhs_shape = ctx->getParam<Shape>(3);
  const auto& rhs_shape = ctx->getParam<Shape>(4);
  const auto batched = ctx->getParam<bool>(5);
  SPU_ENFORCE(ctx->numParams() == 6 || ctx->numParams() == 13,
              "dynamic consumer metadata is incomplete");
  SPU_ENFORCE(!key.plan_sha256.empty() && key.request_epoch >= 0 &&
                  !key.consumer_id.empty(),
              "consumer preprocessing identity is incomplete");
  CheckDynamicShapes(lhs_shape, rhs_shape, batched);

  const auto field = ctx->sctx()->getField();
  auto* prg = ctx->getState<PrgState>();
  auto lhs_mask =
      prg->genPriv(field, lhs_shape).as(makeType<AShrTy>(field));
  auto rhs_mask =
      prg->genPriv(field, rhs_shape).as(makeType<AShrTy>(field));
  const auto kernel = batched ? "batch_mmul_aa" : "mmul_aa";
  auto correlation =
      dynDispatch<Value>(ctx->sctx(), kernel, Value(lhs_mask, DT_INVALID),
                         Value(rhs_mask, DT_INVALID))
          .data();
  int64_t model_epoch = 0;
  std::string physical_family_id;
  if (ctx->numParams() == 13) {
    model_epoch = ctx->getParam<int64_t>(6);
    physical_family_id = ctx->getParam<std::string>(7);
    const auto lifetime = ctx->getParam<int64_t>(10);
    const auto operand = ctx->getParam<int64_t>(11);
    SPU_ENFORCE(lifetime >= 0 && lifetime <= 2 && operand >= 0 && operand <= 1,
                "invalid dynamic runtime family enum");
    const auto object_operand = static_cast<ObjectOperand>(operand);
    const auto& shared_shape =
        object_operand == ObjectOperand::kLhs ? lhs_shape : rhs_shape;
    const auto& right_shape =
        object_operand == ObjectOperand::kLhs ? rhs_shape : lhs_shape;
    ctx->getState<MJVObjectContextState>()->RegisterRequestFamily(
        key.plan_sha256, model_epoch, key.request_epoch, physical_family_id,
        ctx->getParam<std::string>(8), ctx->getParam<int64_t>(9),
        static_cast<ObjectLifetime>(lifetime), object_operand, shared_shape,
        right_shape, ctx->getParam<std::string>(12));
  }
  ctx->getState<MJVDynamicLinearState>()->Store(
      std::move(key),
      {lhs_mask, rhs_mask, std::move(correlation), lhs_shape, rhs_shape,
       batched, model_epoch, std::move(physical_family_id)});
  ctx->pushOutput(std::vector<Value>{Value(std::move(lhs_mask), DT_INVALID),
                                     Value(std::move(rhs_mask), DT_INVALID)});
}

void MJVDynamicOnline::evaluate(KernelEvalContext* ctx) const {
  if (ctx->numParams() == 3) {
    const auto& lhs = ctx->getParam<Value>(0);
    const auto& rhs = ctx->getParam<Value>(1);
    const auto encoded_family = ctx->getParam<int64_t>(2);
    SPU_ENFORCE(encoded_family >= 0 && encoded_family <= 4,
                "invalid dynamic matrix family {}", encoded_family);
    ctx->pushOutput(ConsumeNextMJVDynamicOnline(
        ctx, lhs, rhs, static_cast<LinearKernelFamily>(encoded_family)));
    return;
  }

  SPU_ENFORCE(ctx->numParams() == 7,
              "dynamic online expects routed or explicit-handle inputs");
  const auto& lhs = ctx->getParam<Value>(2);
  const auto& rhs = ctx->getParam<Value>(3);
  const DynamicConsumerKey key{ctx->getParam<std::string>(4),
                               ctx->getParam<int64_t>(5),
                               ctx->getParam<std::string>(6)};
  const auto correlation = ctx->getState<MJVDynamicLinearState>()->Consume(key);
  const auto lhs_mask = ctx->getParam<Value>(0).data();
  const auto rhs_mask = ctx->getParam<Value>(1).data();
  auto family = correlation.batched ? LinearKernelFamily::kBatchMatMulAA
                                    : LinearKernelFamily::kMatMulAA;
  ctx->pushOutput(ReconstructDynamicOnline(ctx, lhs, rhs, key, correlation,
                                           lhs_mask, rhs_mask,
                                           family));
}

}
