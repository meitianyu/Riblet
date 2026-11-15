// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Modified for Riblet; see NOTICE. */

#include "libspu/device/pphlo/pphlo_intrinsic_executor.h"

#include "llvm/ADT/SmallVector.h"
#include "spdlog/spdlog.h"

#include "libspu/device/intrinsic_table.h"
#include "libspu/kernel/hal/debug.h"
#include "libspu/device/pphlo/ours_vit_bridge.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/intrinsic/nn/activation.h"
#include "libspu/kernel/hlo/basic_binary.h"
#include "libspu/kernel/hlo/casting.h"
#include "libspu/kernel/hlo/const.h"
#include "libspu/kernel/hlo/geometrical.h"
#include "libspu/kernel/hlo/indexing.h"
#include "libspu/kernel/hlo/rank.h"
#include "libspu/mpc/cheetah/mjv.h"
#include "libspu/mpc/cheetah/mjv_linear.h"
#include "libspu/mpc/cheetah/mjv_object_context.h"

namespace spu::device::pphlo {

namespace {

std::string ParseLinearPlanDigest(llvm::StringRef text) {
  SPU_ENFORCE(text.size() == 64,
              "Linear plan SHA-256 must contain 64 characters");
  for (const char value : text) {
    SPU_ENFORCE((value >= '0' && value <= '9') ||
                    (value >= 'a' && value <= 'f'),
                "Linear plan SHA-256 must be lowercase hex");
  }
  return text.str();
}

int64_t ParseLinearInteger(llvm::StringRef text, llvm::StringRef field) {
  int64_t value = 0;
  SPU_ENFORCE(!text.getAsInteger(10, value) && value >= 0,
              "Invalid linear plan {} {}", field.str(), text.str());
  return value;
}

llvm::SmallVector<llvm::StringRef, 5> SplitMJVTarget(
    llvm::StringRef target, llvm::StringRef expected_name,
    size_t expected_parts) {
  llvm::SmallVector<llvm::StringRef, 5> parts;
  target.split(parts, ':', -1, false);
  SPU_ENFORCE(parts.size() == expected_parts && parts[0] == expected_name,
              "Invalid MJV-CT call target {}", target.str());
  return parts;
}

int64_t ParseMJVInteger(llvm::StringRef text, llvm::StringRef field,
                        bool allow_zero) {
  int64_t value = 0;
  SPU_ENFORCE(!text.getAsInteger(10, value) &&
                  (allow_zero ? value >= 0 : value > 0),
              "Invalid MJV-CT {} {}", field.str(), text.str());
  return value;
}

Sizes ParseMJVWidths(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef, 5> encoded_widths;
  text.split(encoded_widths, ',', -1, false);
  SPU_ENFORCE(!encoded_widths.empty(), "MJV-CT requires output widths");
  Sizes widths;
  widths.reserve(encoded_widths.size());
  for (const auto encoded : encoded_widths) {
    widths.push_back(ParseMJVInteger(encoded, "output width", false));
  }
  return widths;
}

Shape ParseMJVShape(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef, 4> encoded_dimensions;
  text.split(encoded_dimensions, 'x', -1, false);
  SPU_ENFORCE(!encoded_dimensions.empty(), "MJV-CT requires a matrix shape");
  Shape shape;
  shape.reserve(encoded_dimensions.size());
  for (const auto encoded : encoded_dimensions) {
    shape.push_back(ParseMJVInteger(encoded, "shape dimension", false));
  }
  return shape;
}

mpc::cheetah::LinearKernelFamily ParseMJVLinearFamily(
    llvm::StringRef text) {
  if (text == "matmul_aa") {
    return mpc::cheetah::LinearKernelFamily::kMatMulAA;
  }
  if (text == "matmul_av") {
    return mpc::cheetah::LinearKernelFamily::kMatMulAV;
  }
  SPU_ENFORCE(text == "matmul_vvs", "Invalid MJV linear family {}",
              text.str());
  return mpc::cheetah::LinearKernelFamily::kMatMulVVS;
}

int64_t ParseMJVObjectLifetime(llvm::StringRef text) {
  if (text == "s") {
    return static_cast<int64_t>(
        mpc::cheetah::ObjectLifetime::kModelStatic);
  }
  SPU_ENFORCE(text == "r", "Invalid MJV object lifetime {}", text.str());
  return static_cast<int64_t>(
      mpc::cheetah::ObjectLifetime::kRequestImmutable);
}

int64_t ParseMJVObjectOperand(llvm::StringRef text) {
  if (text == "l") {
    return static_cast<int64_t>(mpc::cheetah::ObjectOperand::kLhs);
  }
  SPU_ENFORCE(text == "r", "Invalid MJV object operand {}", text.str());
  return static_cast<int64_t>(mpc::cheetah::ObjectOperand::kRhs);
}

Value MJVSecretInput(SPUContext* ctx, const Value& input) {
  SPU_ENFORCE(input.isSecret() || input.isPrivate(),
              "MJV-CT inputs must be secret or private");
  auto secret = input.isSecret()
                    ? input
                    : kernel::hlo::Cast(ctx, input, VIS_SECRET, input.dtype());
  SPU_ENFORCE(secret.isFxp(), "MJV-CT inputs must be fixed point");
  return secret;
}

DataType MJVFixedPointDtype(mlir::Type type) {
  mlir::spu::pphlo::TypeTools tools(type.getContext());
  auto expressed = tools.getExpressedType(type);
  if (const auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(expressed)) {
    expressed = tensor.getElementType();
  }
  auto floating = mlir::dyn_cast<mlir::FloatType>(expressed);
  SPU_ENFORCE(floating != nullptr,
              "MJV-CT dynamic handles must use a fixed-point tensor type");
  switch (floating.getWidth()) {
    case 16:
      return DT_F16;
    case 32:
      return DT_F32;
    case 64:
      return DT_F64;
    default:
      SPU_THROW("unsupported MJV-CT fixed-point width {}", floating.getWidth());
  }
}

}  

std::vector<Value> intrinsic_dispatcher(SPUContext* ctx,
                                        mlir::spu::pphlo::CustomCallOp& call,
                                        absl::Span<const Value> inputs) {
  // FIXME: This should be something register by protocol
  auto name = call.getCallTargetName();
  if(name.starts_with("mjv_vit_linear_manifest:")){
    SPU_ENFORCE(inputs.size()==1 && call.getResults().size()==1);
    llvm::SmallVector<llvm::StringRef,3> parts;name.split(parts,':');
    SPU_ENFORCE(parts.size()==3);
    const auto hash=ParseLinearPlanDigest(parts[1]);
    llvm::SmallVector<llvm::StringRef,80> rows;parts[2].split(rows,';');
    SPU_ENFORCE(rows.size()==74, "invalid ViT linear manifest size");
    std::vector<mpc::cheetah::LinearCallFingerprint> calls;
    for(auto row:rows){
      llvm::SmallVector<llvm::StringRef,12> columns;row.split(columns,',');
      SPU_ENFORCE(columns.size()>=8);
      const auto kind=ParseLinearInteger(columns[0],"family");
      const auto rank=ParseLinearInteger(columns[1],"rank");
      SPU_ENFORCE(kind<=4 && (rank==2 || rank==3) && columns.size()==static_cast<size_t>(2+3*rank));
      mpc::cheetah::LinearCallFingerprint c;c.ordinal=calls.size();c.family=static_cast<mpc::cheetah::LinearKernelFamily>(kind);
      for(int64_t i=0;i<rank;++i){
        c.lhs_shape.push_back(ParseLinearInteger(columns[2+i],"lhs"));
        c.rhs_shape.push_back(ParseLinearInteger(columns[2+rank+i],"rhs"));
        c.output_shape.push_back(ParseLinearInteger(columns[2+2*rank+i],"output"));
      }
      calls.push_back(std::move(c));
    }
    ctx->getState<mpc::cheetah::MJVLinearScopeState>()->RegisterValidatedPlan(hash,calls);
    return {inputs[0]};
  }
  if(name=="ours_vit_material_prepare_v2") {SPU_ENFORCE(inputs.size()==1);return {ours_vit::PrepareLanes(ctx,inputs[0])};}
  if(name=="ours_vit_material_lane_status_v2") {return {ours_vit::LaneStatus(ctx)};}
  if(name=="ours_vit_material_retire_v1") {return {ours_vit::Retire(ctx)};}
  if(name.starts_with("ours_vit_inverse_v1:")) {
    SPU_ENFORCE(inputs.size()==4);
    return {ours_vit::Inverse(ctx,inputs[0],inputs[1],inputs[2],inputs[3],name.drop_front(20).str())};
  }
  if (name == "example_binary") {
    SPDLOG_INFO("Binary example, input0 = {}, input1 = {}", inputs[0],
                inputs[1]);

    Shape result_shape = {inputs[0].shape()[0] + inputs[1].shape()[0],
                          inputs[0].shape()[1] + inputs[1].shape()[1]};

    auto zeros = kernel::hlo::Constant(ctx, 0, result_shape);

    if (inputs[0].isSecret() || inputs[1].isSecret()) {
      zeros = kernel::hlo::Cast(ctx, zeros, VIS_SECRET, inputs[0].dtype());
    } else {
      zeros = kernel::hlo::Cast(ctx, zeros, VIS_PUBLIC, inputs[0].dtype());
    }

    return {zeros};
  }

  if (name == MAKE_CACHED_VAR) {
    if (ctx->hasKernel("beaver_cache")) {
      SPU_ENFORCE(inputs.size() == 1);
      dynDispatch(ctx, "beaver_cache", inputs[0], true);
    }

    return {inputs[0]};
  }

  if (name == DROP_CACHED_VAR) {
    if (ctx->hasKernel("beaver_cache")) {
      SPU_ENFORCE(inputs.size() > 0);
      dynDispatch(ctx, "beaver_cache", inputs[0], false);
    }

    return {inputs[0]};
  }

  if (name.starts_with("mjv_linear_begin:")) {
    SPU_ENFORCE(inputs.size() <= 1 &&
                call.getResults().size() == inputs.size());
    const auto parts = SplitMJVTarget(name, "mjv_linear_begin", 7);
    const auto request_id = ParseMJVInteger(parts[2], "request id", true);
    SPU_ENFORCE(!parts[3].empty(), "MJV linear manifest must not be empty");
    const auto common_fallback_prefix_calls =
        ParseMJVInteger(parts[4], "common fallback prefix calls", true);
    const auto allowed_dot_ole_calls =
        ParseMJVInteger(parts[5], "allowed DotOLE calls", true);
    const auto allowed_batch_dot_ole_calls =
        ParseMJVInteger(parts[6], "allowed BatchDotOLE calls", true);
    mpc::cheetah::LinearBackend backend;
    if (parts[1] == "bumblebee") {
      backend = mpc::cheetah::LinearBackend::kBumbleBee;
    } else {
      SPU_ENFORCE(parts[1] == "ours", "Invalid MJV linear backend {}",
                  parts[1].str());
      backend = mpc::cheetah::LinearBackend::kOurs;
    }
    ctx->getState<mpc::cheetah::MJVLinearScopeState>()->Begin(
        request_id, parts[3].str(), backend, common_fallback_prefix_calls,
        allowed_dot_ole_calls, allowed_batch_dot_ole_calls);
    ctx->getState<mpc::cheetah::MJVLinearAuditState>()->Begin(
        mpc::cheetah::LinearPhase::kOnline, backend, allowed_dot_ole_calls,
        allowed_batch_dot_ole_calls);
    if (backend == mpc::cheetah::LinearBackend::kOurs) {
      ctx->getState<mpc::cheetah::MJVDynamicLinearState>()->BeginOnline(
          parts[3].str(), request_id);
    }
    return std::vector<Value>(inputs.begin(), inputs.end());
  }

  if (name.starts_with("mjv_epoch_begin:")) {
    SPU_ENFORCE(inputs.size() <= 1 &&
                call.getResults().size() == inputs.size());
    const auto parts = SplitMJVTarget(name, "mjv_epoch_begin", 4);
    const auto model_epoch = ParseMJVInteger(parts[2], "model epoch", true);
    const auto request_epoch =
        ParseMJVInteger(parts[3], "request epoch", true);
    auto* object_context =
        ctx->getState<mpc::cheetah::MJVObjectContextState>();
    object_context->BeginPlanExecution(parts[1].str(), model_epoch,
                                       request_epoch);
    object_context->RecordSecureInvoke();
    return std::vector<Value>(inputs.begin(), inputs.end());
  }

  if (name.starts_with("mjv_model_setup_batch:")) {
    SPU_ENFORCE(inputs.size() <= 1 &&
                call.getResults().size() == inputs.size());
    const auto parts = SplitMJVTarget(name, "mjv_model_setup_batch", 3);
    const auto model_epoch = ParseMJVInteger(parts[2], "model epoch", true);
    ctx->getState<mpc::cheetah::MJVObjectContextState>()
        ->RecordModelSetupBatch(parts[1].str(), model_epoch);
    return std::vector<Value>(inputs.begin(), inputs.end());
  }

  if (name.starts_with("mjv_request_preprocess_batch:")) {
    SPU_ENFORCE(inputs.size() <= 1 &&
                call.getResults().size() == inputs.size());
    const auto parts =
        SplitMJVTarget(name, "mjv_request_preprocess_batch", 4);
    const auto model_epoch = ParseMJVInteger(parts[2], "model epoch", true);
    const auto request_epoch =
        ParseMJVInteger(parts[3], "request epoch", true);
    ctx->getState<mpc::cheetah::MJVObjectContextState>()
        ->RecordRequestPreprocessBatch(parts[1].str(), model_epoch,
                                       request_epoch);
    return std::vector<Value>(inputs.begin(), inputs.end());
  }

  if (name.starts_with("mjv_audit_snapshot:")) {
    SPU_ENFORCE(inputs.size() <= 1 && call.getResults().size() == 1);
    const auto parts = SplitMJVTarget(name, "mjv_audit_snapshot", 4);
    const auto model_epoch = ParseMJVInteger(parts[2], "model epoch", true);
    const auto request_epoch =
        ParseMJVInteger(parts[3], "request epoch", true);
    const auto audit =
        ctx->getState<mpc::cheetah::MJVObjectContextState>()
            ->FinalizeRuntimeAudit(parts[1].str(), model_epoch, request_epoch);
    return {kernel::hlo::Constant(
        ctx, audit.ToVector(),
        Shape{static_cast<int64_t>(mpc::cheetah::RuntimeObjectAuditV1::
                                       kFieldCount)})};
  }

  if (name.starts_with("mjv_epoch_end:")) {
    SPU_ENFORCE(inputs.size() == 2 && call.getResults().size() == 1);
    const bool release_model_epoch = name.ends_with(":release_model");
    const auto parts = SplitMJVTarget(
        name, "mjv_epoch_end", release_model_epoch ? 5 : 4);
    if (release_model_epoch) {
      SPU_ENFORCE(parts[4] == "release_model",
                  "Invalid MJV-CT model release target {}", name.str());
    }
    const auto model_epoch = ParseMJVInteger(parts[2], "model epoch", true);
    const auto request_epoch =
        ParseMJVInteger(parts[3], "request epoch", true);
    ctx->getState<mpc::cheetah::MJVObjectContextState>()->EndPlanExecution(
        parts[1].str(), model_epoch, request_epoch, release_model_epoch);
    if (release_model_epoch) {
      ctx->getState<mpc::cheetah::MJVContextState>()->ReleaseModelEpoch(
          parts[1].str(), model_epoch);
    }
    return {inputs.front()};
  }

  if (name.starts_with("mjv_linear_end:")) {
    SPU_ENFORCE(inputs.size() <= 1 &&
                call.getResults().size() == inputs.size());
    const auto parts = SplitMJVTarget(name, "mjv_linear_end", 2);
    const auto request_id = ParseMJVInteger(parts[1], "request id", true);
    auto* scope = ctx->getState<mpc::cheetah::MJVLinearScopeState>();
    if (scope->ActiveBackend() == mpc::cheetah::LinearBackend::kOurs) {
      ctx->getState<mpc::cheetah::MJVDynamicLinearState>()->EndOnline(
          scope->ActiveManifestHash(), request_id);
    }
    const auto snapshot =
        ctx->getState<mpc::cheetah::MJVLinearAuditState>()->End();
    auto* object_context =
        ctx->getState<mpc::cheetah::MJVObjectContextState>();
    if (object_context->IsPlanExecutionActive()) {
      object_context->ReconcileLinearAudit(
          snapshot.backend == mpc::cheetah::LinearBackend::kOurs,
          static_cast<int64_t>(snapshot.high_level_calls.size()),
          snapshot.dot_ole_calls, snapshot.batch_dot_ole_calls);
    }
    scope->Finish(request_id, snapshot);
    return std::vector<Value>(inputs.begin(), inputs.end());
  }

  if (name.starts_with("mjv_consumer_preprocess:")) {
    SPU_ENFORCE(inputs.empty() && call.getResults().size() == 2);
    llvm::SmallVector<llvm::StringRef, 14> parts;
    name.split(parts, ':', -1, false);
    SPU_ENFORCE((parts.size() == 7 || parts.size() == 14) &&
                    parts[0] == "mjv_consumer_preprocess",
                "Invalid MJV-CT call target {}", name.str());
    const auto request_epoch =
        ParseMJVInteger(parts[2], "request epoch", true);
    SPU_ENFORCE(!parts[1].empty() && !parts[3].empty(),
                "MJV consumer identity must not be empty");
    SPU_ENFORCE(parts[4] == "m" || parts[4] == "b",
                "Invalid MJV consumer rank code {}", parts[4].str());
    const auto lhs_shape = ParseMJVShape(parts[5]);
    const auto rhs_shape = ParseMJVShape(parts[6]);
    const bool batched = parts[4] == "b";
    std::vector<Value> handles;
    if (parts.size() == 14) {
      handles = dynDispatch<std::vector<Value>>(
          ctx, "mjv_consumer_preprocess", parts[1].str(), request_epoch,
          parts[3].str(), lhs_shape, rhs_shape, batched,
          ParseMJVInteger(parts[7], "model epoch", true), parts[8].str(),
          parts[9].str(), ParseMJVInteger(parts[10], "logical members", false),
          ParseMJVObjectLifetime(parts[11]), ParseMJVObjectOperand(parts[12]),
          parts[13].str());
    } else {
      handles = dynDispatch<std::vector<Value>>(
          ctx, "mjv_consumer_preprocess", parts[1].str(), request_epoch,
          parts[3].str(), lhs_shape, rhs_shape, batched);
    }
    SPU_ENFORCE(handles.size() == 2);
    for (size_t index = 0; index < handles.size(); ++index) {
      handles[index].setDtype(MJVFixedPointDtype(call.getResult(index).getType()));
    }
    return handles;
  }

  if (name.starts_with("mjv_model_setup:")) {
    SPU_ENFORCE(inputs.size() == 1);
    llvm::SmallVector<llvm::StringRef, 15> parts;
    name.split(parts, ':', -1, false);
    SPU_ENFORCE((parts.size() == 4 || parts.size() == 5 ||
                 parts.size() == 15) &&
                    parts[0] == "mjv_model_setup",
                "Invalid MJV-CT call target {}", name.str());
    const auto model_id = ParseMJVInteger(parts[1], "model id", true);
    const auto sum_r = ParseMJVInteger(parts[2], "sum_r", false);
    const auto output_widths = ParseMJVWidths(parts[3]);
    const auto family = parts.size() == 5
                            ? ParseMJVLinearFamily(parts[4])
                            : mpc::cheetah::LinearKernelFamily::kMatMulAA;
    const auto weight = MJVSecretInput(ctx, inputs[0]);
    SPU_ENFORCE(weight.shape().ndim() == 2);
    SPU_ENFORCE(ctx->hasKernel("mjv_model_setup"),
                "MJV-CT is available only in the Cheetah backend");
    if (parts.size() == 15) {
      return {dynDispatch<Value>(
          ctx, "mjv_model_setup", weight, model_id, output_widths, sum_r,
          static_cast<int64_t>(family), parts[5].str(),
          ParseMJVInteger(parts[6], "model epoch", true), parts[7].str(),
          parts[8].str(), ParseMJVInteger(parts[9], "logical members", false),
          ParseMJVObjectLifetime(parts[10]), ParseMJVObjectOperand(parts[11]),
          parts[12].str(), ParseMJVShape(parts[13]),
          ParseMJVShape(parts[14]))};
    }
    return {dynDispatch<Value>(ctx, "mjv_model_setup", weight, model_id,
                               output_widths, sum_r,
                               static_cast<int64_t>(family))};
  }

  if (name.starts_with("mjv_request_preprocess:")) {
    SPU_ENFORCE(inputs.size() == 1);
    llvm::SmallVector<llvm::StringRef, 8> parts;
    name.split(parts, ':', -1, false);
    SPU_ENFORCE((parts.size() == 4 || parts.size() == 8) &&
                    parts[0] == "mjv_request_preprocess",
                "Invalid MJV-CT call target {}", name.str());
    const auto model_id = ParseMJVInteger(parts[1], "model id", true);
    const auto request_id = ParseMJVInteger(parts[2], "request id", true);
    const auto rows = ParseMJVInteger(parts[3], "rows", false);
    const auto model_handle = MJVSecretInput(ctx, inputs[0]);
    SPU_ENFORCE(model_handle.shape().ndim() == 2);
    SPU_ENFORCE(ctx->hasKernel("mjv_request_preprocess"),
                "MJV-CT is available only in the Cheetah backend");
    if (parts.size() == 8) {
      return {dynDispatch<Value>(
          ctx, "mjv_request_preprocess", model_handle, model_id, request_id,
          rows, parts[4].str(),
          ParseMJVInteger(parts[5], "model epoch", true),
          ParseMJVInteger(parts[6], "request epoch", true), parts[7].str())};
    }
    return {dynDispatch<Value>(ctx, "mjv_request_preprocess", model_handle,
                               model_id, request_id, rows)};
  }

  if (name.starts_with("mjv_group_online:")) {
    SPU_ENFORCE(inputs.size() == 2);
    llvm::SmallVector<llvm::StringRef, 5> parts;
    name.split(parts, ':', -1, false);
    SPU_ENFORCE((parts.size() == 4 || parts.size() == 5) &&
                    parts[0] == "mjv_group_online",
                "Invalid MJV-CT call target {}", name.str());
    const auto model_id = ParseMJVInteger(parts[1], "model id", true);
    const auto request_id = ParseMJVInteger(parts[2], "request id", true);
    const auto output_widths = ParseMJVWidths(parts[3]);
    const auto logical_row_members =
        parts.size() == 5
            ? ParseMJVInteger(parts[4], "logical row members", false)
            : int64_t{0};
    SPU_ENFORCE(call.getResults().size() == output_widths.size(),
                "MJV-CT result count does not match output widths");
    const auto request_handle = MJVSecretInput(ctx, inputs[0]);
    const auto activation = MJVSecretInput(ctx, inputs[1]);
    SPU_ENFORCE(request_handle.shape().ndim() == 2 &&
                activation.shape().ndim() == 2);
    SPU_ENFORCE(request_handle.shape() == activation.shape());
    SPU_ENFORCE(request_handle.dtype() == activation.dtype());
    SPU_ENFORCE(ctx->hasKernel("mjv_group_online"),
                "MJV-CT is available only in the Cheetah backend");

    auto products = logical_row_members > 0
                        ? dynDispatch<std::vector<Value>>(
                              ctx, "mjv_group_online", request_handle,
                              activation, model_id, request_id,
                              logical_row_members)
                        : dynDispatch<std::vector<Value>>(
                              ctx, "mjv_group_online", request_handle,
                              activation, model_id, request_id);
    SPU_ENFORCE(products.size() == output_widths.size());
    std::vector<Value> outputs;
    outputs.reserve(products.size());
    for (size_t index = 0; index < products.size(); ++index) {
      SPU_ENFORCE(products[index].shape() ==
                  Shape({request_handle.shape()[0], output_widths[index]}));
      outputs.push_back(kernel::hal::_trunc(ctx, products[index])
                            .setDtype(request_handle.dtype()));
    }
    return outputs;
  }
  // DO-NOT-EDIT: Add_DISPATCH_CODE

  // Default: Identity function
  if (name == "example") {
    SPDLOG_INFO("Calling example intrinsic");
    return {inputs.begin(), inputs.end()};
  }

  if (name == DBG_PRINT) {
    kernel::hal::dbg_print(ctx, inputs[0]);
    return {};
  }

  if (name == ERF) {
    SPU_ENFORCE(inputs.size() == 1 && inputs[0].isFxp());
    return {kernel::hal::f_erf(ctx, inputs[0])};
  }

  if (name == TOPK) {
    SPU_ENFORCE(inputs.size() == 1);
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("mhlo.attributes"));
    auto k = mlir::dyn_cast<mlir::IntegerAttr>(attr.get("k")).getInt();
    auto largest =
        mlir::dyn_cast<mlir::BoolAttr>(attr.get("largest")).getValue();

    auto value_only = false;

    if (auto value_only_attr = attr.get("value_only")) {
      value_only = mlir::dyn_cast<mlir::BoolAttr>(value_only_attr).getValue();
    }

    if (auto k_hi_attr = attr.get("k_hi")) {
      auto k_hi = mlir::dyn_cast<mlir::IntegerAttr>(k_hi_attr).getInt();
      return kernel::hlo::TopK(ctx, inputs[0], k, k_hi, largest, value_only);
    }

    return kernel::hlo::TopK(ctx, inputs[0], k, -1, largest, value_only);
  }

  if (name == GATHER) {
    kernel::hlo::GatherConfig config;
    const auto& output_shape =
        mlir::dyn_cast<mlir::RankedTensorType>(call.getResults()[0].getType())
            .getShape();
    auto attr =
        mlir::dyn_cast<mlir::DictionaryAttr>(call->getAttr("pphlo.attributes"));

    config.sliceSizes =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("slice_sizes"))
            .asArrayRef();
    config.indexVectorDim =
        mlir::dyn_cast<mlir::IntegerAttr>(attr.get("index_vector_dim"))
            .getInt();
    config.offsetDims =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("offset_dims"))
            .asArrayRef();
    config.collapsedSliceDims = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
                                    attr.get("collapsed_slice_dims"))
                                    .asArrayRef();
    config.startIndexMap =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.get("start_index_map"))
            .asArrayRef();

    return {
        kernel::hlo::Gather(ctx, inputs[0], inputs[1], config, output_shape)};
  }

  if (name == PREFER_A) {
    if (ctx->config().protocol() == ProtocolKind::CHEETAH) {
      // NOTE(juhou): For 2PC, MulAB uses COT which is efficient and accurate
      // than MulAA that needs HE. Thus we just by-pass the PreferAOp for 2PC.
      return {inputs[0]};
    }
    auto k0 =
        kernel::hlo::Cast(ctx, kernel::hlo::Constant(ctx, 0, inputs[0].shape()),
                          VIS_PUBLIC, inputs[0].dtype());
    return {kernel::hlo::Add(ctx, inputs[0], k0)};
  }

  if (name == GELU) {
    ours_vit::RestoreFormat restore(ctx);
    SPU_ENFORCE(inputs.size() == 1 && inputs[0].isFxp() &&
                inputs[0].isSecret());
    return {kernel::hal::intrinsic::nn::f_seg3_gelu(
        ctx, inputs[0], /*small_ring_compare*/ true)};
  }

  if (name == I64_GELU) {
    // TODO(lwj): a better lowering with IR attr.
    SPU_ENFORCE(inputs.size() == 1 && inputs[0].isFxp() &&
                inputs[0].isSecret());
    return {kernel::hal::intrinsic::nn::f_seg3_gelu(
        ctx, inputs[0], /*small_ring_compare*/ false)};
  }

  if (name == SILU) {
    SPU_ENFORCE(inputs.size() == 1 && inputs[0].isFxp() &&
                inputs[0].isSecret());
    return {kernel::hal::intrinsic::nn::f_seg4_silu(ctx, inputs[0])};
  }

  if (name == NEG_EXP) {
    SPU_ENFORCE(inputs.size() == 1 && inputs[0].isFxp() &&
                inputs[0].isSecret());
    return {kernel::hal::intrinsic::nn::f_neg_exp_taylor(ctx, inputs[0])};
  }

  SPU_THROW("Unhandled intrinsic call {}", name.str());
}

}  // namespace spu::device::pphlo
