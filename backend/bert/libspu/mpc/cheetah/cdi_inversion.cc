// Copyright 2026

#include "libspu/mpc/cheetah/cdi_inversion.h"
#include "libspu/mpc/cheetah/cdi_schedule.h"

#include <array>
#include <chrono>
#include "spdlog/spdlog.h"
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::mpc::cheetah {
namespace {

constexpr int64_t kFractionalBits = 18;
constexpr uint64_t kScale = uint64_t{1} << kFractionalBits;
struct CostParts { int64_t mul_ns=0, trunc_ns=0, mul_calls=0, trunc_calls=0; };
thread_local CostParts* active_cost = nullptr;
struct CostScope {
  CostParts* previous;
  explicit CostScope(CostParts* value) : previous(active_cost) { active_cost=value; }
  ~CostScope(){active_cost=previous;}
};

Value ConstantRaw(SPUContext* ctx, uint64_t value, const Shape& shape,
                  DataType dtype) {
  return kernel::hal::_constant(ctx, value, shape).setDtype(dtype);
}

Value CheetahTruncate(SPUContext* ctx, const Value& input, size_t bits) {
  SPU_ENFORCE(bits > 0 && bits < 64, "invalid CDI shift width");

  const auto start=active_cost ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  auto result=kernel::hal::_trunc(ctx, input, bits, SignType::Unknown).setDtype(input.dtype());
  if(active_cost){active_cost->trunc_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();++active_cost->trunc_calls;}
  return result;
}

Value CdiMulF18(SPUContext* ctx, const Value& left, const Value& right) {
  SPU_ENFORCE(left.shape() == right.shape() && left.dtype() == right.dtype(),
              "CDI multiply shape/dtype mismatch");
  const auto start=active_cost ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  auto raw = kernel::hal::_mul(ctx, left, right).setDtype(left.dtype());
  if(active_cost){active_cost->mul_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();++active_cost->mul_calls;}
  return CheetahTruncate(ctx, raw, kFractionalBits)
      .setDtype(left.dtype());
}

Value ReciprocalStep(SPUContext* ctx, const Value& operand, const Value& y) {
  const auto two = ConstantRaw(ctx, 2 * kScale, operand.shape(),
                               operand.dtype());
    const auto product = CdiMulF18(ctx, operand, y);
    const auto correction = kernel::hal::_sub(ctx, two, product)
                                .setDtype(operand.dtype());
    return CdiMulF18(ctx, y, correction);
}

Value RsqrtStep(SPUContext* ctx, const Value& operand, const Value& y) {
  const auto three = ConstantRaw(ctx, 3 * kScale, operand.shape(),
                                 operand.dtype());
    const auto yy = CdiMulF18(ctx, y, y);
    const auto xyy = CdiMulF18(ctx, operand, yy);
    const auto correction = kernel::hal::_sub(ctx, three, xyy)
                                .setDtype(operand.dtype());
    return CheetahTruncate(ctx, CdiMulF18(ctx, y, correction), 1);
}

Value GatherVector(SPUContext* ctx, const Value& flat,
                   const std::vector<int64_t>& indices) {
  (void)ctx;
  SPU_ENFORCE(!indices.empty(), "cannot gather an empty CDI group");
  NdArrayRef result(flat.storage_type(), {static_cast<int64_t>(indices.size())});
  const auto source = NdArrayView<const uint64_t>(flat.data());
  auto target = NdArrayView<uint64_t>(result);
  for (size_t i=0; i<indices.size(); ++i) target[i] = source[indices[i]];
  return Value(std::move(result), flat.dtype());
}

Value GatherCoefficient(SPUContext* ctx, const Value& coefficients,
                        const std::vector<int64_t>& indices,
                        int64_t column) {
  (void)ctx;
  SPU_ENFORCE(!indices.empty(), "cannot gather an empty CDI group");
  NdArrayRef result(coefficients.storage_type(), {static_cast<int64_t>(indices.size())});
  const auto source = NdArrayView<const uint64_t>(coefficients.data());
  auto target = NdArrayView<uint64_t>(result);
  for (size_t i=0; i<indices.size(); ++i) target[i] = source[2*indices[i]+column];
  return Value(std::move(result), coefficients.dtype());
}

Value Scatter(const Value& destination, const Value& values,
              const std::vector<int64_t>& indices) {

  auto result = destination.data().clone();
  auto target = NdArrayView<uint64_t>(result);
  const auto source = NdArrayView<const uint64_t>(values.data());
  for(size_t i=0;i<indices.size();++i) target[indices[i]]=source[i];
  return Value(std::move(result), destination.dtype());
}

const Pub2kTy* RequirePublicRing(const Value& value, std::string_view label) {
  const auto* type = value.storage_type().as<Pub2kTy>();
  SPU_ENFORCE(type != nullptr && type->field() == FieldType::FM64,
              "{} must be public FM64", label);
  return type;
}

}  

void CdiInvert::evaluate(KernelEvalContext* ctx) const {
  using Clock = std::chrono::steady_clock;
  const bool profile = ctx->sctx()->config().enable_hal_profile();
  const auto cost_start = profile ? Clock::now() : Clock::time_point{};
  int64_t normal_ns = 0, fallback_ns = 0, normal_count = 0, fallback_count = 0;
  int64_t affine_ns=0, iteration_ns=0;
  CostParts cost_parts;
  CostScope cost_scope(profile ? &cost_parts : nullptr);
  SPU_ENFORCE(ctx->numParams() == 5, "CDI inversion arguments are incomplete");
  const auto& operand_input = ctx->getParam<Value>(0);
  const auto& coefficient_input = ctx->getParam<Value>(1);
  const auto& routes = ctx->getParam<Value>(2);
  const auto& public_ks = ctx->getParam<Value>(3);
  const auto& operator_name = ctx->getParam<std::string>(4);
  SPU_ENFORCE(operator_name == "reciprocal" || operator_name == "rsqrt",
              "unsupported CDI operator {}", operator_name);
  SPU_ENFORCE(operand_input.isSecret() && coefficient_input.isSecret(),
              "CDI operand and coefficients must be secret");
  SPU_ENFORCE(operand_input.dtype() == DT_F32 &&
                  coefficient_input.dtype() == DT_F32,
              "CDI operands and coefficients must use f18/f32 semantics");
  SPU_ENFORCE(coefficient_input.shape().ndim() == 2 &&
                  coefficient_input.shape()[1] == 2,
              "CDI coefficients must have shape [N,2]");
  RequirePublicRing(routes, "CDI route");
  RequirePublicRing(public_ks, "CDI public K");

  auto* sctx = ctx->sctx();
  SPU_ENFORCE(sctx->getField() == FM64 && sctx->getFxpBits() == 18,
              "CDI compiled ABI requires FM64/fxp18");
  const auto original_shape = operand_input.shape();
  const auto count = operand_input.numel();
  SPU_ENFORCE(count > 0 && coefficient_input.shape()[0] == count,
              "CDI operand/coefficient count mismatch");
  SPU_ENFORCE(routes.numel() == count && public_ks.numel() == count,
              "CDI public schedule count mismatch");

  const auto operand =
      kernel::hal::reshape(sctx, operand_input, {count});
  const auto coefficients =
      kernel::hal::reshape(sctx, coefficient_input, {count, 2});
  const auto route_view = NdArrayView<const uint64_t>(routes.data());
  const auto k_view = NdArrayView<const uint64_t>(public_ks.data());

  std::vector<int64_t> normal, fallback, normal_ks;
  for (int64_t index = 0; index < count; ++index) {
    SPU_ENFORCE(route_view[index] <= 1, "CDI route must be a bit");
    SPU_ENFORCE(k_view[index] <= 8, "CDI public K must be in 0..8");
    if(route_view[index]) fallback.push_back(index);
    else {normal.push_back(index);normal_ks.push_back(k_view[index]);}
  }

  auto result = Value(operand.data().clone(), operand.dtype());
  if(!fallback.empty()) {
    const auto start=profile ? Clock::now() : Clock::time_point{};
    const auto x=GatherVector(sctx,operand,fallback);
    const auto output=operator_name=="reciprocal" ? kernel::hal::f_reciprocal(sctx,x) : kernel::hal::f_rsqrt(sctx,x);
    if(profile) fallback_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
    result=Scatter(result,output,fallback);fallback_count=fallback.size();
  }
  if(!normal.empty()) {
    const auto start=profile ? Clock::now() : Clock::time_point{};
    const auto x=GatherVector(sctx,operand,normal);
    const auto a=GatherCoefficient(sctx,coefficients,normal,0);
    const auto b=GatherCoefficient(sctx,coefficients,normal,1);
    const auto affine_start=profile ? Clock::now() : Clock::time_point{};
    auto y=kernel::hal::_add(sctx,CdiMulF18(sctx,a,x),b).setDtype(x.dtype());
    if(profile) affine_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-affine_start).count();
    const auto iteration_start=profile ? Clock::now() : Clock::time_point{};
    for(const auto& active:ActiveSchedule(normal_ks)) {
      const auto xx=GatherVector(sctx,x,active);
      const auto yy=GatherVector(sctx,y,active);
      const auto next=operator_name=="reciprocal" ? ReciprocalStep(sctx,xx,yy) : RsqrtStep(sctx,xx,yy);
      y=Scatter(y,next,active);
    }
    if(profile) {
      iteration_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-iteration_start).count();
      normal_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
    }
    result=Scatter(result,y,normal);normal_count=normal.size();
  }
  result = kernel::hal::reshape(sctx, result, original_shape)
               .setDtype(operand_input.dtype());
  ctx->pushOutput(std::vector<Value>{std::move(result)});
  if(profile) {
  const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now() - cost_start).count();

  SPDLOG_INFO("CDI_COST op={} events={} normal={} fallback={} total_ns={} normal_ns={} fallback_ns={}",
              operator_name, count, normal_count, fallback_count,
              total_ns, normal_ns, fallback_ns);
  SPDLOG_INFO("CDI_PARTS op={} affine_ns={} iteration_ns={} mul_ns={} trunc_ns={} mul_calls={} trunc_calls={}",
              operator_name,affine_ns,iteration_ns,cost_parts.mul_ns,cost_parts.trunc_ns,cost_parts.mul_calls,cost_parts.trunc_calls);
  }
}

}
