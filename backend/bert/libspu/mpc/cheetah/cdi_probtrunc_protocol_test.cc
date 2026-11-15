#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "libspu/core/context.h"
#include "libspu/core/trace.h"
#include "libspu/core/type_util.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/mpc/cheetah/cdi_inversion.h"
#include "libspu/mpc/cheetah/cdi_material.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
namespace {
constexpr int64_t S = int64_t{1} << 18;
struct ShadowRow {std::string op; std::vector<Value> inputs; Value output;size_t bits;};
thread_local std::vector<ShadowRow>* shadow_rows=nullptr;
void Observe(const char* name,const std::vector<Value>& inputs,const Value& output,size_t bits) {
  const std::string op(name);
  if(op=="_mul"||op=="_square"||op=="_trunc")
    shadow_rows->push_back({op,inputs,output,bits});
}
int64_t Clear(const Value& a,const Value& b,int64_t i) {
  const uint64_t x=NdArrayView<const uint64_t>(a.data())[i];
  const uint64_t y=NdArrayView<const uint64_t>(b.data())[i];
  return static_cast<int64_t>(a.isPublic()?x:x+y);
}

void CheckNativeTruncation(const std::string& op, int k, int expected_truncs, bool mixed=false) {
  std::array<NdArrayRef, 2> results;
  std::array<std::vector<ShadowRow>,2> shadows;
  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& link) {
    RuntimeConfig cfg;
    cfg.set_protocol(ProtocolKind::CHEETAH);
    cfg.set_field(FieldType::FM64);
    cfg.set_fxp_fraction_bits(18);
    cfg.set_fxp_div_goldschmidt_iters(2);
    cfg.mutable_cheetah_2pc_config()->set_ot_kind(CheetahOtKind::YACL_Ferret);
    cfg.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
    cfg.mutable_cheetah_2pc_config()->set_approx_less_precision(4);
    auto ctx = makeCheetahProtocol(cfg, link);
    initTrace(ctx->id(), TR_HAL | TR_MPC | TR_REC);
    auto x = ring_zeros(FieldType::FM64, {8});
    auto ab = ring_zeros(FieldType::FM64, {8,2});
    auto routes = ring_zeros(FieldType::FM64, {8});
    auto ks = ring_zeros(FieldType::FM64, {8});
    auto xv = NdArrayView<uint64_t>(x);
    auto av = NdArrayView<uint64_t>(ab);
    auto kv = NdArrayView<uint64_t>(ks);
    for (int i = 0; i < 8; ++i) {
      kv[i] = mixed ? i % (k+1) : k;
      if (link->Rank() == 0) {
        xv[i] = (op == "reciprocal" ? 2 : 4) * S;
        av[2*i] = static_cast<uint64_t>(op == "reciprocal" ? -S/8 : -S/32);
        av[2*i+1] = op == "reciprocal" ? 3*S/4 : 5*S/8;
      }
    }
    shadow_rows=&shadows[link->Rank()];
    kernel::hal::SetRingShadowObserver(Observe);
    auto out = dynDispatch<std::vector<Value>>(
        ctx.get(), "cdi_invert", Value(x.as(makeType<AShrTy>(FieldType::FM64)), DT_F32),
        Value(ab.as(makeType<AShrTy>(FieldType::FM64)), DT_F32),
        Value(routes.as(makeType<Pub2kTy>(FieldType::FM64)), DT_I64),
        Value(ks.as(makeType<Pub2kTy>(FieldType::FM64)), DT_I64), op);
    kernel::hal::SetRingShadowObserver(nullptr);shadow_rows=nullptr;
    ASSERT_EQ(out.size(), 1U);
    results[link->Rank()] = out[0].data();
    int truncs = 0;
    for (const auto& rec : GET_TRACER(ctx.get())->getProfState()->getRecords()) {
      if (rec.name == "trunc_a") ++truncs;
    }
    EXPECT_EQ(truncs, expected_truncs) << "must call native Cheetah truncation";
  });
  ASSERT_EQ(shadows[0].size(),shadows[1].size());
  int64_t checked_products=0,checked_truncations=0;
  for(size_t row=0;row<shadows[0].size();++row){
    const auto& a=shadows[0][row];const auto& b=shadows[1][row];
    ASSERT_EQ(a.op,b.op);ASSERT_EQ(a.bits,b.bits);
    for(int64_t i=0;i<a.output.numel();++i){
      const auto x=Clear(a.inputs[0],b.inputs[0],i);
      const auto out=Clear(a.output,b.output,i);
      if(a.op=="_trunc"){

        EXPECT_LE(std::abs(out-(x>>a.bits)),2);++checked_truncations;
      }else{
        const auto y=a.op=="_square"?x:Clear(a.inputs[1],b.inputs[1],i);
        const __int128 product=static_cast<__int128>(x)*y;
        EXPECT_TRUE(product>=static_cast<__int128>(INT64_MIN) && product<=static_cast<__int128>(INT64_MAX));

        EXPECT_LE(std::abs(out-static_cast<int64_t>(product)),1);++checked_products;
      }
    }
  }
  EXPECT_GT(checked_products,0);EXPECT_GT(checked_truncations,0);
  std::cout<<"SHADOW op="<<op<<" K="<<k<<" mixed="<<mixed<<" products="<<checked_products<<" truncations="<<checked_truncations<<"\n";
  auto clear = ring_add(results[0], results[1]);
  auto view = NdArrayView<const uint64_t>(clear);
  for (int i = 0; i < 8; ++i) {
    const int64_t decoded = static_cast<int64_t>(view[i]);
    EXPECT_GT(decoded, 0);
    EXPECT_LE(std::abs(decoded - S/2), 16) << "known reciprocal/rsqrt fixed point";
  }
}

TEST(CdiProbTruncProtocol, ReciprocalKeepsTwoProductsPerRound) {
  CheckNativeTruncation("reciprocal", 2, 5);
}
TEST(CdiProbTruncProtocol, RsqrtKeepsSeparateHalf) {
  CheckNativeTruncation("rsqrt", 2, 9);
}
TEST(CdiProbTruncProtocol, ZeroKStillComputesAffineInitializer) {
  CheckNativeTruncation("reciprocal", 0, 1);
}
TEST(CdiProbTruncProtocol, MixedKUsesMaximumRoundsNotSumOfGroups) {
  CheckNativeTruncation("reciprocal", 6, 13, true);
  CheckNativeTruncation("rsqrt", 3, 13, true);
}
TEST(CdiProbTruncProtocol, TimingIdentityAndFormatGuard) {
  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& link) {
    RuntimeConfig cfg; cfg.set_protocol(CHEETAH);cfg.set_field(FM64);cfg.set_fxp_fraction_bits(18);
    auto ctx=makeCheetahProtocol(cfg,link);
    auto v=Value(ring_zeros(FM64,{3}).as(makeType<AShrTy>(FM64)),DT_F32);
    auto observe=[&](std::string scope,int64_t site,std::string action){
      return dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{v},scope,site,action);
    };
    EXPECT_EQ(observe("inference",42,"begin")[0].data().data(),v.data().data());
    observe("softmax",0,"begin");observe("reciprocal",0,"begin");
    EXPECT_ANY_THROW(observe("softmax",0,"end"));
    observe("reciprocal",0,"end");observe("softmax",0,"end");observe("inference",42,"end");
    cfg.set_fxp_fraction_bits(16);auto bad=makeCheetahProtocol(cfg,link);
    EXPECT_ANY_THROW(dynDispatch<std::vector<Value>>(bad.get(),"cdi_scope",std::vector<Value>{v},
        std::string("barrier"),int64_t{0},std::string("barrier")));
  });
}
}  
}
