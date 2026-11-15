#include <array>
#include "gtest/gtest.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/cheetah/cdi_material.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/kernel/hal/intrinsic/nn/activation.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
TEST(NativeSyncMaterial, QuotaV2PreparesBothRingsAndFreshEpochs) {
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    RuntimeConfig cfg;cfg.set_protocol(CHEETAH);cfg.set_field(FM64);cfg.set_fxp_fraction_bits(18);
    cfg.mutable_cheetah_2pc_config()->set_ot_kind(YACL_Ferret);
    auto ctx=makeCheetahProtocol(cfg,link);
    auto* state=ctx->getState<CheetahMulState>();
    const int64_t lanes=ctx->getState<CheetahOTState>()->maximum_instances();
    auto data=ring_zeros(FM64,{lanes+2,4});
    auto words=NdArrayView<uint64_t>(data);
    words[4*lanes]=16;words[4*(lanes+1)]=2;words[4*(lanes+1)+1]=16;
    const auto q=Value(data.as(makeType<Pub2kTy>(FM64)),DT_I64);
    Value anchor(ring_zeros(FM64,{1}).as(makeType<Pub2kTy>(FM64)),DT_I64);
    NdArrayRef previous;
    for(int64_t epoch:{1,2}){
      dynDispatch<std::vector<Value>>(ctx.get(),"cdi_material_prepare",q,epoch);
      const auto prepared=state->Resources().generated;
      dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{anchor},std::string("inference"),epoch,std::string("begin"));
      auto value=state->TakeCachedBeaver(FM64,4)[0];
      state->TakeCachedBeaver(FM32,4);
      if(epoch==2){EXPECT_FALSE(ring_all_equal(value,previous));}
      previous=value;
      EXPECT_EQ(state->Resources().generated,prepared);
      dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{anchor},std::string("inference"),epoch,std::string("end"));
      EXPECT_ANY_THROW(state->TakeCachedBeaver(FM32,state->Resources(FM32).stock_remaining+1));
      EXPECT_EQ(state->Resources().generated,prepared);
      dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{anchor},std::string("materials"),epoch,std::string("release"));
      EXPECT_EQ(state->Resources().stock_remaining,0);
    }
    const auto generated=state->Resources().generated;
    words[4*(lanes+1)]=3;
    EXPECT_ANY_THROW(dynDispatch<std::vector<Value>>(ctx.get(),"cdi_material_prepare",q,int64_t{3}));
    EXPECT_EQ(state->Resources().generated,generated);
  });
}

TEST(NativeSyncMaterial, GeluRestoresOuterFormatWhenAnUnpreparedLaneFails) {
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    RuntimeConfig cfg;cfg.set_protocol(CHEETAH);cfg.set_field(FM64);cfg.set_fxp_fraction_bits(18);
    cfg.mutable_cheetah_2pc_config()->set_ot_kind(YACL_Ferret);
    auto ctx=makeCheetahProtocol(cfg,link);
    ctx->getState<CheetahOTState>()->SetStrictInventory(true);
    Value x(ring_zeros(FM64,{4}).as(makeType<AShrTy>(FM64)),DT_F32);
    EXPECT_ANY_THROW(kernel::hal::intrinsic::nn::f_seg3_gelu(ctx.get(),x,true));
    EXPECT_EQ(ctx->getField(),FM64);
    EXPECT_EQ(ctx->getFxpBits(),18);
  });
}

TEST(NativeSyncMaterial, NativeExpRestoresF18AndUsesF12OutputLattice) {
  std::array<NdArrayRef,2> output;
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    RuntimeConfig cfg;cfg.set_protocol(CHEETAH);cfg.set_field(FM64);cfg.set_fxp_fraction_bits(18);
    cfg.set_fxp_exp_iters(6);cfg.mutable_cheetah_2pc_config()->set_ot_kind(YACL_Ferret);
    auto ctx=makeCheetahProtocol(cfg,link);
    auto data=ring_zeros(FM64,{8});
    if(link->Rank()==0)for(int i=0;i<8;++i)data.at<uint64_t>(i)=static_cast<uint64_t>(-int64_t(10000+31001*i));
    output[link->Rank()]=kernel::hal::intrinsic::nn::f_neg_exp_taylor(ctx.get(),Value(data.as(makeType<AShrTy>(FM64)),DT_F32)).data();
    EXPECT_EQ(ctx->getField(),FM64);EXPECT_EQ(ctx->getFxpBits(),18);
  });
  auto clear=ring_add(output[0],output[1]);
  for(int i=0;i<8;++i){
    EXPECT_GT(clear.at<uint64_t>(i),0);

    EXPECT_EQ(clear.at<uint64_t>(i)%64,0);
  }
}

TEST(NativeSyncMaterial, TwoPreparedRingsSurviveAlternationAndRelease) {
  std::array<std::array<NdArrayRef,3>,2> small,large,second;
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    RuntimeConfig cfg;cfg.set_protocol(CHEETAH);cfg.set_field(FM64);
    cfg.set_fxp_fraction_bits(18);
    cfg.mutable_cheetah_2pc_config()->set_ot_kind(YACL_Ferret);
    auto ctx=makeCheetahProtocol(cfg,link);
    auto* state=ctx->getState<CheetahMulState>();
    state->ReserveBeavers(FM64,16);
    state->ReserveBeavers(FM32,16);
    const auto prepared=state->Resources().generated;

    large[link->Rank()]=state->TakeCachedBeaver(FM64,4);
    small[link->Rank()]=state->TakeCachedBeaver(FM32,4);
    second[link->Rank()]=state->TakeCachedBeaver(FM64,4);
    EXPECT_EQ(state->Resources().generated,prepared);
    EXPECT_EQ(state->Resources().stock_served,12);
    state->ReleasePreparedBeavers();
    EXPECT_EQ(state->Resources().stock_remaining,0);
  });
  for(auto* pair:{&small,&large,&second}){
    auto a=ring_add((*pair)[0][0],(*pair)[1][0]);
    auto b=ring_add((*pair)[0][1],(*pair)[1][1]);
    auto c=ring_add((*pair)[0][2],(*pair)[1][2]);
    auto expected=ring_mul(a,b);
    EXPECT_TRUE(ring_all_equal(c,expected));
  }
}
}
