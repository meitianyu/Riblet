#include <array>
#include <cmath>
#include "gtest/gtest.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/cdi_material.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
TEST(CdiMaterialProtocol, PreparedFreshMaterialsRunWithoutGenerationAndRejectReuse) {
  std::array<NdArrayRef,2> outputs;
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    RuntimeConfig cfg;cfg.set_protocol(CHEETAH);cfg.set_field(FM64);cfg.set_fxp_fraction_bits(18);
    cfg.set_fxp_div_goldschmidt_iters(2);
    cfg.mutable_cheetah_2pc_config()->set_ot_kind(CheetahOtKind::YACL_Ferret);
    cfg.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
    cfg.mutable_cheetah_2pc_config()->set_approx_less_precision(4);
    auto ctx=makeCheetahProtocol(cfg,link);auto* ot=ctx->getState<CheetahOTState>();
    ASSERT_TRUE(ctx->hasKernel("cdi_material_prepare"));
    auto quotas=ring_zeros(FM64,{static_cast<int64_t>(ot->maximum_instances())+1,2});
    auto q=NdArrayView<uint64_t>(quotas);q[0]=q[1]=8192;q[2*ot->maximum_instances()]=4096;
    const auto epoch=int64_t{42};
    dynDispatch<std::vector<Value>>(ctx.get(),"cdi_material_prepare",Value(quotas.as(makeType<Pub2kTy>(FM64)),DT_I64),epoch);
    auto x=ring_zeros(FM64,{8}),ab=ring_zeros(FM64,{8,2}),route=ring_zeros(FM64,{8}),ks=ring_zeros(FM64,{8});
    for(int i=0;i<8;++i){ks.at<uint64_t>(i)=2;if(link->Rank()==0){x.at<uint64_t>(i)=2*(1<<18);ab.at<uint64_t>(2*i)=uint64_t(-int64_t(1<<15));ab.at<uint64_t>(2*i+1)=3*(1<<16);}}
    auto xv=Value(x.as(makeType<AShrTy>(FM64)),DT_F32);
    auto scope=[&](std::string action){return dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{xv},std::string("inference"),epoch,action);};
    scope("begin");
    auto out=dynDispatch<std::vector<Value>>(ctx.get(),"cdi_invert",xv,
        Value(ab.as(makeType<AShrTy>(FM64)),DT_F32),Value(route.as(makeType<Pub2kTy>(FM64)),DT_I64),
        Value(ks.as(makeType<Pub2kTy>(FM64)),DT_I64),std::string("reciprocal"));
    outputs[link->Rank()]=out[0].data();scope("end");
    EXPECT_GT(ctx->getState<CheetahMulState>()->Resources().stock_served,0);
    EXPECT_GT(ot->get()->GetSenderCOT()->Resources().stock_served+ot->get()->GetReceiverCOT()->Resources().stock_served,0);
    EXPECT_ANY_THROW(scope("begin"));
    EXPECT_ANY_THROW(dynDispatch<std::vector<Value>>(ctx.get(),"cdi_material_prepare",Value(quotas.as(makeType<Pub2kTy>(FM64)),DT_I64),epoch));
    auto release=[&](){return dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{xv},std::string("materials"),epoch,std::string("release"));};
    EXPECT_NO_THROW(release());
    EXPECT_EQ(ot->get()->GetSenderCOT()->Resources().stock_remaining,0);
    EXPECT_EQ(ot->get()->GetReceiverCOT()->Resources().stock_remaining,0);
    EXPECT_EQ(ctx->getState<CheetahMulState>()->Resources().stock_remaining,0);
    EXPECT_ANY_THROW(release());
    EXPECT_ANY_THROW(scope("begin"));
  });
  auto result=ring_add(outputs[0],outputs[1]);
  for(int i=0;i<8;++i)EXPECT_LE(std::abs(static_cast<int64_t>(result.at<uint64_t>(i))-(1<<17)),16);
}
}
