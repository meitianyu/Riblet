#include <array>
#include <cstdlib>
#include <iostream>
#include "gtest/gtest.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/cdi_material.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
RuntimeConfig CompactConfig(){
  RuntimeConfig c;c.set_protocol(CHEETAH);c.set_field(FM64);c.set_fxp_fraction_bits(18);
  c.set_fxp_div_goldschmidt_iters(2);
  c.mutable_cheetah_2pc_config()->set_ot_kind(CheetahOtKind::YACL_Ferret);
  c.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
  c.mutable_cheetah_2pc_config()->set_approx_less_precision(4);return c;
}
TEST(CompactMaterial, TypedRandomOtCensusAndOriginalRelation){
  std::array<std::vector<uint8_t>,2> first,second,choices,received;
  for(size_t w:{1,2}){size_t n=w==1?65:97;first[w-1].resize(n);second[w-1].resize(n);choices[w-1].resize(n);received[w-1].resize(n);}
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    auto ctx=makeCheetahProtocol(CompactConfig(),link);
    auto* state=ctx->getState<CheetahOTState>();state->LazyInit(ctx->getState<Communicator>());auto base=state->get();
    for(size_t w:{1,2}){
      if(link->Rank()==0)base->GetSenderCOT()->SendRMRC(absl::MakeSpan(first[w-1]),absl::MakeSpan(second[w-1]),w);
      else base->GetReceiverCOT()->RecvRMRC(absl::MakeSpan(choices[w-1]),absl::MakeSpan(received[w-1]),w);
    }
    auto u=(link->Rank()==0?base->GetSenderCOT():base->GetReceiverCOT())->Resources();
    EXPECT_EQ(u.requested,162);EXPECT_EQ(u.rmrc1_requested,65);EXPECT_EQ(u.rmrc2_requested,97);
  });
  for(size_t w:{1,2})for(size_t i=0;i<received[w-1].size();++i)
    EXPECT_EQ(received[w-1][i],choices[w-1][i]?second[w-1][i]:first[w-1][i]);
}
TEST(CompactMaterial, ActualLongestGeluOperandShapeCensus){
  const int64_t length=std::getenv("CDI_CENSUS_LENGTH")?std::stoll(std::getenv("CDI_CENSUS_LENGTH")):65;
  SPU_ENFORCE(length>=10 && length<=65);
  const int64_t n=length*3072;
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    auto ctx=makeCheetahProtocol(CompactConfig(),link);
    auto data=ring_zeros(FM64,{n});if(link->Rank()==0)for(int64_t i=0;i<n;++i)data.at<uint64_t>(i)=1<<17;
    auto result=kernel::hal::f_erf(ctx.get(),Value(data.as(makeType<AShrTy>(FM64)),DT_F32));
    EXPECT_EQ(result.numel(),n);
    auto* state=ctx->getState<CheetahOTState>();uint64_t raw=0,one=0,two=0;
    for(size_t i=0;i<state->maximum_instances();++i)if(auto lane=state->Existing(i)){
      size_t direction=0;
      for(auto cot:{lane->GetSenderCOT(),lane->GetReceiverCOT()}){
        auto u=cot->Resources();raw+=u.requested;one+=u.rmrc1_requested;two+=u.rmrc2_requested;
        SPDLOG_INFO("GELU_CENSUS_SLOT rank={} slot={} raw={} rmrc1={} rmrc2={} length={}",link->Rank(),2*i+direction,u.requested,u.rmrc1_requested,u.rmrc2_requested,length);++direction;
      }
    }
    std::cout<<"GELU_CENSUS rank="<<link->Rank()<<" elements="<<n<<" raw="<<raw<<" rmrc1="<<one<<" rmrc2="<<two<<" residual="<<raw-one-two<<"\n";
    EXPECT_EQ(raw,uint64_t(2641)*n);EXPECT_LE(one+two,raw);
  });
}
TEST(CompactMaterial, PreparedCompactRandomOtIsFreshStrictAndNeverGeneratesOnline){
  std::array<std::vector<uint8_t>,2> first,second,choices,received;
  for(size_t w:{1,2}){size_t n=w==1?65:97;first[w-1].resize(n);second[w-1].resize(n);choices[w-1].resize(n);received[w-1].resize(n);}
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    auto ctx=makeCheetahProtocol(CompactConfig(),link);auto* state=ctx->getState<CheetahOTState>();
    state->LazyInit(ctx->getState<Communicator>());auto base=state->get();
    auto cot=link->Rank()==0?base->GetSenderCOT():base->GetReceiverCOT();
    cot->PrepareCompactMaterials(0,162);
    auto before=cot->Resources();EXPECT_EQ(before.rmrc_stock_remaining,162);
    EXPECT_EQ(before.stock_resident_bytes,162);
    for(size_t w:{1,2}){
      if(link->Rank()==0)cot->SendRMRC(absl::MakeSpan(first[w-1]),absl::MakeSpan(second[w-1]),w);
      else cot->RecvRMRC(absl::MakeSpan(choices[w-1]),absl::MakeSpan(received[w-1]),w);
    }
    auto after=cot->Resources();EXPECT_EQ(after.generated,before.generated);
    EXPECT_EQ(after.requested,before.requested);EXPECT_EQ(after.rmrc_stock_remaining,0);
    EXPECT_EQ(after.rmrc1_served,65);EXPECT_EQ(after.rmrc2_served,97);
    EXPECT_EQ(after.stock_resident_bytes,0);
    std::vector<uint8_t> a(1),b(1);
    if(link->Rank()==0)EXPECT_ANY_THROW(cot->SendRMRC(absl::MakeSpan(a),absl::MakeSpan(b),1));
    else EXPECT_ANY_THROW(cot->RecvRMRC(absl::MakeSpan(a),absl::MakeSpan(b),1));
    cot->ReleasePreparedInventory();
  });
  for(size_t w:{1,2})for(size_t i=0;i<received[w-1].size();++i)
    EXPECT_EQ(received[w-1][i],choices[w-1][i]?second[w-1][i]:first[w-1][i]);
}
TEST(CompactMaterial, TypedPreparationRejectsUndersupplyBeforeGeneratingThenRunsGelu){
  std::array<std::vector<uint8_t>,3> a,b,c,z;
  for(size_t j=0;j<3;++j){a[j].resize(16);b[j].resize(16);c[j].resize(16);z[j].resize(16);}
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    auto ctx=makeCheetahProtocol(CompactConfig(),link);auto* ot=ctx->getState<CheetahOTState>();
    auto data=ring_zeros(FM64,{128});if(link->Rank()==0)for(int i=0;i<128;++i)data.at<uint64_t>(i)=1<<17;
    auto input=Value(data.as(makeType<AShrTy>(FM64)),DT_F32);
    auto scope=[&](int64_t id,std::string action){return dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{input},std::string("inference"),id,action);};
    scope(1,"begin");kernel::hal::f_erf(ctx.get(),input);scope(1,"end");
    auto q=ring_zeros(FM64,{static_cast<int64_t>(ot->maximum_instances())+1,4});
    auto v=NdArrayView<uint64_t>(q);
    for(size_t j=0;j<4;++j){v[4*j]=v[4*j+1]=65536;v[4*j+2]=v[4*j+3]=524288;}
    auto prepare=[&](){return dynDispatch<std::vector<Value>>(ctx.get(),"cdi_material_prepare",Value(q.as(makeType<Pub2kTy>(FM64)),DT_I64),int64_t{2});};
    auto generated=ctx->getState<CheetahMulState>()->Resources().generated;
    EXPECT_ANY_THROW(prepare()); 
    EXPECT_EQ(generated,ctx->getState<CheetahMulState>()->Resources().generated);
    v[4*ot->maximum_instances()]=8192;
    EXPECT_NO_THROW(prepare());
    scope(2,"begin");auto out=kernel::hal::f_erf(ctx.get(),input);
    for(size_t j=0;j<3;++j){
      auto bits=ring_zeros(FM64,{16}).as(makeType<BShrTy>(FM64,1));
      auto check=ot->get(j+1)->BitwiseAnd(bits,bits);
      EXPECT_EQ(check.numel(),16);
    }
    scope(2,"end");
    EXPECT_EQ(out.numel(),128);
    auto u=ot->get()->GetSenderCOT()->Resources();
    EXPECT_GT(u.rmrc1_served+u.rmrc2_served,0);
    dynDispatch<std::vector<Value>>(ctx.get(),"cdi_scope",std::vector<Value>{input},std::string("materials"),int64_t{2},std::string("release"));
    EXPECT_EQ(ot->get()->GetSenderCOT()->Resources().rmrc_stock_remaining,0);
  });
}
TEST(CompactMaterial, PreparedBooleanRecordsSatisfyBothRelationsWithoutInlineGeneration){
  std::array<std::array<NdArrayRef,3>,2> normal;
  std::array<std::array<NdArrayRef,5>,2> correlated;
  utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
    auto ctx=makeCheetahProtocol(CompactConfig(),link);auto* state=ctx->getState<CheetahOTState>();
    state->LazyInit(ctx->getState<Communicator>());auto lane=state->get();
    if(link->Rank()==0){lane->GetSenderCOT()->PrepareCompactMaterials(0,48);lane->GetReceiverCOT()->PrepareCompactMaterials(0,48);}
    else{lane->GetReceiverCOT()->PrepareCompactMaterials(0,48);lane->GetSenderCOT()->PrepareCompactMaterials(0,48);}
    lane->PrepareBooleanFromRandomChoices(48);
    EXPECT_EQ(lane->BooleanStockBytes(),48);
    auto before=lane->GetSenderCOT()->Resources().generated;
    normal[link->Rank()]=lane->AndTriple(FM64,{2},16);
    correlated[link->Rank()]=lane->CorrelatedAndTriple(FM64,{16});
    EXPECT_EQ(lane->GetSenderCOT()->Resources().generated,before);
    EXPECT_EQ(lane->BooleanStockBytes(),0);
    EXPECT_ANY_THROW(lane->AndTriple(FM64,{1},1));
    std::vector<uint8_t> x(1),y(1);
    EXPECT_ANY_THROW(lane->GetSenderCOT()->SendRMRC(absl::MakeSpan(x),absl::MakeSpan(y),1));
    lane->ReleaseBooleanInventory();
    EXPECT_EQ(lane->BooleanStockBytes(),0);
    lane->GetSenderCOT()->ReleasePreparedInventory();
    lane->GetReceiverCOT()->ReleasePreparedInventory();
    if(link->Rank()==0){lane->GetSenderCOT()->PrepareCompactMaterials(0,8);lane->GetReceiverCOT()->PrepareCompactMaterials(0,8);}
    else{lane->GetReceiverCOT()->PrepareCompactMaterials(0,8);lane->GetSenderCOT()->PrepareCompactMaterials(0,8);}
    lane->PrepareBooleanFromRandomChoices(8);
    EXPECT_EQ(lane->BooleanStockBytes(),8);
    EXPECT_NO_THROW(lane->AndTriple(FM64,{1},8));
    EXPECT_EQ(lane->BooleanStockBytes(),0);
    lane->ReleaseBooleanInventory();
  });
  auto a=ring_xor(normal[0][0],normal[1][0]);
  auto b=ring_xor(normal[0][1],normal[1][1]);
  auto c=ring_xor(normal[0][2],normal[1][2]);
  for(int i=0;i<2;++i)EXPECT_EQ(c.at<uint64_t>(i),a.at<uint64_t>(i)&b.at<uint64_t>(i));
  auto u=ring_xor(correlated[0][0],correlated[1][0]);
  for(int branch=0;branch<2;++branch){
    auto v=ring_xor(correlated[0][1+2*branch],correlated[1][1+2*branch]);
    auto w=ring_xor(correlated[0][2+2*branch],correlated[1][2+2*branch]);
    for(int i=0;i<16;++i)EXPECT_EQ(w.at<uint64_t>(i),u.at<uint64_t>(i)&v.at<uint64_t>(i));
  }
}
}
