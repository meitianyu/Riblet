#include "libspu/core/cpp_prepare_observer.h"

#pragma once
#include "libspu/device/pphlo/vit_cdi_inverse.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/common/pv2k.h"

namespace spu::device::pphlo::ours_vit {
namespace h=kernel::hal;

struct RestoreFormat {
 SPUContext* ctx;FieldType field;int bits;
 explicit RestoreFormat(SPUContext* c):ctx(c),field(c->getField()),bits(c->getFxpBits()){}
 ~RestoreFormat(){
  const_cast<RuntimeConfig*>(&ctx->config())->set_field(field);
  const_cast<RuntimeConfig*>(&ctx->config())->set_fxp_fraction_bits(bits);
  ctx->getState<mpc::Z2kState>()->setField(field);
 }
};
inline void Check(SPUContext* ctx){
 SPU_ENFORCE(ctx->config().protocol()==CHEETAH&&ctx->getField()==FM64&&ctx->getFxpBits()==18,
             "Ours ViT inverse external ABI requires CHEETAH/FM64/f18");
}
inline Value PublicWords(SPUContext* ctx,const std::array<uint64_t,12>& words){
 auto seed=h::_constant(ctx,0,{12}).setDtype(DT_I64);
 auto out=Value(seed.data().clone(),DT_I64);auto v=NdArrayView<uint64_t>(out.data());
 for(int i=0;i<12;++i) { v[i]=words[i]; }
 return out;
}
inline Value Status(SPUContext* ctx){
 auto* m=ctx->getState<mpc::cheetah::CheetahMulState>();
 std::array<uint64_t,12> s{m->BenchUsed(),m->BenchRemaining(FM64),m->BenchRemaining(FM32),
     uint64_t(ctx->getField()),uint64_t(ctx->getFxpBits()),uint64_t(m->BenchArmed())};
 if(m->BenchArmed()) {
  auto* ot=ctx->getState<mpc::cheetah::CheetahOTState>();
  for(size_t i=0;i<ot->maximum_instances();++i)
   for(auto p:{ot->get(i)->GetSenderCOT(),ot->get(i)->GetReceiverCOT()}) {
    auto v=p->BenchStats();s[6]=std::max(s[6],v[0]);s[7]=std::max(s[7],v[1]);
    s[8]+=v[0];s[9]+=v[1];s[10]+=v[2];s[11]+=v[3];
   }
 }
 return PublicWords(ctx,s);
}
inline Value Retire(SPUContext* ctx){

spu::lifecycle::Scope scope_vit_material_release("ViT.material-release", "A");
scope_vit_material_release.Outcome("retired");

 auto* m=ctx->getState<mpc::cheetah::CheetahMulState>();
 SPU_ENFORCE(m->BenchArmed(),"material already retired");
 m->BenchRelease();return Status(ctx);
}
inline Value Inverse(SPUContext* ctx,const Value& x,const Value& ab,const Value& r,const Value& k,const std::string& op){
 Check(ctx);SPU_ENFORCE(ctx->getState<mpc::cheetah::CheetahMulState>()->BenchArmed(),"unprepared inverse");
 SPU_ENFORCE(op=="reciprocal"||op=="rsqrt");

 return op=="reciprocal"?vit_cdi::Reciprocal(ctx,x,ab,r,k):vit_cdi::RSqrt(ctx,x,ab,r,k);
}
}

#include "libspu/device/pphlo/vit_lane_quota.h"
namespace spu::device::pphlo::ours_vit {
inline Value PrepareLanes(SPUContext* ctx,const Value& value) {

spu::lifecycle::Scope scope_vit_complete_material_prepare_v2("ViT.complete-material-prepare-v2", "unclassified");
scope_vit_complete_material_prepare_v2.Outcome("prepared");

  Check(ctx);
  SPU_ENFORCE(ctx->config().max_concurrency()==64 &&
              !ctx->config().experimental_enable_intra_op_par(),
              "bound64 worker resource contract required");
  SPU_ENFORCE(value.isPublic() && value.dtype()==DT_I64);
  auto* ot=ctx->getState<mpc::cheetah::CheetahOTState>();
  SPU_ENFORCE(ot->maximum_instances()==48,"actual OT lane count differs");
  auto words=NdArrayView<const uint64_t>(value.data());
  std::vector<uint64_t> flat;
  for(int64_t i=0;i<value.numel();++i)flat.push_back(words[i]);
  const auto q=::spu::vit_resource::Parse(flat,ot->maximum_instances());
  auto* m=ctx->getState<mpc::cheetah::CheetahMulState>();
  SPU_ENFORCE(!m->BenchArmed(),"release previous request material first");
  auto* comm=ctx->getState<mpc::Communicator>();
  m->BenchPrepare(FM64,q.beaver64);m->BenchPrepare(FM32,q.beaver32);
  ot->SetStrictInventory(false);
  for(size_t i=0;i<q.lanes.size();++i) {
    ot->LazyInit(comm,i);auto basic=ot->get(i);
    auto first=comm->getRank()==0?basic->GetSenderCOT():basic->GetReceiverCOT();
    auto second=comm->getRank()==0?basic->GetReceiverCOT():basic->GetSenderCOT();
    first->BenchPrepare(q.lanes[i][0],q.lanes[i][1]);
    second->BenchPrepare(q.lanes[i][0],q.lanes[i][1]);
  }
  ot->SetStrictInventory(true);
  return Status(ctx);
}
inline Value LaneStatus(SPUContext* ctx) {
  Check(ctx);
  auto* ot=ctx->getState<mpc::cheetah::CheetahOTState>();
  const auto lanes=ot->maximum_instances();
  auto base=Status(ctx);auto head=NdArrayView<const uint64_t>(base.data());
  auto seed=h::_constant(ctx,uint64_t{0},{static_cast<int64_t>(12+4*lanes)}).setDtype(DT_I64);
  auto out=Value(seed.data().clone(),DT_I64);auto dst=NdArrayView<uint64_t>(out.data());
  for(int i=0;i<12;++i)dst[i]=head[i];
  for(size_t i=0;i<lanes;++i) {
    auto sender=ot->get(i)->GetSenderCOT()->BenchStats();
    auto receiver=ot->get(i)->GetReceiverCOT()->BenchStats();
    dst[12+4*i]=sender[0];dst[13+4*i]=sender[1];
    dst[14+4*i]=receiver[0];dst[15+4*i]=receiver[1];
  }
  return out;
}
}
