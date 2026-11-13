#include "libspu/core/cpp_prepare_observer.h"

// Copyright 2026

#include "libspu/mpc/cheetah/cdi_material.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
namespace {
std::vector<MaterialUsage> ResourceSnapshot(SPUContext* ctx) {
  if(!ctx)return {};
  auto* ot=ctx->getState<CheetahOTState>();
  std::vector<MaterialUsage> result(2*ot->maximum_instances()+1);
  for(size_t i=0;i<ot->maximum_instances();++i) {
    if(auto lane=ot->Existing(i)) {
      result[2*i]=lane->GetSenderCOT()->Resources();
      result[2*i+1]=lane->GetReceiverCOT()->Resources();
    }
  }
  result.back()=ctx->getState<CheetahMulState>()->Resources();
  return result;
}
}

void CdiTimingState::Observe(const std::string& scope, int64_t site,
                             const std::string& action, SPUContext* ctx) {
  if(scope=="materials" && action=="release"){ReleaseMaterials(site,ctx);return;}
  if (scope == "barrier" && action == "barrier") return;
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(scope == "inference" || scope == "softmax" ||
                  scope == "layernorm" || scope == "reciprocal" ||
                  scope == "rsqrt" || scope=="gelu_erf" || scope=="equal", "unknown CDI timing scope");
  if (action == "begin") {
    if (scope == "inference") {
      SPU_ENFORCE(stack_.empty(), "nested inference");next_gelu_=0;next_equal_=0;
      if(!material_epochs_.empty()){SPU_ENFORCE(site==material_epoch_ && !material_consumed_,"prepared materials request binding/reuse failure");material_consumed_=true;}
    }
    auto resources=ResourceSnapshot(ctx);
    std::array<MaterialUsage,2> beavers{};
    if(ctx)beavers={ctx->getState<CheetahMulState>()->Resources(FM64),ctx->getState<CheetahMulState>()->Resources(FM32)};
    stack_.push_back({scope, site, std::chrono::steady_clock::now(),MaterialNowNs(),std::move(resources),std::move(beavers)});
    return;
  }
  SPU_ENFORCE(action == "end" && !stack_.empty(), "unmatched CDI timing end");
  SPU_ENFORCE(stack_.back().scope == scope && stack_.back().site == site,
              "CDI timing nesting mismatch: {}:{} vs {}:{}",
              stack_.back().scope, stack_.back().site, scope, site);
  const int64_t request = stack_.front().scope == "inference" ? stack_.front().site : -1;
  auto entry = std::move(stack_.back());
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - entry.start).count();
  const auto end_ns=MaterialNowNs();
  stack_.pop_back();
  SPDLOG_INFO("CDI_SCOPE request={} op={} site={} ns={} parent={}",
              request, scope, site, ns, stack_.empty() ? "none" : stack_.back().scope);
  auto after=ResourceSnapshot(ctx);
  if(scope=="inference" && ctx){
    auto* mul=ctx->getState<CheetahMulState>();
    last_beaver_requirements_={mul->Resources(FM64).requested-entry.beavers[0].requested,
                              mul->Resources(FM32).requested-entry.beavers[1].requested};
    SPDLOG_INFO("CDI_MATERIAL_RINGS request={} fm64_requested={} fm32_requested={}",request,last_beaver_requirements_[0],last_beaver_requirements_[1]);
  }
  uint64_t ot_requested=0,beaver_requested=0,generated_calls=0,served=0;
  uint64_t small_requested=0,small_served=0;
  if(scope=="inference")last_requirements_.assign(after.size(),{0,0});
  int64_t sum_ns=0;std::vector<GenerationInterval> intervals;
  for(size_t i=0;i<after.size();++i){
    const MaterialUsage before=i<entry.resources.size()?entry.resources[i]:MaterialUsage{};
    const auto& value=after[i];const auto demand=value.requested-before.requested;
    const auto sr=value.rmrc1_requested+value.rmrc2_requested-before.rmrc1_requested-before.rmrc2_requested;
    const auto ss=value.rmrc1_served+value.rmrc2_served-before.rmrc1_served-before.rmrc2_served;
    small_requested+=sr;small_served+=ss;
    SPU_ENFORCE(sr>=ss && demand>=sr-ss,"typed material accounting mismatch");
    if(scope=="inference"){
      last_requirements_[i]={demand-(sr-ss),sr};
      SPDLOG_INFO("CDI_MATERIAL_TYPED_SLOT request={} slot={} raw_required={} rmrc_required={} rmrc_served={}",request,i,demand-(sr-ss),sr,ss);
    }
    if(i+1==after.size())beaver_requested=demand;else ot_requested+=demand;
    served+=value.stock_served-before.stock_served;
    for(auto interval:value.generations)if(interval.end>entry.start_ns && interval.begin<end_ns){
      ++generated_calls;sum_ns+=std::min(interval.end,end_ns)-std::max(interval.begin,entry.start_ns);intervals.push_back(interval);
    }
    if(scope=="inference")SPDLOG_INFO("CDI_MATERIAL_SLOT request={} slot={} requested={} served={} remaining={} generation_calls={}",
       request,i,demand,value.stock_served-before.stock_served,value.stock_remaining,value.generations.size()-before.generations.size());
  }
  SPDLOG_INFO("CDI_MATERIAL request={} op={} site={} wall_ns={} ot_requested={} beaver_requested={} generated_calls={} generator_sum_ns={} generator_union_ns={} stock_served={}",
      request,scope,site,ns,ot_requested,beaver_requested,generated_calls,sum_ns,GenerationUnionNs(intervals,entry.start_ns,end_ns),served);
  SPDLOG_INFO("CDI_COMPACT_MATERIAL request={} op={} site={} rmrc_requested={} rmrc_served={}",request,scope,site,small_requested,small_served);
  if(scope=="inference" && material_epoch_>=0)SPU_ENFORCE(generated_calls==0,"material generation occurred during prepared online execution");
}

void CdiTimingState::ReleaseMaterials(int64_t epoch,SPUContext* ctx) {

spu::lifecycle::Scope scope_bert_material_release("BERT.material-release", "A", [&] { return static_cast<uint64_t>((ctx->getState<Communicator>()->lctx())->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_bert_material_release.Outcome("retired");

  std::lock_guard guard(mutex_);
  SPU_ENFORCE(ctx && stack_.empty() && material_epoch_==epoch && material_consumed_,
              "material release requires the completed prepared request");
  auto* ot=ctx->getState<CheetahOTState>();uint64_t bytes=0;
  for(size_t i=0;i<ot->maximum_instances();++i)if(auto lane=ot->Existing(i)){
    bytes+=lane->BooleanStockBytes();lane->ReleaseBooleanInventory();
    for(const auto& cot:{lane->GetSenderCOT(),lane->GetReceiverCOT()}){
      bytes+=cot->Resources().stock_capacity*16;cot->ReleasePreparedInventory();
    }
  }
  ctx->getState<CheetahMulState>()->ReleasePreparedBeavers();
  ot->SetStrictInventory(false);material_epoch_=-1;
  SPDLOG_INFO("CDI_MATERIAL_RELEASE request={} released_ot_bytes={} model_state_preserved=true",epoch,bytes);
}

void CdiScope::evaluate(KernelEvalContext* ctx) const {
  SPU_ENFORCE(ctx->numParams() == 4, "CDI timing arguments incomplete");
  SPU_ENFORCE(ctx->sctx()->getField() == FM64 && ctx->sctx()->getFxpBits() == 18,
              "CDI compiled ABI requires FM64/fxp18");
  const auto& values = ctx->getParam<std::vector<Value>>(0);
  const auto& scope = ctx->getParam<std::string>(1);
  const auto site = ctx->getParam<int64_t>(2);
  const auto& action = ctx->getParam<std::string>(3);
  SPU_ENFORCE(!values.empty() && site >= 0, "invalid CDI timing boundary");
  ctx->getState<CdiTimingState>()->Observe(scope, site, action,ctx->sctx());
  ctx->pushOutput(values);
}

void CdiMaterialPrepare::evaluate(KernelEvalContext* ctx) const {

spu::lifecycle::Scope scope_bert_complete_material_prepare("BERT.complete-material-prepare", "unclassified", [&] { return static_cast<uint64_t>((ctx->getState<Communicator>()->lctx())->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_bert_complete_material_prepare.Outcome("prepared");

  SPU_ENFORCE(ctx->numParams()==2,"incomplete material preparation request");
  const auto& quotas=ctx->getParam<Value>(0);const auto epoch=ctx->getParam<int64_t>(1);
  SPU_ENFORCE(quotas.isPublic() && quotas.dtype()==DT_I64,"material quotas must be public int64");
  SPU_ENFORCE(ctx->sctx()->getField()==FM64 && ctx->sctx()->getFxpBits()==18,"material ABI requires FM64/f18");
  SPU_ENFORCE(ctx->sctx()->config().cheetah_2pc_config().ot_kind()==CheetahOtKind::YACL_Ferret,"prepared material path requires the frozen YACL Ferret backend");
  auto* ot=ctx->getState<CheetahOTState>();const auto lanes=ot->maximum_instances();
  SPU_ENFORCE(quotas.shape().ndim()==2 && (quotas.shape()[0]==static_cast<int64_t>(lanes)+1 || quotas.shape()[0]==static_cast<int64_t>(lanes)+2) && (quotas.shape()[1]==2 || quotas.shape()[1]==4),"quota shape mismatch");
  const size_t cols=quotas.shape()[1];
  const auto values=NdArrayView<const uint64_t>(quotas.data());
  const bool mixed=quotas.shape()[0]==static_cast<int64_t>(lanes)+2;
  if(mixed){
    SPU_ENFORCE(cols==4 && values[cols*(lanes+1)]==2,"unknown mixed-ring quota version");
    SPU_ENFORCE(values[cols*lanes+1]==0 && values[cols*lanes+2]==0 && values[cols*lanes+3]==0 &&
                values[cols*(lanes+1)+2]==0 && values[cols*(lanes+1)+3]==0,"nonzero quota reserved fields");
  }
  uint64_t total_ot=0,total_small=0,total_boolean=0,largest=0;
  for(size_t i=0;i<lanes;++i)for(size_t j=0;j<cols;++j){
    SPU_ENFORCE(values[cols*i+j]<(uint64_t{1}<<32),"OT quota too large");
    if(j<2){total_ot+=values[cols*i+j];largest=std::max(largest,values[cols*i+j]*16);}
    else total_small+=values[cols*i+j];
  }
  const auto beavers=values[cols*lanes];
  const auto beavers32=mixed?values[cols*(lanes+1)+1]:0;
  if(cols==4)for(size_t i=0;i<lanes;++i)total_boolean+=std::min(values[cols*i+2],values[cols*i+3]);
  SPU_ENFORCE(beavers<(uint64_t{1}<<32) && beavers32<(uint64_t{1}<<32),"Beaver quota too large");
  auto& storage=InventoryStoragePolicy::Get();
  if(cols==4){
    SPU_ENFORCE(!storage.Bounded(),"compact profile forbids disk material storage");
    SPU_ENFORCE(total_ot*16+total_small+beavers*24+beavers32*12<(72ULL<<30),"compact RAM quota exceeds budget");
  }else if(storage.Bounded()){
    storage.Admit(total_ot*16,largest,beavers*24);
  }else{
    SPU_ENFORCE(total_ot*16+beavers*24<80ULL*1024*1024*1024,"prepared inventory exceeds memory safety cap");
  }
  auto* comm=ctx->getState<Communicator>();const auto rank=comm->getRank();
  if(cols==4){
    auto need=ctx->getState<CdiTimingState>()->LastRequirements();
    if(!need.empty()){
      SPU_ENFORCE(need.size()==2*lanes+1,"warmup material layout changed");
      auto rings=ctx->getState<CdiTimingState>()->LastBeaverRequirements();
      SPU_ENFORCE(beavers>=rings[0] && beavers32>=rings[1],"per-ring Beaver quota below existing demand");
      for(size_t i=0;i<lanes;++i)for(size_t j=0;j<2;++j){
        const auto direction=rank==0?j:1-j;
        SPU_ENFORCE(values[cols*i+direction]>=need[2*i+j][0],"raw COT quota below warmup demand at lane {}",i);
        SPU_ENFORCE(values[cols*i+2+direction]>=need[2*i+j][1],"compact ROT quota below warmup demand at lane {}",i);
      }
    }
  }
  const auto start=MaterialNowNs();
  ctx->getState<CdiTimingState>()->ArmMaterials(epoch);
  ctx->getState<CheetahMulState>()->ReserveBeavers(FM64,beavers);
  if(mixed)ctx->getState<CheetahMulState>()->ReserveBeavers(FM32,beavers32);
  ot->SetStrictInventory(false);
  std::vector<size_t> active_lanes;
  for(size_t i=0;i<lanes;++i){
    bool active=false;for(size_t j=0;j<cols;++j)active|=values[cols*i+j]!=0;
    if(!active && !ot->Existing(i))continue;
    ot->LazyInit(comm,i);active_lanes.push_back(i);
  }
  auto prepare_lane=[&](size_t i){
    auto lane=ot->get(i);

    auto prepare=[&](auto cot,size_t j){
      if(cols==4)cot->PrepareCompactMaterials(values[cols*i+j],values[cols*i+2+j]);
      else cot->PrepareRandomCots(values[cols*i+j]);
    };
    if(rank==0){prepare(lane->GetSenderCOT(),0);prepare(lane->GetReceiverCOT(),1);}
    else{prepare(lane->GetReceiverCOT(),0);prepare(lane->GetSenderCOT(),1);}
    if(cols==4)lane->PrepareBooleanFromRandomChoices(std::min(values[cols*i+2],values[cols*i+3]));
  };

  const size_t parallel=cols==4?4:1;
  for(size_t first=0;first<active_lanes.size();first+=parallel){
    std::vector<std::future<void>> work;
    for(size_t j=first;j<std::min(first+parallel,active_lanes.size());++j)
      work.emplace_back(std::async(std::launch::async,prepare_lane,active_lanes[j]));
    for(auto& task:work)task.get();
  }
  ot->SetStrictInventory(true);
  uint64_t resident=0,disk=0;
  for(size_t i=0;i<lanes;++i)if(auto lane=ot->Existing(i)){
    resident+=lane->BooleanStockBytes();
    for(const auto& cot:{lane->GetSenderCOT(),lane->GetReceiverCOT()}){
      const auto usage=cot->Resources();resident+=usage.stock_resident_bytes;disk+=usage.stock_disk_bytes;
    }
  }
  SPDLOG_INFO("CDI_STOCK_STORAGE request={} ram_limit={} resident_bytes={} disk_bytes={} compact_random_ot={}",epoch,cols==4?(72ULL<<30):storage.Limit(),resident,disk,cols==4);
  SPDLOG_INFO("CDI_MATERIAL_PREPARE request={} ns={} ot_capacity={} beaver_capacity={} requested_bytes={} compact_rot_capacity={} boolean_pair_capacity={}",epoch,MaterialNowNs()-start,total_ot,beavers+beavers32,total_ot*16+total_boolean+beavers*24+beavers32*12,total_small,total_boolean);
  if(mixed)SPDLOG_INFO("CDI_MATERIAL_QUOTA_V2 request={} fm64={} fm32={}",epoch,beavers,beavers32);
  ctx->pushOutput(std::vector<Value>{Value(ring_zeros(FM64,{1}).as(makeType<Pub2kTy>(FM64)),DT_I64)});
}

}
