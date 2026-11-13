#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "spdlog/spdlog.h"
#include "libspu/core/prelude.h"
#include "libspu/core/context.h"
#include "libspu/kernel/hal/fxp_base.h"
#include "libspu/kernel/hal/fxp_approx.h"
#include "libspu/kernel/hal/ring.h"
#include "libspu/kernel/hal/shape_ops.h"

namespace spu::device::pphlo::vit_cdi {
namespace h=kernel::hal;
inline Value Gather(const Value& v,const std::vector<int64_t>& indices,int column=-1) {
  NdArrayRef out(v.storage_type(),{static_cast<int64_t>(indices.size())});
  auto dst=NdArrayView<uint64_t>(out);auto src=NdArrayView<const uint64_t>(v.data());
  for(size_t j=0;j<indices.size();++j) dst[j]=src[column<0?indices[j]:2*indices[j]+column];
  return Value(std::move(out),v.dtype());
}
inline Value Scatter(const Value& base,const Value& value,const std::vector<int64_t>& ids) {
  auto out=base.data().clone();auto dst=NdArrayView<uint64_t>(out);
  auto src=NdArrayView<const uint64_t>(value.data());
  for(size_t j=0;j<ids.size();++j)dst[ids[j]]=src[j];
  return Value(std::move(out),base.dtype());
}
inline Value RoundTo18(SPUContext* ctx,const Value& y){
 auto half=h::_constant(ctx,8,y.shape()).setDtype(DT_F32);
 return h::_prefer_a(ctx,h::_arshift(ctx,h::_add(ctx,y,half).setDtype(DT_F32),4)).setDtype(DT_F32);
}
inline Value RSqrt(SPUContext* ctx,const Value& x0,const Value& ab0,const Value& routes,const Value& ks){
 SPU_ENFORCE(x0.isSecret()&&ab0.isSecret()&&x0.dtype()==DT_F32&&ab0.dtype()==DT_F32);
 SPU_ENFORCE(routes.isPublic()&&ks.isPublic()&&routes.dtype()==DT_I64&&ks.dtype()==DT_I64);
 int64_t n=x0.numel();SPU_ENFORCE(ab0.numel()==2*n&&routes.numel()==n&&ks.numel()==n);
 auto x=h::reshape(ctx,x0,{n});auto ab=h::reshape(ctx,ab0,{n,2});
 auto r=NdArrayView<const uint64_t>(routes.data()),k=NdArrayView<const uint64_t>(ks.data());
 std::vector<int64_t> normal,fallback,counts;
 for(int64_t i=0;i<n;++i){SPU_ENFORCE(r[i]<=1&&k[i]<=5);if(r[i])fallback.push_back(i);else{normal.push_back(i);counts.push_back(k[i]);}}
 auto out=Value(x.data().clone(),DT_F32);
 if(!fallback.empty())out=Scatter(out,h::f_rsqrt(ctx,Gather(x,fallback)),fallback);
 if(normal.empty())return h::reshape(ctx,out,x0.shape()).setDtype(DT_F32);
 constexpr int bits=18;
 auto xx=Gather(x,normal),a=Gather(ab,normal,0),b=Gather(ab,normal,1);
 auto mul=[&](const Value& l,const Value& rr){return h::_trunc(ctx,h::_mul(ctx,l,rr).setDtype(DT_F32),bits,SignType::Unknown).setDtype(DT_F32);};
 auto y=h::_add(ctx,mul(a,xx),b).setDtype(DT_F32);
 for(int64_t step=1;step<=*std::max_element(counts.begin(),counts.end());++step){
  std::vector<int64_t> active;for(size_t j=0;j<normal.size();++j)if(counts[j]>=step)active.push_back(j);
  auto vx=Gather(xx,active),vy=Gather(y,active);
  auto product=mul(vx,mul(vy,vy));
  auto c=h::_constant(ctx,uint64_t{3}<<bits,vx.shape()).setDtype(DT_F32);
  auto tmp=mul(vy,h::_sub(ctx,c,product).setDtype(DT_F32));
  y=Scatter(y,h::_trunc(ctx,tmp,1,SignType::Unknown).setDtype(DT_F32),active);
 }
 out=Scatter(out,y,normal);return h::reshape(ctx,out,x0.shape()).setDtype(DT_F32);
}
inline Value Reciprocal(SPUContext* ctx,const Value& x0,const Value& ab0,const Value& routes,const Value& ks){
 SPU_ENFORCE(x0.isSecret()&&ab0.isSecret()&&x0.dtype()==DT_F32&&ab0.dtype()==DT_F32);
 SPU_ENFORCE(routes.isPublic()&&ks.isPublic()&&routes.dtype()==DT_I64&&ks.dtype()==DT_I64);
 const auto n=x0.numel();SPU_ENFORCE(n>0&&ab0.numel()==2*n&&routes.numel()==n&&ks.numel()==n);
 auto x=h::reshape(ctx,x0,{n});auto ab=h::reshape(ctx,ab0,{n,2});
 auto r=NdArrayView<const uint64_t>(routes.data()),k=NdArrayView<const uint64_t>(ks.data());
 std::vector<int64_t> normal,fallback,counts;
 for(int64_t i=0;i<n;++i){SPU_ENFORCE(r[i]<=1&&k[i]<=10);if(r[i])fallback.push_back(i);else{normal.push_back(i);counts.push_back(k[i]);}}
 auto out=Value(x.data().clone(),DT_F32);
 if(!fallback.empty())out=Scatter(out,h::f_reciprocal(ctx,Gather(x,fallback)),fallback);
 if(normal.empty())return h::reshape(ctx,out,x0.shape()).setDtype(DT_F32);
 constexpr int bits=22;
 auto xx=Gather(x,normal),a=Gather(ab,normal,0),b=Gather(ab,normal,1);
 xx=h::_lshift(ctx,xx,4).setDtype(DT_F32);a=h::_lshift(ctx,a,4).setDtype(DT_F32);b=h::_lshift(ctx,b,4).setDtype(DT_F32);
 auto mul=[&](const Value& l,const Value& rr){return h::_trunc(ctx,h::_mul(ctx,l,rr).setDtype(DT_F32),bits,SignType::Unknown).setDtype(DT_F32);};
 auto y=h::_add(ctx,mul(a,xx),b).setDtype(DT_F32);
 for(int64_t step=1;step<=*std::max_element(counts.begin(),counts.end());++step){
  std::vector<int64_t> active;for(size_t j=0;j<normal.size();++j){if(counts[j]>=step)active.push_back(j);}
  auto vx=Gather(xx,active),vy=Gather(y,active);
  auto two=h::_constant(ctx,uint64_t{2}<<bits,vx.shape()).setDtype(DT_F32);
  y=Scatter(y,mul(vy,h::_sub(ctx,two,mul(vx,vy)).setDtype(DT_F32)),active);
 }
 y=RoundTo18(ctx,y);
 out=Scatter(out,y,normal);return h::reshape(ctx,out,x0.shape()).setDtype(DT_F32);
}
}
