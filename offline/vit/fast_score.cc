#include "exact_predicate.cc"
#include <openssl/sha.h>
#include <charconv>
#include <cstring>
namespace vf {
constexpr uint64_t S=uint64_t{1}<<18;
int64_t si(uint64_t x){int64_t y;std::memcpy(&y,&x,8);return y;}
uint64_t word(uint64_t seed,uint64_t index,uint64_t domain=0){
 uint64_t z=seed+(index+1)*0x9E3779B97F4A7C15ULL+domain;
 z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);
}
struct W{int64_t v;uint64_t s;};
W constant(int64_t v){return {v,uint64_t(v)};}
W neg(W w){return {si(0-uint64_t(w.v)),0-w.s};}
struct C{
 uint64_t seed,counter=0,tn=0;bool bad=false;uint8_t error=0;int8_t first=-1;int k=0;
 void record(bool b,int e){if(b&&!error){error=e;first=k;}}
 void mark(bool b,int e){record(b,e);bad|=b;}
 W input(int64_t v){return {v,word(seed,counter++)};}
 W trunc(W w,int bits){
  mark(w.v<-(int64_t{1}<<62)||w.v>=(int64_t{1}<<62),6);
  uint64_t b=uint64_t(w.v)-w.s,a=w.s+(uint64_t{1}<<62)+(uint64_t{1}<<63);
  uint64_t carry=(a>>63)&(b>>63),bal=(a>>bits)&1,off=uint64_t{1}<<(63-bits),ho=uint64_t{1}<<(62-bits);
  uint64_t s=(a>>bits)-(word(seed,tn++,0xA0761D6478BD642FULL)<<(64-bits))-off+bal-ho;
  return {si((a>>bits)+(b>>bits)-(carry<<(64-bits))-off+bal-ho),s};
 }
 W mul(W a,W b){
  __int128 full=__int128(a.v)*b.v;
  mark(full<std::numeric_limits<int64_t>::min()||full>std::numeric_limits<int64_t>::max(),4);
  return trunc(input(si(uint64_t(a.v)*uint64_t(b.v))),18);
 }
 W add(W a,W b){
  int64_t v=si(uint64_t(a.v)+uint64_t(b.v));mark(((a.v^v)&(b.v^v))<0,5);
  return {v,a.s+b.s};
 }
};
}
extern "C" int vit_clear_trajectory(const int64_t* xs,const uint64_t* ids,int64_t n,
 int64_t av,int64_t bv,int op,int maximum,const char* prefix,int prefix_len,
 int64_t* ys,uint8_t* valid,uint8_t* design,uint8_t* diag,uint8_t* errors,uint8_t* first,int8_t* first_k){
 if(n<0||op<0||op>1||maximum<0||maximum>(op?9:10))return -1;
 SHA256_CTX base;SHA256_Init(&base);SHA256_Update(&base,prefix,prefix_len);
 #pragma omp parallel for if(n>8192)
 for(int64_t j=0;j<n;++j){
  char digits[32];auto end=std::to_chars(digits,digits+32,ids[j]).ptr;
  SHA256_CTX h=base;unsigned char digest[32];SHA256_Update(&h,digits,end-digits);SHA256_Final(digest,&h);
  uint64_t seed=0;for(int i=0;i<8;++i)seed|=uint64_t(digest[i])<<(8*i);
  vf::C c{seed};int64_t x=xs[j];c.record(x<=0,1);auto operand=c.input(x),a=c.input(av),b=c.input(bv);
  auto y=c.add(c.mul(a,operand),b);c.record(y.v<=0,2);
  unsigned __int128 upper=op?3*(unsigned __int128)vf::S*vf::S*vf::S:2*(unsigned __int128)vf::S*vf::S;
  bool inside=false;
  if(x>0&&y.v>0){unsigned __int128 q=(unsigned __int128)x*y.v;
   if(op){if(q<upper)q*=y.v;else q=upper;}
   inside=q>0&&q<upper;
  }
  c.record(!inside,3);bool active=x>0&&y.v>0&&inside&&!c.bad;
  for(int k=0;k<=maximum;++k){
   c.k=k;c.record(y.v<=0,k?7:2);active=active&&y.v>0&&!c.bad;auto index=int64_t(k)*n+j;
   ys[index]=y.v;valid[index]=active;errors[index]=active?0:c.error;
   design[index]=active&&(op?rsqrt_precision(x,y.v,12,false):reciprocal_precision(x,y.v,12,false));
   diag[index]=active&&(op?rsqrt_precision(x,y.v,13,false):reciprocal_precision(x,y.v,13,false));
   if(k==maximum)break;c.k=k+1;
   if(!op){auto t=c.mul(operand,y);auto corr=c.add(vf::constant(2*vf::S),vf::neg(t));y=c.mul(y,corr);}
   else{auto yy=c.mul(y,y);auto xyy=c.mul(operand,yy);auto corr=c.add(vf::constant(3*vf::S),vf::neg(xyy));y=c.trunc(c.mul(y,corr),1);}
  }
  first[j]=c.error;first_k[j]=c.first;
 }
 return 0;
}
