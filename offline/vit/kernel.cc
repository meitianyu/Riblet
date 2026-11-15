#include "fast_score.cc"
#include <cmath>
#include <algorithm>
namespace nv {
struct C: vf::C {
 int bits=18; bool nearest=false;
 vf::W shift(vf::W w,int n,bool round_nearest){
  auto out=vf::C::trunc(w,n);
  if(round_nearest){
   int64_t v=floor_shift64(w.v,n)+((uint64_t(w.v)>>(n-1))&1);
   out.s+=uint64_t(v)-uint64_t(out.v);out.v=v;
  }
  return out;
 }
 vf::W mul(vf::W a,vf::W b){
  __int128 raw=__int128(a.v)*b.v;
  mark(raw<std::numeric_limits<int64_t>::min()||raw>std::numeric_limits<int64_t>::max(),4);
  return shift(input(vf::si(uint64_t(a.v)*uint64_t(b.v))),bits,nearest);
 }
 vf::W lift(vf::W w,int n){
  __int128 v=__int128(w.v)*(int64_t{1}<<n);
  mark(v<std::numeric_limits<int64_t>::min()||v>std::numeric_limits<int64_t>::max(),4);
  return {vf::si(uint64_t(w.v)<<n),w.s<<n};
 }
};
uint64_t seed_for(SHA256_CTX base,uint64_t id){
 char digits[32];auto end=std::to_chars(digits,digits+32,id).ptr;unsigned char d[32];
 SHA256_Update(&base,digits,end-digits);SHA256_Final(d,&base);uint64_t seed=0;
 for(int i=0;i<8;++i)seed|=uint64_t(d[i])<<(8*i);return seed;
}
bool basin(int op,int64_t x,int64_t y,int bits){
 if(x<=0||y<=0)return false;
 uint64_t s=uint64_t{1}<<bits;
 if(!op)return __int128(x)*y<2*__int128(s)*s;
 return compare_u256(exact_product({uint64_t(x),uint64_t(y),uint64_t(y)}),exact_product({3,s,s,s}))<0;
}
struct Path {
 C c;vf::W operand,y;int64_t original;int op,flags,scale,bits;bool active;
 Path(uint64_t seed,int64_t x,int64_t a,int64_t b,int o,int f,int p,int s):original(x),op(o),flags(f),scale(s),bits(p){
  c.seed=seed;c.bits=p;c.nearest=flags&2;

  operand=c.input(x);auto aw=c.input(a),bw=c.input(b);
  c.record(x<=0,1);
  if(scale){operand=c.shift(operand,scale,false);aw=c.lift(aw,op?3*scale/2:2*scale);bw=c.lift(bw,op?scale/2:scale);}
  operand=c.lift(operand,bits-18);aw=c.lift(aw,bits-18);bw=c.lift(bw,bits-18);
  y=c.add(c.mul(aw,operand),bw);
  if(flags&8){ 
   long double xr=(long double)operand.v/(uint64_t{1}<<bits);
   int64_t v=llroundl((uint64_t{1}<<bits)/(op?sqrtl(xr):xr));
   y.s+=uint64_t(v)-uint64_t(y.v);y.v=v;
  }
  c.record(y.v<=0,2);bool inside=basin(op,operand.v,y.v,bits);c.record(!inside,3);
  active=x>0&&inside&&!c.bad;c.nearest=flags&4;
 }
 vf::W output(){

  C out=c;int n=bits-18+(op?scale/2:scale);
  vf::W result=n?out.shift(y,n,flags&16):y;
  if(out.bad){active=false;c.mark(true,out.error);}return result;
 }
 void next(int k){
  c.k=k;
  if(!op){auto xy=c.mul(operand,y);y=c.mul(y,c.add(vf::constant(int64_t{2}<<bits),vf::neg(xy)));}
  else{
   auto inner=(flags&1)?c.mul(operand,y):c.mul(y,y);
   auto term=(flags&1)?c.mul(inner,y):c.mul(operand,inner);
   y=c.shift(c.mul(y,c.add(vf::constant(int64_t{3}<<bits),vf::neg(term))),1,c.nearest);
  }
  c.record(y.v<=0,7);active=active&&y.v>0&&!c.bad;
 }
};
}
extern "C" int numeric_eval(const int64_t* x,const uint64_t* ids,int64_t n,int64_t a,int64_t b,
 int op,int maximum,const char* prefix,int plen,int flags,int bits,int scale,int aggregate,int certified,
 uint32_t* mask,uint8_t* initial,uint64_t* visited,uint64_t* failures,uint8_t* why,
 int64_t* ys,uint8_t* valid,uint8_t* passed,double* residual){
 if(n<1||maximum>20||maximum<0||(bits!=18&&bits!=20&&bits!=22)||scale<0||scale>8||(op&&scale%2))return -1;
 *mask=(1U<<(maximum+1))-1;*initial=1;*visited=0;
 for(int k=0;k<=maximum;++k){failures[k]=UINT64_MAX;why[k]=0;}
 SHA256_CTX base;SHA256_Init(&base);SHA256_Update(&base,prefix,plen);
 for(int64_t j=0;j<n;++j){
  if(aggregate&&(!*initial||(!*mask&&certified)))break;
  nv::Path p(nv::seed_for(base,ids[j]),x[j],a,b,op,flags,bits,scale);++*visited;
  if(!p.active){*initial=0;if(certified)return -3;}
  int last=maximum;if(aggregate&&*mask)last=31-__builtin_clz(*mask);
  for(int k=0;k<=last;++k){
   auto out=p.output();bool ok=p.active&&out.v>0;
   bool pass=ok&&(op?rsqrt_precision(x[j],out.v,12,false):reciprocal_precision(x[j],out.v,12,false));
   if(!aggregate){int64_t at=int64_t(k)*n+j;ys[at]=out.v;valid[at]=ok;passed[at]=pass;
    long double prod=op?sqrtl((long double)x[j]/vf::S)*out.v/vf::S:(long double)x[j]*out.v/vf::S/vf::S;
    residual[at]=ok?fabsl(1-prod):INFINITY;}
   if((*mask&(1U<<k))&&!pass){*mask&=~(1U<<k);failures[k]=ids[j];why[k]=ok?100:p.c.error;}
   if(k==last)break;
   if(aggregate&&!*mask)break;
   if(aggregate&&!p.active){for(int t=k+1;t<=last;++t)if(*mask&(1U<<t)){*mask&=~(1U<<t);failures[t]=ids[j];why[t]=p.c.error;}break;}
   p.next(k+1);
  }
 }
 return 0;
}
