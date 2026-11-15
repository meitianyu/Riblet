import numpy as np
from vit_cdi import reference_numeric as p
xp=p.xp
M32=np.uint64((1<<32)-1)
S12=1<<12


def signed32(v):
    u=xp.asarray(v,xp.uint64)&M32
    return u.astype(xp.int64)-xp.where(u>=xp.uint64(1<<31),xp.int64(1<<32),xp.int64(0))


def cast_down(x):
    x=p.as_word(x)
    return p.Word(signed32(x.value),x.state&M32)


def cast_up(x):

    x=cast_down(x)
    return p.Word(x.value,x.state)


class Ring32:
    def __init__(self,parent):self.parent=parent
    def input(self,v):return cast_down(self.parent.input(signed32(v)))
    def add(self,a,b):
        a=cast_down(a);b=cast_down(b)
        return p.Word(signed32(a.value+b.value),(a.state+b.state)&M32)
    def local(self,x,c):
        x=cast_down(x);c=xp.asarray(c,xp.int64)
        return p.Word(signed32(x.value*c),(x.state*c.astype(xp.uint64))&M32)
    def trunc(self,x,bits=12,sign='unknown'):
        if not 1<=bits<=30 or sign not in ('unknown','positive','negative'):raise ValueError('32-bit truncation')
        x=cast_down(x);a=x.state;b=(x.value.astype(xp.uint64)-a)&M32
        h=xp.uint64(1<<30) if sign=='unknown' else xp.uint64(0)
        if sign=='unknown':self.parent.mark((x.value<-(1<<30))|(x.value>=(1<<30)),2)
        else:self.parent.hint_warning_count+=xp.sum(x.value<0 if sign=='positive' else x.value>=0,dtype=xp.int64)
        a=(a+h+xp.uint64(1<<31))&M32
        ma=a>>xp.uint64(31);mb=b>>xp.uint64(31)
        carry=ma|mb if sign=='negative' else ma&mb
        balance=(a>>xp.uint64(bits))&xp.uint64(1)
        offset=xp.uint64(1<<(31-bits));ho=h>>xp.uint64(bits)
        w0=p.words(self.parent.seed,self.parent.truncation_count,x.shape,0xA0761D6478BD642F)&M32
        self.parent.truncation_count+=1
        state=((a>>xp.uint64(bits))-(w0<<xp.uint64(32-bits))-offset+balance-ho)&M32
        value=(a>>xp.uint64(bits))+(b>>xp.uint64(bits))-(carry<<xp.uint64(32-bits))-offset+balance-ho
        return p.Word(signed32(value),state)
    def mul(self,a,b,sign='unknown'):
        a=cast_down(a);b=cast_down(b);raw=a.value*b.value
        self.parent.mark((raw<-(1<<31))|(raw>=(1<<31)),1)
        return self.trunc(self.input(raw),12,sign)
    def square(self,x):return self.mul(x,x,'positive')
    def less(self,x,constants):
        x=cast_down(x);a=x.state;b=(x.value.astype(xp.uint64)-a)&M32
        return [(((((a-xp.uint64(int(c)&((1<<32)-1)))&M32)>>xp.uint64(8))+(b>>xp.uint64(8)))>>xp.uint64(23))&xp.uint64(1)!=0 for c in constants]


def native_gelu(ctx,x):
    r=Ring32(ctx);x=r.trunc(cast_down(x),6)
    low,negative,high=r.less(x,[-3*S12,0,3*S12])
    nonnegative=~negative;large=~high;middle=~(low^large)
    absolute=r.input(xp.where(nonnegative,x.value,-x.value))
    previous=ctx.active_mask;ctx.active_mask=middle if previous is None else previous&middle
    x2=r.square(absolute);x4=r.square(x2);x3=r.mul(absolute,x2)
    c=np.array([.001620808531841547,-.03798164612714154,.5410550166368381,-.18352506127082727,.020848611754127593],np.float32)
    cs=[int(np.trunc(float(v)*S12)) for v in c]
    seg0=int(np.trunc(float(np.float32(c[0]*np.float32(S12)))*S12))
    polynomial=r.add(seg0,r.add(r.local(absolute,cs[1]),r.add(r.local(x2,cs[2]),r.add(r.local(x3,cs[3]),r.local(x4,cs[4])))))
    seg=r.trunc(r.add(r.local(x,S12//2),polynomial))
    ctx.active_mask=previous

    result=r.add(r.input(xp.where(large,x.value,0)),r.input(xp.where(middle,seg.value,0)))
    result=r.local(result,64)
    ctx.mark((result.value<-(1<<30))|(result.value>=(1<<30)),2)
    return cast_up(result)
