from fractions import Fraction
import numpy as np
S=1<<18

def critical(L,H,a,c,op=0):
    out={int(L),int(H)}
    if a:
        for d in ([2] if not op else [1,3]):
            q=Fraction(-c,d*a);i=q.numerator//q.denominator
            out.update(x for x in (i,i+1) if L<=x<=H)
    return out

def conservative_valid(L,H,a,b):
    if L<=0 or H<L:return False
    p,q=sorted((a*L,a*H))
    if p<-(1<<62) or q>=(1<<62) or p//S+b-1<=0:return False
    c=(b+1)*S
    return max(a*x*x+c*x for x in critical(L,H,a,c))<=S**3

def conservative(L,H,numerator):


    den=L*L+6*L*H+H*H
    aa=Fraction(-8*S**3,den);bb=Fraction(8*(L+H)*S*S,den)
    peak=max(Fraction(x)*(aa*x/S+bb) for x in critical(L,H,aa,bb*S))
    scale=Fraction(S*S,peak)*Fraction(numerator,16)
    a=int(aa*scale);guess=int(bb*scale)
    def upper_ok(b):
        c=(b+1)*S
        return max(a*x*x+c*x for x in critical(L,H,a,c))<=S**3
    lo=0;hi=max(0,guess)
    while lo<hi:
        mid=(lo+hi+1)//2
        if upper_ok(mid):lo=mid
        else:hi=mid-1
    b=lo
    if not conservative_valid(L,H,a,b):
        a=0;b=(S*S//H-1)*numerator//16
    if not conservative_valid(L,H,a,b):raise ValueError('no positive conservative f18 initializer')
    return a,b

def training_indices(x,folds,heldout,maximum=256):
    idx=np.flatnonzero(np.ones(len(x),bool) if heldout==5 else folds!=heldout)
    values=x[idx]
    selected=[idx[values.argmin()],idx[values.argmax()]]
    selected.extend(idx[np.linspace(0,len(idx)-1,min(maximum,len(idx)),dtype=np.int64)])
    return np.asarray(list(dict.fromkeys(selected)),np.int64)

def initial_certificate(op,L,H,a,b,bits=18,scale=0):
    if scale:
        L=(L>>scale)-1;H=(H>>scale)+1
        a*=1<<(2*scale if op=='reciprocal' else 3*scale//2)
        b*=1<<(scale if op=='reciprocal' else scale//2)
    a*=1<<(bits-18);b*=1<<(bits-18)
    L*=1<<(bits-18);H*=1<<(bits-18);s=1<<bits
    if L<=0 or H<L:return False
    p,q=sorted((a*L,a*H))
    if min(a,b)<-(1<<63) or max(a,b)>=(1<<63):return False
    if p<-(1<<62) or q>=(1<<62) or p//s+b-1<=0 or q//s+b+1>=(1<<63):return False
    c=(b+1)*s;xx=critical(L,H,a,c,op=='rsqrt')
    bound=max(a*x*x+c*x for x in xx) if op=='reciprocal' else max(x*(a*x+c)**2 for x in xx)
    return bound<(2*s**3 if op=='reciprocal' else 3*s**5)

def terminal_pool(op,L,H):
    from vit_cdi.candidates import coefficients
    base=coefficients(op,L,H,('analytic',));pairs={base}
    offsets=(-16,-8,-4,0,4,8,16)
    pairs.update((base[0]+da,base[1]+db) for da in offsets for db in offsets)
    if op=='reciprocal':
        for n in (16,15,14,12,8):
            try:pairs.add(conservative(L,H,n))
            except ValueError:pass
        if H>L:
            products=[Fraction(n,16) for n in (1,2,4,8,12,16,20,24)]
            for qL in products:
                for qH in products:
                    yL=qL*S*S/L;yH=qH*S*S/H
                    aa=S*(yH-yL)/(H-L);bb=yL-aa*L/S
                    pairs.add((int(aa),int(bb)))
    else:
        for j in range(0,33,4):
            for m in range(-16,17,4):pairs.add(coefficients(op,L,H,('robust',j,m)))
    return sorted(pairs)
