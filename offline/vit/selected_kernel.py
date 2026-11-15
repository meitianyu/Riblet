import numpy as np
from vit_cdi import reference_numeric as p
xp=p.xp

def selected(operator,x,a,b,k,seeds):
    if operator not in ('reciprocal','rsqrt'):raise ValueError('unknown inverse')
    bits=22 if operator=='reciprocal' else 18;S=1<<bits
    ctx=p.Arithmetic(seeds,per_event=True)
    operand=ctx.input(x);av=ctx.input(a);bv=ctx.input(b)
    if bits==22:
        operand=ctx.local_product(operand,16);av=ctx.local_product(av,16);bv=ctx.local_product(bv,16)
    y=ctx.add(ctx.mul(av,operand,bits=bits),bv)
    initial=operand.value*y.value;overflow=p.signed_mul_overflow(operand.value,y.value)
    if operator=='rsqrt':overflow|=p.signed_mul_overflow(initial,y.value);initial=initial*y.value
    upper=2*S*S if operator=='reciprocal' else 3*S**3
    active=(x>0)&(y.value>0)&~overflow&(initial>0)&(initial<upper)&(ctx.flags==0)
    raw_round=2 if operator=='reciprocal' else 3;trunc_round=2 if operator=='reciprocal' else 4
    def body(i,carry):
        old,valid,flags=carry;step=p.Arithmetic(seeds,per_event=True)
        step.counter=xp.uint64(4)+xp.asarray(i,xp.uint64)*raw_round
        step.truncation_count=xp.uint64(1)+xp.asarray(i,xp.uint64)*trunc_round;step.flags=flags
        if operator=='reciprocal':new=step.mul(old,step.add(2*S,-step.mul(operand,old,bits=bits)),bits=bits)
        else:new=step.trunc(step.mul(old,step.add(3*S,-step.mul(operand,step.mul(old,old)))),1)
        enabled=i<k
        return p.Word(xp.where(enabled,new.value,old.value),xp.where(enabled,new.state,old.state)),xp.where(enabled,valid&(step.flags==0)&(new.value>0),valid),xp.where(enabled,step.flags,flags)
    carry=(y,active,ctx.flags)
    if xp is np:
        for i in range(int(np.max(k))):carry=body(i,carry)
    else:
        import jax
        carry=jax.lax.fori_loop(0,xp.max(k),body,carry)
    y,valid,flags=carry
    if bits==22:


        out=p.Arithmetic(seeds,per_event=True);out.flags=flags
        out.truncation_count=xp.uint64(1)+xp.asarray(k,xp.uint64)*2
        rounded=out.trunc(y,4)
        v=(y.value>>4)+((y.value.astype(xp.uint64)>>3)&xp.uint64(1)).astype(xp.int64)
        y=p.Word(v,rounded.state+v.astype(xp.uint64)-rounded.value.astype(xp.uint64))
        valid=valid&(out.flags==0)&(v>0)
    return y,valid
