import numpy as np
import plaintext_numeric as p
import model as m
from layout import physical_length
xp=p.xp


def target_points(mode):
    if mode=='reciprocal':return tuple(1+3*i for i in range(12))
    if mode=='rsqrt':return tuple(i for i in range(37) if i==0 or (i-1)%3!=0)
    if mode=='joint':return tuple(range(37))
    raise ValueError('unknown lineage mode')


def event_ids(request,point,head,n):
    if not (0<=point<37 and 0<=head<12 and 1<=n<=128):raise ValueError('invalid event coordinates')
    return np.uint64((int(request)<<17)|(point<<11)|(head<<7)) | np.arange(n,dtype=np.uint64)


def snapshot(ctx):
    return dict(seed=ctx.seed,counter=ctx.counter,truncations=ctx.truncation_count,
        flags=ctx.flags,warnings=ctx.warning_flags,hint_warnings=ctx.hint_warning_count)


def restore(prep,counts=None):
    ctx=p.Arithmetic(prep['seed'])
    ctx.counter=int(prep['counter']) if counts is None else counts[0]
    ctx.truncation_count=int(prep['truncations']) if counts is None else counts[1]
    ctx.flags=prep['flags'];ctx.warning_flags=prep['warnings'];ctx.hint_warning_count=prep['hint_warnings']
    return ctx


def prepare_ln(ctx,h):
    mean=ctx.mul_public(ctx.sum(h),341)
    mean_square=ctx.mul_public(ctx.sum(ctx.square(h)),341)
    raw_variance=ctx.add(mean_square,-ctx.square(mean))
    variance=ctx.select(raw_variance.value<0,0,raw_variance)
    centered=ctx.add(h,-mean)
    return dict(operand=variance,centered=centered,mean=mean.value,
                mean_square=mean_square.value,raw_variance=raw_variance.value,**snapshot(ctx))


def prepare_embedding(params,ids,segments,seed):
    ctx=p.Arithmetic(m.point_seed(seed,0));w=params['bert']['embeddings']


    seg=ctx.input(w['token_type_embeddings']['embedding'][segments])
    word=ctx.trunc(ctx.input(w['word_embeddings']['embedding'][ids]*xp.int64(p.S)))
    pos=ctx.input(w['position_embeddings']['embedding'][xp.arange(ids.shape[1])][None,:,:])
    return prepare_ln(ctx,ctx.add(ctx.add(word,pos),seg))


def prepare_softmax(h,weights,seed,mask):
    ctx=p.Arithmetic(seed);sp=weights['self']
    q=m.dense(ctx,h,sp['query']);k=m.dense(ctx,h,sp['key']);v=m.dense(ctx,h,sp['value'])
    n=h.shape[1];shape=(1,n,12,64)
    q=ctx.mul_public(q,p.S//8).reshape(shape).transpose(0,2,1,3);k=k.reshape(shape).transpose(0,2,3,1)
    scores=m.attention_bias(ctx,ctx.dot(q,k),mask)
    idx=xp.argmax(scores.value,axis=-1,keepdims=True)
    maximum=p.Word(xp.take_along_axis(scores.value,idx,-1),xp.take_along_axis(scores.state,idx,-1))
    exps=m.neg_exp(ctx,ctx.add(scores,-maximum));operand=ctx.sum(exps)
    return dict(operand=operand,exps=exps,value=v,residual=h,**snapshot(ctx))


def replay_selected(operator,x,a,b,k,seeds):

    ctx=p.Arithmetic(seeds,per_event=True);operand=ctx.input(x)
    av=ctx.input(a);bv=ctx.input(b);y=ctx.add(ctx.mul(av,operand),bv)
    initial=x*y.value
    overflow=p.signed_mul_overflow(x,y.value)
    if operator=='rsqrt':
        overflow=overflow|p.signed_mul_overflow(initial,y.value);initial=initial*y.value
    limit=2*p.S*p.S if operator=='reciprocal' else 3*p.S**3
    active=(x>0)&(y.value>0)&~overflow&(initial>0)&(initial<limit)&(ctx.flags==0)
    if xp is not np:
        import jax
        raw_per_round=2 if operator=='reciprocal' else 3
        trunc_per_round=2 if operator=='reciprocal' else 4
        def body(i,carry):
            previous,valid,flags=carry
            step=p.Arithmetic(seeds,per_event=True)
            step.counter=xp.uint64(4)+xp.asarray(i,xp.uint64)*xp.uint64(raw_per_round)
            step.truncation_count=xp.uint64(1)+xp.asarray(i,xp.uint64)*xp.uint64(trunc_per_round)
            step.flags=flags
            if operator=='reciprocal':updated=step.mul(previous,step.add(2*p.S,-step.mul(operand,previous)))
            else:
                yy=step.mul(previous,previous);xyy=step.mul(operand,yy)
                updated=step.trunc(step.mul(previous,step.add(3*p.S,-xyy)),1)
            enabled=i<k
            result=p.Word(xp.where(enabled,updated.value,previous.value),xp.where(enabled,updated.state,previous.state))
            return result,xp.where(enabled,valid&(step.flags==0)&(updated.value>0),valid),xp.where(enabled,step.flags,flags)
        result,valid,_=jax.lax.fori_loop(0,xp.max(k),body,(y,active,ctx.flags))
        return result,valid
    ys=[];states=[];valid=[]
    for step in range(9):
        active=active&(ctx.flags==0)&(y.value>0)
        ys.append(y.value);states.append(y.state);valid.append(active)
        if step==8:break
        if operator=='reciprocal':y=ctx.mul(y,ctx.add(2*p.S,-ctx.mul(operand,y)))
        else:
            yy=ctx.mul(y,y);xyy=ctx.mul(operand,yy)
            y=ctx.trunc(ctx.mul(y,ctx.add(3*p.S,-xyy)),1)
    index=xp.asarray(k,xp.int64)[None,:]
    return p.Word(xp.take_along_axis(xp.stack(ys),index,axis=0)[0],
        xp.take_along_axis(xp.stack(states),index,axis=0)[0]),xp.take_along_axis(xp.stack(valid),index,axis=0)[0]


def inverse(ctx,operand,selection,operator):
    if selection is None:
        if operator=='rsqrt':y,trace=p.spu_rsqrt(ctx,operand)
        else:y=p.spu_reciprocal(ctx,operand);trace={}
        return y,trace,xp.zeros(operand.shape,dtype=xp.bool_),xp.zeros(operand.shape,dtype=xp.int64)
    enabled=selection['enabled'].reshape(operand.shape);previous=ctx.active_mask;ctx.active_mask=~enabled
    if operator=='rsqrt':fallback,trace=p.spu_rsqrt(ctx,operand)
    else:fallback=p.spu_reciprocal(ctx,operand);trace={}
    ctx.active_mask=previous
    x=xp.where(selection['enabled'],operand.value.reshape(-1),xp.int64(p.S))
    y,valid=replay_selected(operator,x,selection['a'],selection['b'],selection['k'],selection['seeds'])
    ctx.mark((~valid&selection['enabled']).reshape(operand.shape),16)
    value=xp.where(enabled,y.value.reshape(operand.shape),fallback.value)
    state=xp.where(enabled,y.state.reshape(operand.shape),fallback.state)
    return p.Word(value,state),trace,enabled,selection['k'].reshape(operand.shape)


def finish_ln(prep,weights,selection,operator='rsqrt',counts=None):
    ctx=restore(prep,counts);y,trace,normal,k=inverse(ctx,prep['operand'],selection,operator)
    scaled_inverse=ctx.mul(y,ctx.input(weights['scale']))
    output=ctx.add(ctx.mul(prep['centered'],scaled_inverse),ctx.input(weights['bias']))
    return output,dict(operand=prep['operand'].value,inverse=y.value,normal=normal,k=k,
        flags=ctx.flags,warnings=ctx.warning_flags,mean=prep['mean'],
        mean_square=prep['mean_square'],raw_variance=prep['raw_variance'],**trace)


def finish_softmax(prep,weights,selection,counts=None):
    ctx=restore(prep,counts);y,trace,normal,k=inverse(ctx,prep['operand'],selection,'reciprocal')
    probs=ctx.mul(prep['exps'],y);n=prep['residual'].shape[1]
    masked=m.stack([ctx.mul_public(probs[0,h].reshape(1,n*n),p.S).reshape(n,n) for h in range(12)])
    vt=prep['value'].reshape(1,n,12,64).transpose(0,2,3,1)[0]
    context=ctx.dot(vt,masked.transpose(0,2,1)).transpose(2,0,1).reshape(1,n,768)
    output=ctx.add(m.dense(ctx,context,weights['output']['dense']),prep['residual'])
    return output,dict(operand=prep['operand'].value,inverse=y.value,normal=normal,k=k,
        flags=ctx.flags,warnings=ctx.warning_flags,normalization_output=probs.value)


def selection_arrays(document,row,point):
    from public_sites import project
    n=row['effective_length']
    return project(document['sites'],point,row['request_ordinal'],n,physical_length(n))
