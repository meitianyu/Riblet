import hashlib
import numpy as np
from vit_cdi import reference_numeric as p
xp=p.xp;S=p.S;Word=p.Word


def patch_matrix(image):
    if image.shape!=(1,224,224,3):raise ValueError('B1 NHWC224 RGB image required')
    return image.reshape(1,14,16,14,16,3).transpose(0,1,3,2,4,5).reshape(1,196,768)


def encode(value):
    x=np.asarray(value);v=np.trunc(x.astype(np.float64)*S)
    if not np.all(np.isfinite(v)) or np.any(np.abs(v)>=2**63):raise ValueError('bad model/input encoding')
    return xp.asarray(v.astype(np.int64))


def point_seed(row,point):
    s=int.from_bytes(hashlib.sha256(('VIT_CDI_G9_DESIGN_V1\0'+row['relative_id']).encode()).digest()[:8],'little')
    return np.uint64(s ^ (((point+1)*0xD1B54A32D192ED03)&((1<<64)-1)))


def snapshot(ctx):
    return dict(seed=ctx.seed,counter=ctx.counter,truncations=ctx.truncation_count,
                flags=ctx.flags,warnings=ctx.warning_flags,hint_warnings=ctx.hint_warning_count)


def restore(prep,counts):
    ctx=p.Arithmetic(prep['seed']);ctx.counter=counts[0];ctx.truncation_count=counts[1]
    ctx.flags=prep['flags'];ctx.warning_flags=prep['warnings'];ctx.hint_warning_count=prep['hint_warnings']
    return ctx


def dense(ctx,x,weights):return ctx.add(ctx.dot(x,ctx.input(weights['kernel'])),ctx.input(weights['bias']))


def add_residual(ctx,state,update):return ctx.add(update,state['residual'])


def erf(ctx,x):

    negative=x.value<0;absolute=ctx.select(negative,-x,x)
    inside=absolute.value<3*S
    previous=ctx.active_mask;ctx.active_mask=inside if previous is None else previous&inside
    coeffs=[int(np.float32(c)*S) for c in (1.,.278393,.230389,.000972,.078108)]
    total=p.public(xp.full_like(x.value,S*coeffs[0]));power=absolute
    for degree,c in enumerate(coeffs[1:],1):
        if degree>1:power=ctx.mul(power,absolute,sign='positive')
        total=ctx.add(total,ctx.local_product(power,c))
    z=ctx.trunc(total,sign='positive');z=ctx.square(z);z=ctx.square(z)
    result=ctx.add(S,-p.spu_reciprocal(ctx,z,positive=True))
    ctx.active_mask=previous;result=ctx.select(inside,result,S)
    return ctx.select(negative,-result,result)


def gelu(ctx,x):
    from native_mixed import native_gelu
    return native_gelu(ctx,x)


def neg_exp(ctx,x):
    small=ctx.less_public(x,[-14*S])[0];previous=ctx.active_mask;ctx.active_mask=~small
    y=ctx.add(ctx.trunc(x,6),S)
    for _ in range(6):y=ctx.square(y)
    ctx.active_mask=previous
    return ctx.select(small,0,y)


def prepare_ln(ctx,h):
    mean=ctx.mul_public(ctx.sum(h),341)
    mean_square=ctx.mul_public(ctx.sum(ctx.square(h)),341)
    raw=ctx.add(mean_square,-ctx.square(mean));variance=ctx.select(raw.value<0,0,raw)
    return dict(operand=variance,centered=ctx.add(h,-mean),residual=h,
                mean=mean.value,mean_square=mean_square.value,raw_variance=raw.value,**snapshot(ctx))


def embedding(ctx,pixels,weights):
    patch=patch_matrix(pixels)
    w=weights['patch_embeddings']['projection']
    h=ctx.add(ctx.dot(ctx.input(patch),ctx.input(w['kernel'].reshape(768,768))),ctx.input(w['bias']))
    cls=ctx.input(weights['cls_token'])
    h=Word(xp.concatenate([cls.value,h.value],1),xp.concatenate([cls.state,h.state],1))
    return ctx.add(h,ctx.input(weights['position_embeddings']))


def prepare(kind,state,w,pixels,seed):
    ctx=p.Arithmetic(seed)
    if kind==0:
        h=embedding(ctx,pixels,w['embeddings']);return prepare_ln(ctx,h)
    if kind in (1,3,4):return prepare_ln(ctx,state['hidden'])

    sp=w['attention'];q=dense(ctx,state['norm'],sp['query'])
    v=dense(ctx,state['norm'],sp['value']);k=dense(ctx,state['norm'],sp['key'])
    q=ctx.mul_public(q,S//8).reshape(1,197,12,64).transpose(0,2,1,3)
    k=k.reshape(1,197,12,64).transpose(0,2,3,1);scores=ctx.dot(q,k)
    indices=xp.argmax(scores.value,axis=-1,keepdims=True)
    maximum=Word(xp.take_along_axis(scores.value,indices,-1),xp.take_along_axis(scores.state,indices,-1))
    exps=neg_exp(ctx,ctx.add(scores,-maximum))
    return dict(operand=ctx.sum(exps),exps=exps,value=v,residual=state['residual'],**snapshot(ctx))


def selected(operator,x,a,b,k,seeds):
    from selected_kernel import selected as evaluate
    return evaluate(operator,x,a,b,k,seeds)


def inverse(ctx,operand,selection,operator):
    if selection is None:
        y,trace=p.spu_rsqrt(ctx,operand) if operator=='rsqrt' else (p.spu_reciprocal(ctx,operand),{})
        return y,trace,xp.zeros(operand.shape,bool),xp.zeros(operand.shape,xp.int64)
    enabled=selection['enabled'].reshape(operand.shape);ctx.active_mask=~enabled
    fallback,trace=p.spu_rsqrt(ctx,operand) if operator=='rsqrt' else (p.spu_reciprocal(ctx,operand),{})
    ctx.active_mask=None
    x=xp.where(selection['enabled'],operand.value.reshape(-1),xp.int64(S))
    y,valid=selected(operator,x,selection['a'],selection['b'],selection['k'],selection['seeds'])
    ctx.mark((~valid&selection['enabled']).reshape(operand.shape),16)
    return Word(xp.where(enabled,y.value.reshape(operand.shape),fallback.value),
                xp.where(enabled,y.state.reshape(operand.shape),fallback.state)),trace,enabled,selection['k'].reshape(operand.shape)


def finish(kind,prep,w,selection,counts):
    ctx=restore(prep,counts);op='reciprocal' if kind==2 else 'rsqrt'
    y,trace,normal,k=inverse(ctx,prep['operand'],selection,op)
    if kind==2:
        probs=ctx.mul(prep['exps'],y)

        v=prep['value'].reshape(1,197,12,64).transpose(0,2,3,1)[0]
        context=ctx.dot(v,probs[0].transpose(0,2,1)).transpose(2,0,1).reshape(1,197,768)
        result={'hidden':ctx.add(dense(ctx,context,w['output']['dense']),prep['residual'])}
    else:
        normweights=w['ln'];scaled=ctx.mul(y,ctx.input(normweights['scale']))
        norm=ctx.add(ctx.mul(prep['centered'],scaled),ctx.input(normweights['bias']))
        if kind==3:
            mid=gelu(ctx,dense(ctx,norm,w['intermediate']['dense']))
            result={'hidden':ctx.add(dense(ctx,mid,w['output']['dense']),prep['residual'])}
        elif kind==4:
            result={'hidden':norm,'logits':dense(ctx,norm[:,0,:],w['classifier']).value}
        else:result={'norm':norm,'residual':prep['residual']}
    return result,dict(operand=prep['operand'].value,inverse=y.value,normal=normal,k=k,flags=ctx.flags,
                       warnings=ctx.warning_flags,hint_warnings=ctx.hint_warning_count,**trace)


def kind(point):return 0 if point==0 else 4 if point==36 else (2 if point%3==1 else 3 if point%3==2 else 1)


def weights(params,point):
    v=params['vit']
    if point==36:return {'ln':v['layernorm'],'classifier':params['classifier']}
    layer=v['encoder']['layer'][str(point//3)]
    if point==0:return {'embeddings':v['embeddings'],'ln':layer['layernorm_before']}
    if point%3==1:return layer['attention']
    if point%3==2:return {'ln':layer['layernorm_after'],'intermediate':layer['intermediate'],'output':layer['output']}
    return {'ln':layer['layernorm_before']}


def full(kind,state,w,pixels,seed):
    prep=prepare(kind,state,w,pixels,seed)
    return finish(kind,prep,w,None,(prep['counter'],prep['truncations']))


class Kernels:
    def __init__(self):
        import jax
        self.prepare=jax.jit(prepare,static_argnums=(0,));self.finish=jax.jit(finish,static_argnums=(0,4))
        self.full=jax.jit(full,static_argnums=(0,))


        self.prepare_batch=jax.jit(jax.vmap(prepare,in_axes=(None,0,None,0,0)),static_argnums=(0,))
        self.finish_batch=jax.jit(jax.vmap(finish,in_axes=(None,0,None,0,None)),static_argnums=(0,4))
