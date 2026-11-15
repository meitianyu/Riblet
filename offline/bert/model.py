import numpy as np
from plaintext_numeric import Arithmetic, Word, S, public, xp, spu_reciprocal, spu_rsqrt


def validate_b1(ids, segments):
    ids=np.asarray(ids); segments=np.asarray(segments)
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 128:
        raise ValueError('actual-length B1 required')
    if ids.shape != segments.shape or np.any(ids == 0) or np.any((segments < 0)|(segments > 1)):
        raise ValueError('PAD/segment mismatch')


def validate_physical_b1(ids,segments,attention_mask):
    ids=np.asarray(ids);segments=np.asarray(segments);mask=np.asarray(attention_mask)
    if ids.shape!=(1,128) or segments.shape!=ids.shape or mask.shape!=ids.shape:
        raise ValueError('native padded BERT requires [1,128]')
    n=int(mask.sum())
    if not 0<n<=128 or not np.array_equal(mask[0],(np.arange(128)<n).astype(mask.dtype)):
        raise ValueError('nonempty right-padded public mask required')
    if np.any(ids[0,:n]==0) or np.any(ids[0,n:]!=0) or np.any((segments<0)|(segments>1)):
        raise ValueError('PAD/segment mismatch')
    return n


def encode_tensor(value):
    value=np.asarray(value)
    scaled=np.trunc(value.astype(np.float64)*S)
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) >= 2**63):
        raise ValueError('invalid model tensor')
    return xp.asarray(scaled.astype(np.int64))


def point_seed(seed, point):
    return xp.asarray(seed,xp.uint64) ^ ((xp.asarray(point,xp.uint64)+xp.uint64(1))*xp.uint64(0xD1B54A32D192ED03))


def stack(values, axis=0):
    return Word(xp.stack([v.value for v in values],axis), xp.stack([v.state for v in values],axis))


def dense(ctx, x, p): return ctx.add(ctx.dot(x, p['kernel']), ctx.input(p['bias']))


def ln(ctx, x, p):
    if x.shape[-1] != 768: raise ValueError('LN width must be768')
    mean=ctx.mul_public(ctx.sum(x),341)
    mean_square=ctx.mul_public(ctx.sum(ctx.square(x)),341)
    raw_variance=ctx.add(mean_square,-ctx.square(mean))
    variance=ctx.select(raw_variance.value<0,0,raw_variance)
    centered=ctx.add(x,-mean)

    inverse,trace=spu_rsqrt(ctx,variance)
    scaled_inverse=ctx.mul(inverse,ctx.input(p['scale']))
    result=ctx.add(ctx.mul(centered,scaled_inverse),ctx.input(p['bias']))
    return result, dict(input=x.value,mean=mean.value,operand=variance.value,
        mean_square=mean_square.value,raw_variance=raw_variance.value,
        inverse=inverse.value,output=result.value,flags=ctx.flags,
        warnings=ctx.warning_flags,hint_warnings=ctx.hint_warning_count,**trace)


def neg_exp(ctx,x):
    from native_mixed import native_exp
    return native_exp(ctx,x)


def erf(ctx,x):
    negative=x.value < 0; absolute=ctx.select(negative,-x,x)
    inside=absolute.value < 3*S
    previous=ctx.active_mask;ctx.active_mask=inside if previous is None else previous&inside
    coeffs=[int(np.float32(c)*S) for c in (1.,.278393,.230389,.000972,.078108)]
    total=public(xp.full_like(x.value,S*coeffs[0]));power=absolute
    for degree,c in enumerate(coeffs[1:],1):
        if degree > 1: power=ctx.mul(power,absolute,sign='positive')
        total=ctx.add(total,ctx.local_product(power,c))
    z=ctx.trunc(total,sign='positive');z=ctx.square(z);z=ctx.square(z)
    result=ctx.add(S,-spu_reciprocal(ctx,z,positive=True))
    ctx.active_mask=previous;result=ctx.select(inside,result,S)
    return ctx.select(negative,-result,result)


def gelu(ctx,x):
    from native_mixed import native_gelu
    return native_gelu(ctx,x)


def embedding(params,ids,segments,seed):
    ctx=Arithmetic(point_seed(seed,0));p=params['bert']['embeddings']
    seg=ctx.input(p['token_type_embeddings']['embedding'][segments])

    raw=ctx.input(p['word_embeddings']['embedding'][ids]*xp.int64(S))
    word=ctx.trunc(raw)
    pos=ctx.input(p['position_embeddings']['embedding'][xp.arange(ids.shape[1])][None,:,:])
    x=ctx.add(ctx.add(word,pos),seg)
    return ln(ctx,x,p['LayerNorm'])


def attention_bias(ctx,scores,mask):

    bias=xp.where(xp.asarray(mask,xp.bool_)[:,None,None,:],xp.int64(0),xp.int64(-(1<<62)))
    return ctx.add(scores,public(bias))


def attention(ctx,h,p,mask=None):
    sp=p['self']; q=dense(ctx,h,sp['query']);k=dense(ctx,h,sp['key']);v=dense(ctx,h,sp['value'])
    n=h.shape[1];shape=(1,n,12,64)
    q=ctx.mul_public(q,S//8).reshape(shape).transpose(0,2,1,3)
    k=k.reshape(shape).transpose(0,2,3,1);scores=ctx.dot(q,k)
    if mask is not None:scores=attention_bias(ctx,scores,mask)

    indices=xp.argmax(scores.value,axis=-1,keepdims=True)
    maximum=Word(xp.take_along_axis(scores.value,indices,-1),xp.take_along_axis(scores.state,indices,-1))
    exps=neg_exp(ctx,ctx.add(scores,-maximum));d=ctx.sum(exps);inverse=spu_reciprocal(ctx,d)
    probs=ctx.mul(exps,inverse)
    masked=stack([ctx.mul_public(probs[0,head].reshape(1,n*n),S).reshape(n,n) for head in range(12)])
    vt=v.reshape(shape).transpose(0,2,3,1)[0]
    context=ctx.dot(vt,masked.transpose(0,2,1)).transpose(2,0,1).reshape(1,n,768)
    merged=ctx.add(dense(ctx,context,p['output']['dense']),h)
    return merged,dict(input=scores.value,operand=d.value,inverse=inverse.value,output=probs.value,
        exponentials=exps.value,
        flags=ctx.flags,warnings=ctx.warning_flags,hint_warnings=ctx.hint_warning_count,
        exp_min=xp.min(exps.value),exp_max=xp.max(exps.value))


def feedforward(ctx,h,p):
    activation=gelu(ctx,dense(ctx,h,p['intermediate']['dense']))
    return ctx.add(dense(ctx,activation,p['output']['dense']),h)


def run_point(point,h,params,ids,segments,seed):
    if point == 0: return embedding(params,ids,segments,seed)
    layer=(point-1)//3; kind=(point-1)%3
    lp=params['bert']['encoder']['layer'][str(layer)];ctx=Arithmetic(point_seed(seed,point))
    if kind == 0: return attention(ctx,h,lp['attention'])
    if kind == 1: return ln(ctx,h,lp['attention']['output']['LayerNorm'])
    merged=feedforward(ctx,h,lp)
    return ln(ctx,merged,lp['output']['LayerNorm'])


def tanh(ctx,x):
    x=ctx.select(x.value < -5*S,-5*S,ctx.select(x.value > 5*S,5*S,x))
    x=ctx.mul_public(x,int(np.float32(.2)*S));y=ctx.add(ctx.local_product(ctx.square(x),4),-2*S)
    z=ctx.add(y,-S);poly=[x,ctx.mul(x,z)]
    for _ in range(2,9):poly.append(ctx.add(ctx.mul(y,poly[-1]),-poly[-2]))
    coefficients=[1.2514045938932097,-.3655987797163166,.17253141478140663,-.08943445792774211,
        .047703017901250824,-.025830290571688078,.014338801903468182,-.008541730970059077,.0061230685785789475]
    values=stack(poly,axis=-2)
    cs=xp.asarray([int(np.float32(c)*S) for c in coefficients],xp.int64)[None,None,:]
    return ctx.dot(public(cs),values).reshape(x.shape)


def classify(h,params,seed):
    ctx=Arithmetic(point_seed(seed,37));x=dense(ctx,h[:,0,:],params['bert']['pooler']['dense'])
    logits=dense(ctx,tanh(ctx,x),params['classifier'])
    return logits.value,ctx.flags
