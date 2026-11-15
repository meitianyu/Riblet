# Modified for Riblet from OpenBumbleBee's Flax activation adapter (Copyright 2023 Ant Group Co., Ltd.).
from contextlib import contextmanager
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import flax.linen as nn
from transformers.models.vit import modeling_flax_vit as vit
import spu.intrinsic as intrinsic
from public_inverse import invert
from riblet.vit_parameters import operator,point_shape,validate_schedule

@contextmanager
def native_nonlinears(coefficients=None,schedules=None,capture=None):
    old=(vit.ACT2FN['gelu'],jnn.gelu,nn.gelu,jnn.softmax,nn.softmax,lax.rsqrt)
    full=coefficients is not None
    if full and (len(coefficients)!=37 or schedules is None or len(schedules)!=37):raise ValueError('37 inverse records required')
    point=0
    def inverse(x,op):
        nonlocal point
        if point>=37 or operator(point)!=op or tuple(x.shape)!=point_shape(point):raise ValueError(f'ViT inverse topology at {point}: {op} {x.shape}')
        r,k=schedules[point];r,k=validate_schedule(point,r,k)
        out=invert(x,coefficients[point],jnp.asarray(r,dtype=jnp.int64),jnp.asarray(k,dtype=jnp.int64),operator=op)
        if capture is not None:capture.append((point,op,x,out))
        point+=1;return out
    def gelu(x):return intrinsic.spu_gelu(x)
    def softmax(x,axis=-1,where=None,initial=None):
        maximum=jnp.max(x,axis,where=where,initial=initial,keepdims=True)
        exponent=intrinsic.spu_neg_exp(x-maximum)
        denominator=jnp.sum(exponent,axis,where=where,keepdims=True)
        return exponent*inverse(denominator,'reciprocal') if full else exponent/denominator
    def rsqrt(x):return inverse(x,'rsqrt') if full else old[5](x)
    vit.ACT2FN['gelu']=jnn.gelu=nn.gelu=gelu
    jnn.softmax=nn.softmax=softmax;lax.rsqrt=rsqrt
    try:
        yield
        if full and point!=37:raise ValueError(f'incomplete inverse topology: {point}')
    finally:
        vit.ACT2FN['gelu'],jnn.gelu,nn.gelu,jnn.softmax,nn.softmax,lax.rsqrt=old

def make_forward(model,calls,group_ids,schedules,*,plan_sha256,model_epoch,request_epoch,diagnostics=False):
    from vit_full_secure_entry import make_full_evaluate
    def forward(params,image,coefficients,*handles):
        captured=[] if diagnostics else None
        core=make_full_evaluate(mode='ours-object-context',model_apply=lambda p,x:model(x,params=p,train=False).logits,
            dense_class=nn.Dense,dense_calls=calls,runtime_scope=lambda:native_nonlinears(coefficients,schedules,capture=captured),
            intrinsic=intrinsic,plan_sha256=plan_sha256,model_epoch=model_epoch,request_epoch=request_epoch,
            group_request_ids=group_ids,release_model=False)
        result=core(params,image,*handles)
        return (*result,tuple((row[2],row[3]) for row in captured)) if diagnostics else result
    return forward
