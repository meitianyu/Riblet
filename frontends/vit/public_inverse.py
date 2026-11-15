import numpy as np


def target(operator):
    if operator not in ('reciprocal','rsqrt'):raise ValueError('unknown inverse')
    return 'ours_vit_inverse_v1:'+operator


def invert(operand,coefficients,routes,public_k,*,operator):

    return _primitive().bind(operand,coefficients,routes,public_k,target=target(operator))


_PRIMITIVE=None


def _primitive():
    global _PRIMITIVE
    if _PRIMITIVE is not None:return _PRIMITIVE
    from functools import partial
    from jax import core
    from jax.interpreters import mlir,xla
    from jaxlib.hlo_helpers import custom_call
    def abstract(x,ab,r,k,*,target):
        if x.dtype!=np.float32 or ab.dtype!=np.float32 or ab.shape!=x.shape+(2,):
            raise ValueError('aligned float32 fixed-point inverse inputs required')
        if r.shape!=x.shape or k.shape!=x.shape or r.dtype!=np.int64 or k.dtype!=np.int64:
            raise ValueError('aligned public int64 schedule required')
        return core.ShapedArray(x.shape,x.dtype)
    def lower(ctx,x,ab,r,k,*,target):
        return custom_call(target,result_types=[x.type],operands=[x,ab,r,k]).results
    prim=core.Primitive('vit_cdi_invert');prim.def_impl(partial(xla.apply_primitive,prim))
    prim.def_abstract_eval(abstract);mlir.register_lowering(prim,lower)
    _PRIMITIVE=prim;return prim
