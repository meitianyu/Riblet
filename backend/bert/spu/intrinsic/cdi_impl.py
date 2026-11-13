__all__ = [
    "cdi_invert",
    "cdi_scope",
    "cdi_material_prepare",
    "invert_target",
]

from functools import partial

from jax import core
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call
import numpy as np


def _operator(value):
    value = str(value)
    assert value in ("reciprocal", "rsqrt"), "operator is invalid"
    return value


def _nonnegative(value, name):
    value = int(value)
    assert value >= 0, f"{name} must be nonnegative"
    return value


def cdi_scope(values, *, scope, site, action):

    assert scope in ("inference", "softmax", "layernorm", "reciprocal", "rsqrt", "barrier", "materials")
    assert action in ("begin", "end", "barrier", "release")
    assert (scope=="materials")== (action=="release"), 'material cleanup has a separate lifecycle boundary'
    sequence = isinstance(values, (tuple, list))
    operands = tuple(values) if sequence else (values,)
    result = tuple(_cdi_scope_prim.bind(*operands,
        target=f"cdi_scope:{scope}:{_nonnegative(site, 'site')}:{action}"))
    return result if sequence else result[0]


def cdi_material_prepare(quotas, *, request_epoch):
    assert quotas.ndim==2 and quotas.shape[1] in (2,4), 'public material quota must use raw2 or compact4 columns'
    _require_i64(quotas,'quotas')
    return _cdi_material_prepare_prim.bind(quotas,
        target=f"cdi_material_prepare:{_nonnegative(request_epoch,'request_epoch')}")


def _scope_abstract(*values, target):
    del target
    return tuple(core.ShapedArray(v.shape, v.dtype) for v in values)


def _scope_lowering(ctx, *values, target):
    return custom_call(target,
        result_types=[mlir.aval_to_ir_type(v) for v in ctx.avals_out],
        operands=list(values), has_side_effect=True).results


def invert_target(operator):
    return f"cdi_invert:{_operator(operator)}"


def _require_i64(value, name):
    assert np.dtype(value.dtype) == np.dtype(np.int64), f"{name} must be int64"


def cdi_invert(operand, coefficients, routes, public_ks, *, operator):

    assert operand.ndim >= 1, "operand must have at least one dimension"
    assert coefficients.shape == operand.shape + (
        2,
    ), "coefficients must have shape operand.shape + [2]"
    assert routes.shape == operand.shape, "routes must match operand shape"
    assert public_ks.shape == operand.shape, "public_ks must match operand shape"
    assert np.issubdtype(np.dtype(operand.dtype), np.floating), (
        "operand must be fixed-point floating input"
    )
    assert coefficients.dtype == operand.dtype, "coefficient dtype must match"
    _require_i64(routes, "routes")
    _require_i64(public_ks, "public_ks")
    original_shape = operand.shape
    count = operand.size
    result = _cdi_invert_prim.bind(
        operand.reshape((count,)),
        coefficients.reshape((count, 2)),
        routes.reshape((count,)),
        public_ks.reshape((count,)),
        target=invert_target(operator),
    )
    return result.reshape(original_shape)


def _setup_abstract(table, *, target):
    del target
    _require_i64(table, "table")
    return core.ShapedArray((1,), np.dtype(np.int64))


def _setup_lowering(ctx, table, *, target):
    return custom_call(
        target,
        result_types=[mlir.aval_to_ir_type(ctx.avals_out[0])],
        operands=[table],
        has_side_effect=True,
    ).results


def _invert_abstract(operand, coefficients, routes, public_ks, *, target):
    del target, coefficients, routes, public_ks
    return core.ShapedArray(operand.shape, operand.dtype)


def _invert_lowering(
    ctx, operand, coefficients, routes, public_ks, *, target
):
    return custom_call(
        target,
        result_types=[mlir.aval_to_ir_type(ctx.avals_out[0])],
        operands=[operand, coefficients, routes, public_ks],
        has_side_effect=True,
    ).results


def _primitive(name, abstract, lowering, multiple_results=False):
    primitive = core.Primitive(name)
    primitive.multiple_results = multiple_results
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(abstract)
    mlir.register_lowering(primitive, lowering)
    return primitive


_cdi_invert_prim = _primitive(
    "cdi_invert", _invert_abstract, _invert_lowering
)
_cdi_scope_prim = _primitive("cdi_scope", _scope_abstract, _scope_lowering, True)
_cdi_material_prepare_prim = _primitive('cdi_material_prepare',_setup_abstract,_setup_lowering)
