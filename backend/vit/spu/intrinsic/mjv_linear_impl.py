__all__ = [
    "mjv_consumer_preprocess",
    "mjv_linear_begin",
    "mjv_linear_end",
]

from functools import partial

import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call


def _public_id(value, name):
    value = int(value)
    assert value >= 0, f"{name} must be nonnegative"
    return value


def _manifest_hash(value):
    value = str(value)
    assert len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    ), "manifest must be a 64-character lowercase hex digest"
    return value


def _backend(value):
    value = str(value)
    assert value in ("bumblebee", "ours"), "invalid linear backend"
    return value


def _consumer_id(value):
    value = str(value)
    assert value and all(
        character.isalnum() or character in "_.-" for character in value
    ), "consumer_id must be a non-empty stable identifier"
    return value


def _matrix_shapes(lhs_shape, rhs_shape):
    lhs_shape = tuple(int(dimension) for dimension in lhs_shape)
    rhs_shape = tuple(int(dimension) for dimension in rhs_shape)
    assert len(lhs_shape) in (2, 3), "matrix rank must be 2 or 3"
    assert len(lhs_shape) == len(rhs_shape), "matrix ranks must match"
    if len(lhs_shape) == 2:
        assert lhs_shape[1] == rhs_shape[0], "matrix contract dimensions differ"
    else:
        assert lhs_shape[0] == rhs_shape[0], "matrix batch dimensions differ"
        assert lhs_shape[2] == rhs_shape[1], "matrix contract dimensions differ"
    assert all(
        dimension > 0 for dimension in lhs_shape + rhs_shape
    ), "matrix dimensions must be positive"
    return lhs_shape, rhs_shape


def _shape_text(shape):
    return "x".join(str(dimension) for dimension in shape)


def _stable_identifier(value, name):
    value = str(value)
    assert value and all(
        character.isalnum() or character in "_.-" for character in value
    ), f"{name} must be a non-empty stable identifier"
    return value


def _lifetime_code(value):
    value = str(value)
    codes = {"model-static": "s", "request-immutable": "r"}
    assert value in codes, "object lifetime must be model-static or request-immutable"
    return codes[value]


def _operand_code(value):
    value = str(value)
    codes = {"lhs": "l", "rhs": "r"}
    assert value in codes, "object operand must be lhs or rhs"
    return codes[value]


def _runtime_metadata(
    *,
    model_epoch,
    physical_family_id,
    logical_object_id,
    logical_members,
    object_lifetime,
    object_operand,
    family_kind,
):
    values = (
        model_epoch,
        physical_family_id,
        logical_object_id,
        logical_members,
        object_lifetime,
        object_operand,
        family_kind,
    )
    if all(value is None for value in values):
        return None
    assert all(value is not None for value in values), (
        "runtime consumer metadata must be complete"
    )
    logical_members = int(logical_members)
    assert logical_members > 0, "logical_members must be positive"
    return (
        _public_id(model_epoch, "model_epoch"),
        _stable_identifier(physical_family_id, "physical_family_id"),
        _stable_identifier(logical_object_id, "logical_object_id"),
        logical_members,
        _lifetime_code(object_lifetime),
        _operand_code(object_operand),
        _stable_identifier(family_kind, "family_kind"),
    )


def linear_begin_target(
    request_id,
    manifest_hash,
    backend="ours",
    *,
    common_fallback_prefix_calls=0,
    allowed_online_dot_ole_calls=0,
    allowed_online_batch_dot_ole_calls=0,
):
    request_id = _public_id(request_id, "request_id")
    manifest_hash = _manifest_hash(manifest_hash)
    backend = _backend(backend)
    common_fallback_prefix_calls = _public_id(
        common_fallback_prefix_calls, "common_fallback_prefix_calls"
    )
    allowed_online_dot_ole_calls = _public_id(
        allowed_online_dot_ole_calls, "allowed_online_dot_ole_calls"
    )
    allowed_online_batch_dot_ole_calls = _public_id(
        allowed_online_batch_dot_ole_calls,
        "allowed_online_batch_dot_ole_calls",
    )
    if backend == "bumblebee":
        assert (
            common_fallback_prefix_calls == 0
            and allowed_online_dot_ole_calls == 0
            and allowed_online_batch_dot_ole_calls == 0
        ), "BumbleBee common fallback allowance must be zero"
    return (
        f"mjv_linear_begin:{backend}:{request_id}:{manifest_hash}:"
        f"{common_fallback_prefix_calls}:{allowed_online_dot_ole_calls}:"
        f"{allowed_online_batch_dot_ole_calls}"
    )


def linear_end_target(request_id):
    request_id = _public_id(request_id, "request_id")
    return f"mjv_linear_end:{request_id}"


def consumer_preprocess_target(
    plan_sha256,
    request_epoch,
    consumer_id,
    lhs_shape,
    rhs_shape,
    *,
    model_epoch=None,
    physical_family_id=None,
    logical_object_id=None,
    logical_members=None,
    object_lifetime=None,
    object_operand=None,
    family_kind=None,
):
    plan_sha256 = _manifest_hash(plan_sha256)
    request_epoch = _public_id(request_epoch, "request_epoch")
    consumer_id = _consumer_id(consumer_id)
    lhs_shape, rhs_shape = _matrix_shapes(lhs_shape, rhs_shape)
    rank_code = "b" if len(lhs_shape) == 3 else "m"
    target = (
        f"mjv_consumer_preprocess:{plan_sha256}:{request_epoch}:"
        f"{consumer_id}:{rank_code}:"
        f"{_shape_text(lhs_shape)}:{_shape_text(rhs_shape)}"
    )
    metadata = _runtime_metadata(
        model_epoch=model_epoch,
        physical_family_id=physical_family_id,
        logical_object_id=logical_object_id,
        logical_members=logical_members,
        object_lifetime=object_lifetime,
        object_operand=object_operand,
        family_kind=family_kind,
    )
    if metadata is None:
        return target
    epoch, family_id, object_id, members, lifetime, operand, kind = metadata
    return (
        f"{target}:{epoch}:{family_id}:{object_id}:{members}:"
        f"{lifetime}:{operand}:{kind}"
    )


def mjv_linear_begin(
    *args,
    request_id=None,
    manifest_hash=None,
    backend="ours",
    common_fallback_prefix_calls=0,
    allowed_online_dot_ole_calls=0,
    allowed_online_batch_dot_ole_calls=0,
):
    if request_id is None:
        assert 2 <= len(args) <= 3, "scope begin requires request and manifest"
        anchor = None
        request_id, manifest_hash = args[:2]
        if len(args) == 3:
            backend = args[2]
    else:
        assert len(args) <= 1, "scope begin accepts at most one anchor"
        anchor = None if not args else args[0]
    request_id = _public_id(request_id, "request_id")
    manifest_hash = _manifest_hash(manifest_hash)
    backend = _backend(backend)
    common_fallback_prefix_calls = _public_id(
        common_fallback_prefix_calls, "common_fallback_prefix_calls"
    )
    allowed_online_dot_ole_calls = _public_id(
        allowed_online_dot_ole_calls, "allowed_online_dot_ole_calls"
    )
    allowed_online_batch_dot_ole_calls = _public_id(
        allowed_online_batch_dot_ole_calls,
        "allowed_online_batch_dot_ole_calls",
    )
    linear_begin_target(
        request_id,
        manifest_hash,
        backend,
        common_fallback_prefix_calls=common_fallback_prefix_calls,
        allowed_online_dot_ole_calls=allowed_online_dot_ole_calls,
        allowed_online_batch_dot_ole_calls=allowed_online_batch_dot_ole_calls,
    )
    operands = () if anchor is None else (anchor,)
    outputs = _mjv_linear_begin_prim.bind(
        *operands,
        request_id=request_id,
        manifest_hash=manifest_hash,
        backend=backend,
        common_fallback_prefix_calls=common_fallback_prefix_calls,
        allowed_online_dot_ole_calls=allowed_online_dot_ole_calls,
        allowed_online_batch_dot_ole_calls=allowed_online_batch_dot_ole_calls,
    )
    return None if anchor is None else outputs[0]


def mjv_linear_end(anchor=None, *, request_id=None):
    if request_id is None:
        request_id = anchor
        anchor = None
    request_id = _public_id(request_id, "request_id")
    operands = () if anchor is None else (anchor,)
    outputs = _mjv_linear_end_prim.bind(*operands, request_id=request_id)
    return None if anchor is None else outputs[0]


def mjv_consumer_preprocess(
    *,
    plan_sha256,
    request_epoch,
    consumer_id,
    lhs_shape,
    rhs_shape,
    dtype=jnp.float32,
    model_epoch=None,
    physical_family_id=None,
    logical_object_id=None,
    logical_members=None,
    object_lifetime=None,
    object_operand=None,
    family_kind=None,
):
    plan_sha256 = _manifest_hash(plan_sha256)
    request_epoch = _public_id(request_epoch, "request_epoch")
    consumer_id = _consumer_id(consumer_id)
    lhs_shape, rhs_shape = _matrix_shapes(lhs_shape, rhs_shape)
    dtype = dtypes.canonicalize_dtype(dtype)
    runtime_metadata = _runtime_metadata(
        model_epoch=model_epoch,
        physical_family_id=physical_family_id,
        logical_object_id=logical_object_id,
        logical_members=logical_members,
        object_lifetime=object_lifetime,
        object_operand=object_operand,
        family_kind=family_kind,
    )
    return tuple(
        _mjv_consumer_preprocess_prim.bind(
            plan_sha256=plan_sha256,
            request_epoch=request_epoch,
            consumer_id=consumer_id,
            lhs_shape=lhs_shape,
            rhs_shape=rhs_shape,
            dtype=dtype,
            runtime_metadata=runtime_metadata,
        )
    )


def _anchor_abstract(operands):
    assert len(operands) <= 1, "linear scope accepts at most one anchor"
    return () if not operands else (operands[0],)


def _begin_abstract(
    *operands,
    request_id,
    manifest_hash,
    backend,
    common_fallback_prefix_calls,
    allowed_online_dot_ole_calls,
    allowed_online_batch_dot_ole_calls,
):
    linear_begin_target(
        request_id,
        manifest_hash,
        backend,
        common_fallback_prefix_calls=common_fallback_prefix_calls,
        allowed_online_dot_ole_calls=allowed_online_dot_ole_calls,
        allowed_online_batch_dot_ole_calls=allowed_online_batch_dot_ole_calls,
    )
    return _anchor_abstract(operands)


def _begin_lowering(
    ctx,
    *operands,
    request_id,
    manifest_hash,
    backend,
    common_fallback_prefix_calls,
    allowed_online_dot_ole_calls,
    allowed_online_batch_dot_ole_calls,
):
    result_types = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
    return custom_call(
        linear_begin_target(
            request_id,
            manifest_hash,
            backend,
            common_fallback_prefix_calls=common_fallback_prefix_calls,
            allowed_online_dot_ole_calls=allowed_online_dot_ole_calls,
            allowed_online_batch_dot_ole_calls=(
                allowed_online_batch_dot_ole_calls
            ),
        ),
        result_types=result_types,
        operands=list(operands),
        has_side_effect=True,
    ).results


def _end_abstract(*operands, request_id):
    linear_end_target(request_id)
    return _anchor_abstract(operands)


def _end_lowering(ctx, *operands, request_id):
    result_types = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
    return custom_call(
        linear_end_target(request_id),
        result_types=result_types,
        operands=list(operands),
        has_side_effect=True,
    ).results


def _preprocess_abstract(
    *,
    plan_sha256,
    request_epoch,
    consumer_id,
    lhs_shape,
    rhs_shape,
    dtype,
    runtime_metadata,
):
    consumer_preprocess_target(
        plan_sha256,
        request_epoch,
        consumer_id,
        lhs_shape,
        rhs_shape,
        **(
            {}
            if runtime_metadata is None
            else dict(
                model_epoch=runtime_metadata[0],
                physical_family_id=runtime_metadata[1],
                logical_object_id=runtime_metadata[2],
                logical_members=runtime_metadata[3],
                object_lifetime={"s": "model-static", "r": "request-immutable"}[
                    runtime_metadata[4]
                ],
                object_operand={"l": "lhs", "r": "rhs"}[runtime_metadata[5]],
                family_kind=runtime_metadata[6],
            )
        ),
    )
    return ShapedArray(lhs_shape, dtype), ShapedArray(rhs_shape, dtype)


def _preprocess_lowering(
    ctx,
    *,
    plan_sha256,
    request_epoch,
    consumer_id,
    lhs_shape,
    rhs_shape,
    dtype,
    runtime_metadata,
):
    del ctx
    result_types = [
        mlir.aval_to_ir_type(ShapedArray(lhs_shape, dtype)),
        mlir.aval_to_ir_type(ShapedArray(rhs_shape, dtype)),
    ]
    metadata_kwargs = {}
    if runtime_metadata is not None:
        metadata_kwargs = dict(
            model_epoch=runtime_metadata[0],
            physical_family_id=runtime_metadata[1],
            logical_object_id=runtime_metadata[2],
            logical_members=runtime_metadata[3],
            object_lifetime={"s": "model-static", "r": "request-immutable"}[
                runtime_metadata[4]
            ],
            object_operand={"l": "lhs", "r": "rhs"}[runtime_metadata[5]],
            family_kind=runtime_metadata[6],
        )
    return custom_call(
        consumer_preprocess_target(
            plan_sha256,
            request_epoch,
            consumer_id,
            lhs_shape,
            rhs_shape,
            **metadata_kwargs,
        ),
        result_types=result_types,
        operands=[],
        has_side_effect=True,
    ).results


_mjv_linear_begin_prim = core.Primitive("mjv_linear_begin")
_mjv_linear_begin_prim.multiple_results = True
_mjv_linear_begin_prim.def_impl(partial(xla.apply_primitive, _mjv_linear_begin_prim))
_mjv_linear_begin_prim.def_abstract_eval(_begin_abstract)
mlir.register_lowering(_mjv_linear_begin_prim, _begin_lowering)

_mjv_linear_end_prim = core.Primitive("mjv_linear_end")
_mjv_linear_end_prim.multiple_results = True
_mjv_linear_end_prim.def_impl(partial(xla.apply_primitive, _mjv_linear_end_prim))
_mjv_linear_end_prim.def_abstract_eval(_end_abstract)
mlir.register_lowering(_mjv_linear_end_prim, _end_lowering)

_mjv_consumer_preprocess_prim = core.Primitive("mjv_consumer_preprocess")
_mjv_consumer_preprocess_prim.multiple_results = True
_mjv_consumer_preprocess_prim.def_impl(
    partial(xla.apply_primitive, _mjv_consumer_preprocess_prim)
)
_mjv_consumer_preprocess_prim.def_abstract_eval(_preprocess_abstract)
mlir.register_lowering(_mjv_consumer_preprocess_prim, _preprocess_lowering)
