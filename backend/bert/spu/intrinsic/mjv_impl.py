__all__ = [
    "model_setup_target",
    "request_preprocess_target",
    "mjv_model_setup",
    "mjv_request_preprocess",
    "mjv_group_online",
    "mjv_linear_group_setup",
    "mjv_linear_group_online",
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


def _output_widths(widths):
    widths = tuple(int(width) for width in widths)
    assert widths and all(
        width > 0 for width in widths
    ), "output widths must be positive"
    return widths


def _width_text(widths):
    return ",".join(str(width) for width in widths)


def _linear_family(value):
    value = str(value)
    assert value in (
        "matmul_aa",
        "matmul_av",
        "matmul_vvs",
    ), "invalid static linear family"
    return value


def _manifest_hash(value):
    value = str(value)
    assert len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    ), "plan_sha256 must be a 64-character lowercase hex digest"
    return value


def _stable_identifier(value, name):
    value = str(value)
    assert value and all(
        character.isalnum() or character in "_.-" for character in value
    ), f"{name} must be a non-empty stable identifier"
    return value


def _shape(value, name):
    value = tuple(int(dimension) for dimension in value)
    assert value and all(dimension > 0 for dimension in value), (
        f"{name} dimensions must be positive"
    )
    return value


def _shape_text(value):
    return "x".join(str(dimension) for dimension in value)


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


def _model_runtime_metadata(
    *,
    plan_sha256,
    model_epoch,
    physical_family_id,
    logical_object_id,
    logical_members,
    object_lifetime,
    object_operand,
    family_kind,
    shared_shape,
    right_shape,
):
    values = (
        plan_sha256,
        model_epoch,
        physical_family_id,
        logical_object_id,
        logical_members,
        object_lifetime,
        object_operand,
        family_kind,
        shared_shape,
        right_shape,
    )
    if all(value is None for value in values):
        return None
    assert all(value is not None for value in values), (
        "runtime family metadata must be complete"
    )
    logical_members = int(logical_members)
    assert logical_members > 0, "logical_members must be positive"
    return (
        _manifest_hash(plan_sha256),
        _public_id(model_epoch, "model_epoch"),
        _stable_identifier(physical_family_id, "physical_family_id"),
        _stable_identifier(logical_object_id, "logical_object_id"),
        logical_members,
        _lifetime_code(object_lifetime),
        _operand_code(object_operand),
        _stable_identifier(family_kind, "family_kind"),
        _shape(shared_shape, "shared_shape"),
        _shape(right_shape, "right_shape"),
    )


def model_setup_target(
    *,
    model_id,
    output_widths,
    sum_r,
    linear_family="matmul_aa",
    plan_sha256=None,
    model_epoch=None,
    physical_family_id=None,
    logical_object_id=None,
    logical_members=None,
    object_lifetime=None,
    object_operand=None,
    family_kind=None,
    shared_shape=None,
    right_shape=None,
):
    model_id = _public_id(model_id, "model_id")
    output_widths = _output_widths(output_widths)
    sum_r = int(sum_r)
    assert sum_r > 0, "sum_r must be positive"
    linear_family = _linear_family(linear_family)
    target = (
        f"mjv_model_setup:{model_id}:{sum_r}:{_width_text(output_widths)}:"
        f"{linear_family}"
    )
    metadata = _model_runtime_metadata(
        plan_sha256=plan_sha256,
        model_epoch=model_epoch,
        physical_family_id=physical_family_id,
        logical_object_id=logical_object_id,
        logical_members=logical_members,
        object_lifetime=object_lifetime,
        object_operand=object_operand,
        family_kind=family_kind,
        shared_shape=shared_shape,
        right_shape=right_shape,
    )
    if metadata is None:
        return target
    plan, epoch, family_id, object_id, members, lifetime, operand, kind, shared, right = metadata
    return (
        f"{target}:{plan}:{epoch}:{family_id}:{object_id}:{members}:"
        f"{lifetime}:{operand}:{kind}:{_shape_text(shared)}:{_shape_text(right)}"
    )


def request_preprocess_target(
    *,
    model_id,
    request_id,
    rows,
    plan_sha256=None,
    model_epoch=None,
    request_epoch=None,
    physical_family_id=None,
):
    model_id = _public_id(model_id, "model_id")
    request_id = _public_id(request_id, "request_id")
    rows = int(rows)
    assert rows > 0, "rows must be positive"
    target = f"mjv_request_preprocess:{model_id}:{request_id}:{rows}"
    values = (plan_sha256, model_epoch, request_epoch, physical_family_id)
    if all(value is None for value in values):
        return target
    assert all(value is not None for value in values), (
        "runtime request metadata must be complete"
    )
    return (
        f"{target}:{_manifest_hash(plan_sha256)}:"
        f"{_public_id(model_epoch, 'model_epoch')}:"
        f"{_public_id(request_epoch, 'request_epoch')}:"
        f"{_stable_identifier(physical_family_id, 'physical_family_id')}"
    )


def mjv_model_setup(
    weight,
    *,
    model_id,
    output_widths,
    sum_r,
    linear_family="matmul_aa",
    plan_sha256=None,
    model_epoch=None,
    physical_family_id=None,
    logical_object_id=None,
    logical_members=None,
    object_lifetime=None,
    object_operand=None,
    family_kind=None,
    shared_shape=None,
    right_shape=None,
):

    assert weight.ndim == 2, "weight rank must be 2"
    model_id = _public_id(model_id, "model_id")
    output_widths = _output_widths(output_widths)
    sum_r = int(sum_r)
    linear_family = _linear_family(linear_family)
    assert (
        sum(output_widths) == weight.shape[1]
    ), "output widths must match the combined weight"
    assert sum_r >= weight.shape[0], "sum_r must be at least rho"
    metadata = _model_runtime_metadata(
        plan_sha256=plan_sha256,
        model_epoch=model_epoch,
        physical_family_id=physical_family_id,
        logical_object_id=logical_object_id,
        logical_members=logical_members,
        object_lifetime=object_lifetime,
        object_operand=object_operand,
        family_kind=family_kind,
        shared_shape=shared_shape,
        right_shape=right_shape,
    )
    return _mjv_model_setup_prim.bind(
        weight,
        model_id=model_id,
        output_widths=output_widths,
        sum_r=sum_r,
        linear_family=linear_family,
        runtime_metadata=metadata,
    )


def mjv_request_preprocess(
    model_handle,
    *,
    model_id,
    request_id,
    rows,
    plan_sha256=None,
    model_epoch=None,
    request_epoch=None,
    physical_family_id=None,
):

    assert model_handle.ndim == 2, "model handle rank must be 2"
    model_id = _public_id(model_id, "model_id")
    request_id = _public_id(request_id, "request_id")
    rows = int(rows)
    assert rows > 0, "rows must be positive"
    request_preprocess_target(
        model_id=model_id,
        request_id=request_id,
        rows=rows,
        plan_sha256=plan_sha256,
        model_epoch=model_epoch,
        request_epoch=request_epoch,
        physical_family_id=physical_family_id,
    )
    return _mjv_request_preprocess_prim.bind(
        model_handle,
        model_id=model_id,
        request_id=request_id,
        rows=rows,
        plan_sha256=plan_sha256,
        model_epoch=model_epoch,
        request_epoch=request_epoch,
        physical_family_id=physical_family_id,
    )


def mjv_group_online(
    request_handle,
    joint_activation,
    *,
    model_id,
    request_id,
    output_widths,
    logical_row_members=0,
):

    assert request_handle.ndim == 2, "request handle rank must be 2"
    assert joint_activation.ndim == 2, "joint activation rank must be 2"
    assert (
        request_handle.shape == joint_activation.shape
    ), "request handle and activation shapes must match"
    model_id = _public_id(model_id, "model_id")
    request_id = _public_id(request_id, "request_id")
    output_widths = _output_widths(output_widths)
    logical_row_members = int(logical_row_members)
    assert logical_row_members >= 0, "logical_row_members must be nonnegative"
    if logical_row_members:
        assert len(output_widths) == 1, "row expansion requires one output"
        assert (
            logical_row_members == request_handle.shape[0]
        ), "logical row members must equal physical rows"
    return tuple(
        _mjv_group_online_prim.bind(
            request_handle,
            joint_activation,
            model_id=model_id,
            request_id=request_id,
            output_widths=output_widths,
            logical_row_members=logical_row_members,
        )
    )


def mjv_linear_group_setup(weights, *, model_id, sum_r, linear_family="matmul_aa"):

    weights = tuple(weights)
    assert weights, "weight group must not be empty"
    rho = weights[0].shape[0]
    dtype = weights[0].dtype
    assert all(weight.ndim == 2 for weight in weights), "weight ranks must be 2"
    assert all(
        weight.shape[0] == rho for weight in weights
    ), "weight input dimensions must match"
    assert all(weight.dtype == dtype for weight in weights), "weight dtypes must match"
    output_widths = tuple(weight.shape[1] for weight in weights)
    combined_weight = jnp.concatenate(weights, axis=1)
    return mjv_model_setup(
        combined_weight,
        model_id=model_id,
        output_widths=output_widths,
        sum_r=sum_r,
        linear_family=linear_family,
    )


def mjv_linear_group_online(
    activation,
    request_handle,
    *,
    model_id,
    request_id,
    output_widths,
    logical_row_members=0,
):

    assert 2 <= activation.ndim <= 3, "activation rank must be 2 or 3"
    assert request_handle.ndim == 2, "request handle rank must be 2"
    assert (
        activation.shape[-1] == request_handle.shape[1]
    ), "activation width must match rho"
    leading_shape = activation.shape[:-1]
    activation_rows = activation.reshape((-1, activation.shape[-1]))
    assert (
        activation_rows.shape == request_handle.shape
    ), "request rows must match flattened activation rows"
    raw_outputs = mjv_group_online(
        request_handle,
        activation_rows,
        model_id=model_id,
        request_id=request_id,
        output_widths=output_widths,
        logical_row_members=logical_row_members,
    )
    return tuple(
        output.reshape(leading_shape + (output.shape[-1],)) for output in raw_outputs
    )


def _mjv_model_setup_abstract(
    weight, *, model_id, output_widths, sum_r, linear_family, runtime_metadata
):
    del model_id, linear_family, runtime_metadata
    assert len(weight.shape) == 2
    assert sum(output_widths) == weight.shape[1]
    assert sum_r >= weight.shape[0]
    return ShapedArray(weight.shape, dtypes.canonicalize_dtype(weight.dtype))


def _mjv_model_setup_lowering(
    ctx,
    weight,
    *,
    model_id,
    output_widths,
    sum_r,
    linear_family,
    runtime_metadata,
):
    del ctx
    weight_type = mlir.ir.RankedTensorType(weight.type)
    metadata_kwargs = {}
    if runtime_metadata is not None:
        (
            plan_sha256,
            model_epoch,
            physical_family_id,
            logical_object_id,
            logical_members,
            lifetime_code,
            operand_code,
            family_kind,
            shared_shape,
            right_shape,
        ) = runtime_metadata
        metadata_kwargs = dict(
            plan_sha256=plan_sha256,
            model_epoch=model_epoch,
            physical_family_id=physical_family_id,
            logical_object_id=logical_object_id,
            logical_members=logical_members,
            object_lifetime={"s": "model-static", "r": "request-immutable"}[
                lifetime_code
            ],
            object_operand={"l": "lhs", "r": "rhs"}[operand_code],
            family_kind=family_kind,
            shared_shape=shared_shape,
            right_shape=right_shape,
        )
    target = model_setup_target(
        model_id=model_id,
        output_widths=output_widths,
        sum_r=sum_r,
        linear_family=linear_family,
        **metadata_kwargs,
    )
    return custom_call(
        target,
        result_types=[weight_type],
        operands=[weight],
        has_side_effect=True,
    ).results


def _mjv_request_preprocess_abstract(
    model_handle,
    *,
    model_id,
    request_id,
    rows,
    plan_sha256,
    model_epoch,
    request_epoch,
    physical_family_id,
):
    del model_id, request_id, plan_sha256, model_epoch, request_epoch, physical_family_id
    assert len(model_handle.shape) == 2
    return ShapedArray(
        (rows, model_handle.shape[0]),
        dtypes.canonicalize_dtype(model_handle.dtype),
    )


def _mjv_request_preprocess_lowering(
    ctx,
    model_handle,
    *,
    model_id,
    request_id,
    rows,
    plan_sha256,
    model_epoch,
    request_epoch,
    physical_family_id,
):
    del ctx
    model_type = mlir.ir.RankedTensorType(model_handle.type)
    result_type = mlir.ir.RankedTensorType.get(
        (rows, model_type.shape[0]), model_type.element_type
    )
    return custom_call(
        request_preprocess_target(
            model_id=model_id,
            request_id=request_id,
            rows=rows,
            plan_sha256=plan_sha256,
            model_epoch=model_epoch,
            request_epoch=request_epoch,
            physical_family_id=physical_family_id,
        ),
        result_types=[result_type],
        operands=[model_handle],
        has_side_effect=True,
    ).results


def _mjv_group_online_abstract(
    request_handle,
    joint_activation,
    *,
    model_id,
    request_id,
    output_widths,
    logical_row_members,
):
    del model_id, request_id, logical_row_members
    request_dtype = dtypes.canonicalize_dtype(request_handle.dtype)
    assert dtypes.canonicalize_dtype(joint_activation.dtype) == request_dtype
    assert request_handle.shape == joint_activation.shape
    return tuple(
        ShapedArray((request_handle.shape[0], width), request_dtype)
        for width in output_widths
    )


def _mjv_group_online_lowering(
    ctx,
    request_handle,
    joint_activation,
    *,
    model_id,
    request_id,
    output_widths,
    logical_row_members,
):
    del ctx
    request_type = mlir.ir.RankedTensorType(request_handle.type)
    result_types = [
        mlir.ir.RankedTensorType.get(
            (request_type.shape[0], width), request_type.element_type
        )
        for width in output_widths
    ]
    target = (
        f"mjv_group_online:{model_id}:{request_id}:" f"{_width_text(output_widths)}"
    )
    if logical_row_members:
        target += f":{logical_row_members}"
    return custom_call(
        target,
        result_types=result_types,
        operands=[request_handle, joint_activation],
        has_side_effect=True,
    ).results


_mjv_model_setup_prim = core.Primitive("mjv_model_setup")
_mjv_model_setup_prim.multiple_results = False
_mjv_model_setup_prim.def_impl(partial(xla.apply_primitive, _mjv_model_setup_prim))
_mjv_model_setup_prim.def_abstract_eval(_mjv_model_setup_abstract)
mlir.register_lowering(_mjv_model_setup_prim, _mjv_model_setup_lowering)

_mjv_request_preprocess_prim = core.Primitive("mjv_request_preprocess")
_mjv_request_preprocess_prim.multiple_results = False
_mjv_request_preprocess_prim.def_impl(
    partial(xla.apply_primitive, _mjv_request_preprocess_prim)
)
_mjv_request_preprocess_prim.def_abstract_eval(_mjv_request_preprocess_abstract)
mlir.register_lowering(_mjv_request_preprocess_prim, _mjv_request_preprocess_lowering)

_mjv_group_online_prim = core.Primitive("mjv_group_online")
_mjv_group_online_prim.multiple_results = True
_mjv_group_online_prim.def_impl(partial(xla.apply_primitive, _mjv_group_online_prim))
_mjv_group_online_prim.def_abstract_eval(_mjv_group_online_abstract)
mlir.register_lowering(_mjv_group_online_prim, _mjv_group_online_lowering)
