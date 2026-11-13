__all__ = [
    "RUNTIME_OBJECT_AUDIT_V1_FIELDS",
    "mjv_plan_register",
    "mjv_linear_manifest",
    "mjv_model_setup_batch",
    "mjv_request_preprocess_batch",
    "mjv_epoch_begin",
    "mjv_epoch_end",
    "mjv_audit_snapshot",
]

from functools import partial

from jax import core
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call
import numpy as np


RUNTIME_OBJECT_AUDIT_V1_FIELDS = (
    "schema_version",
    "plan_matches",
    "model_epoch",
    "request_epoch",
    "logical_consumer_count",
    "physical_family_count",
    "model_required_shared_object_contexts",
    "model_shared_side_logical_openings",
    "model_shared_side_opened_elements",
    "model_shared_side_physical_collectives",
    "model_setup_right_logical_openings",
    "model_setup_right_opened_elements",
    "model_setup_right_physical_collectives",
    "request_required_shared_object_contexts",
    "request_shared_side_logical_openings",
    "request_shared_side_opened_elements",
    "request_shared_side_physical_collectives",
    "request_online_right_logical_openings",
    "request_online_right_opened_elements",
    "request_online_right_physical_collectives",
    "embedding_physical_executions",
    "model_setup_rpc_calls",
    "request_preprocess_rpc_calls",
    "consumed_logical_consumers",
    "expected_logical_consumers",
    "fresh_request_masks",
    "static_object_reused",
    "secure_invoke_rpc_calls",
    "protocol_matrix_calls",
    "dot_ole_calls",
    "batch_dot_ole_calls",
    "live_object_contexts",
    "live_consumer_contexts",
)


def _plan_hash(value):
    value = str(value)
    assert len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    ), "plan must be a 64-character lowercase hex digest"
    return value


def _epoch(value, name):
    value = int(value)
    assert value >= 0, f"{name} must be nonnegative"
    return value


def plan_register_target(plan_sha256):
    return f"mjv_plan_register:{_plan_hash(plan_sha256)}"


def model_setup_batch_target(plan_sha256, model_epoch):
    return (
        f"mjv_model_setup_batch:{_plan_hash(plan_sha256)}:"
        f"{_epoch(model_epoch, 'model_epoch')}"
    )


def request_preprocess_batch_target(plan_sha256, model_epoch, request_epoch):
    return (
        f"mjv_request_preprocess_batch:{_plan_hash(plan_sha256)}:"
        f"{_epoch(model_epoch, 'model_epoch')}:"
        f"{_epoch(request_epoch, 'request_epoch')}"
    )


def epoch_begin_target(plan_sha256, model_epoch, request_epoch):
    return (
        f"mjv_epoch_begin:{_plan_hash(plan_sha256)}:"
        f"{_epoch(model_epoch, 'model_epoch')}:"
        f"{_epoch(request_epoch, 'request_epoch')}"
    )


def epoch_end_target(
    plan_sha256, model_epoch, request_epoch, release_model_epoch=False
):
    target = (
        f"mjv_epoch_end:{_plan_hash(plan_sha256)}:"
        f"{_epoch(model_epoch, 'model_epoch')}:"
        f"{_epoch(request_epoch, 'request_epoch')}"
    )
    return f"{target}:release_model" if release_model_epoch else target


def audit_snapshot_target(plan_sha256, model_epoch, request_epoch):
    return (
        f"mjv_audit_snapshot:{_plan_hash(plan_sha256)}:"
        f"{_epoch(model_epoch, 'model_epoch')}:"
        f"{_epoch(request_epoch, 'request_epoch')}"
    )


def _bind_anchor(primitive, anchor, *, target):
    operands = () if anchor is None else (anchor,)
    outputs = primitive.bind(*operands, target=target)
    return None if anchor is None else outputs[0]


def mjv_plan_register(anchor=None, *, plan_sha256):
    return _bind_anchor(
        _mjv_context_call_prim,
        anchor,
        target=plan_register_target(plan_sha256),
    )


def mjv_model_setup_batch(anchor=None, *, plan_sha256, model_epoch):
    return _bind_anchor(
        _mjv_context_call_prim,
        anchor,
        target=model_setup_batch_target(plan_sha256, model_epoch),
    )


def mjv_request_preprocess_batch(
    anchor=None, *, plan_sha256, model_epoch, request_epoch
):
    return _bind_anchor(
        _mjv_context_call_prim,
        anchor,
        target=request_preprocess_batch_target(
            plan_sha256, model_epoch, request_epoch
        ),
    )


def mjv_epoch_begin(anchor=None, *, plan_sha256, model_epoch, request_epoch):
    return _bind_anchor(
        _mjv_epoch_begin_prim,
        anchor,
        target=epoch_begin_target(plan_sha256, model_epoch, request_epoch),
    )


def mjv_epoch_end(
    anchor,
    audit,
    *,
    plan_sha256,
    model_epoch,
    request_epoch,
    release_model_epoch=False,
):
    return _mjv_epoch_end_prim.bind(
        anchor,
        audit,
        target=epoch_end_target(
            plan_sha256,
            model_epoch,
            request_epoch,
            release_model_epoch=release_model_epoch,
        ),
    )[0]


def mjv_audit_snapshot(anchor=None, *, plan_sha256, model_epoch, request_epoch):
    operands = () if anchor is None else (anchor,)
    return _mjv_audit_snapshot_prim.bind(
        *operands,
        target=audit_snapshot_target(plan_sha256, model_epoch, request_epoch),
    )


def mjv_linear_manifest(anchor, *, plan_sha256, records):
    _plan_hash(plan_sha256)
    if not records or any(c not in '0123456789,;' for c in records):
        raise ValueError('invalid public linear fingerprint encoding')
    return _mjv_context_call_prim.bind(anchor,target=f'mjv_linear_manifest:{plan_sha256}:{records}')[0]


def _anchor_abstract(*operands, target):
    assert target.startswith("mjv_"), "invalid MJV context target"
    assert len(operands) <= 1, "MJV context calls accept at most one anchor"
    return () if not operands else (operands[0],)


def _anchor_lowering(ctx, *operands, target):
    result_types = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
    return custom_call(
        target,
        result_types=result_types,
        operands=list(operands),
        has_side_effect=True,
    ).results


def _audit_snapshot_abstract(*operands, target):
    assert target.startswith("mjv_audit_snapshot:"), "invalid audit target"
    assert len(operands) <= 1, "MJV audit accepts at most one anchor"
    return core.ShapedArray(
        (len(RUNTIME_OBJECT_AUDIT_V1_FIELDS),), np.dtype(np.int64)
    )


def _audit_snapshot_lowering(ctx, *operands, target):
    return custom_call(
        target,
        result_types=[mlir.aval_to_ir_type(ctx.avals_out[0])],
        operands=list(operands),
        has_side_effect=True,
    ).results


def _epoch_end_abstract(anchor, audit, *, target):
    assert target.startswith("mjv_epoch_end:"), "invalid epoch end target"
    return (anchor,)


def _epoch_end_lowering(ctx, anchor, audit, *, target):
    return custom_call(
        target,
        result_types=[mlir.aval_to_ir_type(ctx.avals_out[0])],
        operands=[anchor, audit],
        has_side_effect=True,
    ).results


def _primitive(name):
    primitive = core.Primitive(name)
    primitive.multiple_results = True
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_anchor_abstract)
    mlir.register_lowering(primitive, _anchor_lowering)
    return primitive


_mjv_context_call_prim = _primitive("mjv_context_call")
_mjv_epoch_begin_prim = _primitive("mjv_epoch_begin")

_mjv_epoch_end_prim = core.Primitive("mjv_epoch_end")
_mjv_epoch_end_prim.multiple_results = True
_mjv_epoch_end_prim.def_impl(partial(xla.apply_primitive, _mjv_epoch_end_prim))
_mjv_epoch_end_prim.def_abstract_eval(_epoch_end_abstract)
mlir.register_lowering(_mjv_epoch_end_prim, _epoch_end_lowering)

_mjv_audit_snapshot_prim = core.Primitive("mjv_audit_snapshot")
_mjv_audit_snapshot_prim.multiple_results = False
_mjv_audit_snapshot_prim.def_impl(
    partial(xla.apply_primitive, _mjv_audit_snapshot_prim)
)
_mjv_audit_snapshot_prim.def_abstract_eval(_audit_snapshot_abstract)
mlir.register_lowering(_mjv_audit_snapshot_prim, _audit_snapshot_lowering)
