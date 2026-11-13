# Modified for Riblet from OpenBumbleBee's Flax activation adapter (Copyright 2023 Ant Group Co., Ltd.).
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


CDI_NONLINEAR_MODES = ("joint",)


def native_segment_prefix(params, token_type_ids, *, jnp):

    table = params["bert"]["embeddings"]["token_type_embeddings"]["embedding"]
    if table.shape != (2, 768):
        raise ValueError("unexpected Native BERT segment table")
    return jnp.take(table, token_type_ids, axis=0)


@contextmanager
def use_native_segment_prefix(embed_class, value):

    original = embed_class.__call__

    def call(module, inputs):
        if module.name != "token_type_embeddings":
            return original(module, inputs)
        if value.shape != (*inputs.shape, 768):
            raise ValueError("Native segment prefix shape mismatch")
        return value

    embed_class.__call__ = call
    try:
        yield
    finally:
        embed_class.__call__ = original


def mode_uses_operator(mode: str, operator: str) -> bool:
    if mode not in CDI_NONLINEAR_MODES:
        raise ValueError(f"unsupported CDI nonlinear mode: {mode}")
    if operator not in ("reciprocal", "rsqrt"):
        raise ValueError(f"unsupported CDI operator: {operator}")
    return True


def _expected_site_ids(operator: str) -> tuple[tuple[str, ...], ...] | tuple[str, ...]:
    if operator == "reciprocal":
        return tuple(
            tuple(f"layer={layer}|head={head}" for head in range(12)) for layer in range(12)
        )
    if operator == "rsqrt":
        return tuple(f"layernorm_site={site}" for site in range(25))
    raise ValueError(f"unsupported CDI operator: {operator}")


def _compile_site_indices(
    operator: str, public_site_ids: tuple[str, ...]
) -> tuple[tuple[int, ...], ...] | tuple[int, ...]:
    if len(set(public_site_ids)) != len(public_site_ids):
        raise ValueError("CDI public-site order contains duplicates")
    index = {site_id: position for position, site_id in enumerate(public_site_ids)}
    expected = _expected_site_ids(operator)
    flat_expected = (
        tuple(site for layer in expected for site in layer)
        if operator == "reciprocal"
        else expected
    )
    if set(index) != set(flat_expected):
        raise ValueError(f"{operator} public-site registry is incomplete")
    if operator == "reciprocal":
        return tuple(tuple(index[site] for site in layer) for layer in expected)
    return tuple(index[site] for site in expected)


@dataclass(frozen=True)
class OperatorRuntimeGroup:

    operator: str
    public_site_ids: tuple[str, ...]
    public_ks: object
    routes: object
    coefficients: object

    def __post_init__(self) -> None:
        site_ids = tuple(str(value) for value in self.public_site_ids)
        object.__setattr__(self, "public_site_ids", site_ids)
        group_size = 144 if self.operator == "reciprocal" else 25
        if len(site_ids) != group_size:
            raise ValueError(f"{self.operator} runtime group must contain {group_size} sites")
        if tuple(self.public_ks.shape) != (group_size,):
            raise ValueError("public K vector shape is inconsistent")
        if len(self.routes.shape) != 2 or self.routes.shape[1] != group_size:
            raise ValueError("route matrix shape is inconsistent")
        token_count = int(self.routes.shape[0])
        if tuple(self.coefficients.shape) != (token_count, group_size, 2):
            raise ValueError("coefficient tensor shape is inconsistent")
        _compile_site_indices(self.operator, site_ids)

    @property
    def token_count(self) -> int:
        return int(self.routes.shape[0])

    @property
    def site_indices(self):
        return _compile_site_indices(self.operator, self.public_site_ids)


@dataclass(frozen=True)
class CdiBertExecutionPlan:

    mode: str
    public_groups: tuple[tuple[str, tuple[str, ...], object], ...]

    @classmethod
    def from_public_groups(cls, mode: str, groups):

        if mode not in CDI_NONLINEAR_MODES:
            raise ValueError(f"unsupported CDI nonlinear mode: {mode}")
        expected = tuple(
            operator for operator in ("reciprocal", "rsqrt") if mode_uses_operator(mode, operator)
        )
        if set(groups) != set(expected):
            raise ValueError("CDI public groups do not match the selected mode")
        normalized = []
        for operator in expected:
            site_ids, public_ks = groups[operator]
            site_ids = tuple(str(value) for value in site_ids)
            _compile_site_indices(operator, site_ids)
            normalized.append((operator, site_ids, _selected_ks(public_ks)))
        return cls(mode=mode, public_groups=tuple(normalized))

    @property
    def runtime_argument_count(self) -> int:
        return 2 * len(self.public_groups)

    def bind_runtime_arguments(self, jnp, intrinsic, *arguments):
        if len(arguments) != self.runtime_argument_count:
            raise ValueError(
                f"CDI mode {self.mode} expects {self.runtime_argument_count} "
                f"runtime arguments, got {len(arguments)}"
            )
        bound = {}
        offset = 0
        for operator, site_ids, public_ks in self.public_groups:
            routes, coefficients = arguments[offset : offset + 2]
            offset += 2
            bound[operator] = OperatorRuntimeGroup(
                operator=operator,
                public_site_ids=site_ids,
                public_ks=public_ks,
                routes=routes,
                coefficients=coefficients,
            )
        return CdiBertRuntimeContext(
            jnp=jnp,
            intrinsic=intrinsic,
            reciprocal=bound.get("reciprocal"),
            rsqrt=bound.get("rsqrt"),
        )


class CdiBertRuntimeContext:

    def __init__(self, *, jnp, intrinsic, reciprocal=None, rsqrt=None):
        if reciprocal is not None and reciprocal.operator != "reciprocal":
            raise ValueError("reciprocal runtime group has the wrong operator")
        if rsqrt is not None and rsqrt.operator != "rsqrt":
            raise ValueError("RSqrt runtime group has the wrong operator")
        counts = {group.token_count for group in (reciprocal, rsqrt) if group is not None}
        if len(counts) > 1:
            raise ValueError("CDI operator groups disagree on token count")
        self.jnp = jnp
        self.intrinsic = intrinsic
        self.reciprocal = reciprocal
        self.rsqrt = rsqrt
        self.reciprocal_calls = 0
        self.rsqrt_calls = 0
        self.allow_native = False
        self.effective_length = next(iter(counts)) if counts else None
        self.physical_length = self.effective_length

    def enable_execution(
        self,
        *,
        native_rsqrt,
        effective_length=None,
        physical_length=None,
    ):
        from padding128 import extend_selection

        n = self.effective_length if effective_length is None else int(effective_length)
        physical = n if physical_length is None else int(physical_length)
        if n is not None:
            if not 0 < n <= physical <= 128:
                raise ValueError("invalid public effective/physical length")
            for name in ("reciprocal", "rsqrt"):
                group = getattr(self, name)
                if group is not None:
                    if group.token_count != n:
                        raise ValueError("public schedule length must equal effective length")
                    routes, coefficients = extend_selection(
                        group.routes, group.coefficients, physical, jnp=self.jnp
                    )
                    setattr(
                        self,
                        name,
                        OperatorRuntimeGroup(
                            group.operator,
                            group.public_site_ids,
                            group.public_ks,
                            routes,
                            coefficients,
                        ),
                    )
        self.effective_length = n
        self.physical_length = physical
        self.allow_native = True
        self.native_rsqrt = native_rsqrt

    def reciprocal_inverse(self, divisor):
        if self.physical_length is not None and tuple(divisor.shape) != (
            1,
            12,
            self.physical_length,
            1,
        ):
            raise ValueError("physical reciprocal shape differs from public execution plan")
        group = self.reciprocal
        if group is None:
            if self.allow_native:
                self.reciprocal_calls += 1
                return self.jnp.reciprocal(divisor)
            raise ValueError("reciprocal CDI is not active")
        layer = self.reciprocal_calls
        if layer >= 12:
            raise ValueError("model invoked more than 12 reciprocal sites")
        expected_shape = (1, 12, group.token_count, 1)
        if tuple(divisor.shape) != expected_shape:
            raise ValueError(
                f"Softmax divisor must have shape {expected_shape}, got {divisor.shape}"
            )
        indices = self.jnp.asarray(group.site_indices[layer], dtype=self.jnp.int64)
        coefficients = self.jnp.take(group.coefficients, indices, axis=1)
        coefficients = self.jnp.transpose(coefficients, (1, 0, 2))[None, :, :, None, :]
        routes = self.jnp.take(group.routes, indices, axis=1)
        routes = self.jnp.transpose(routes, (1, 0))[None, :, :, None]
        public_ks = self.jnp.take(self.jnp.asarray(group.public_ks, dtype=self.jnp.int64), indices)
        public_ks = self.jnp.broadcast_to(public_ks[None, :, None, None], expected_shape)
        if self.effective_length is not None:
            from padding128 import pad_ks

            public_ks = pad_ks(public_ks, self.effective_length, token_axis=2, jnp=self.jnp)
        self.reciprocal_calls += 1
        return self.intrinsic.cdi_invert(
            divisor,
            coefficients,
            routes,
            public_ks,
            operator="reciprocal",
        )

    def rsqrt_inverse(self, operand):
        if self.physical_length is not None and tuple(operand.shape) != (
            1,
            self.physical_length,
            1,
        ):
            raise ValueError("physical RSqrt shape differs from public execution plan")
        group = self.rsqrt
        if group is None:
            if self.allow_native:
                self.rsqrt_calls += 1
                return self.native_rsqrt(operand)
            raise ValueError("RSqrt CDI is not active")
        site = self.rsqrt_calls
        if site >= 25:
            raise ValueError("model invoked more than 25 RSqrt sites")
        expected_shape = (1, group.token_count, 1)
        if tuple(operand.shape) != expected_shape:
            raise ValueError(
                f"LayerNorm operand must have shape {expected_shape}, got {operand.shape}"
            )
        index = group.site_indices[site]
        coefficients = group.coefficients[:, index, :][None, :, None, :]
        routes = group.routes[:, index][None, :, None]
        public_ks = self.jnp.broadcast_to(
            self.jnp.asarray(group.public_ks[index], dtype=self.jnp.int64),
            expected_shape,
        )
        if self.effective_length is not None:
            from padding128 import pad_ks

            public_ks = pad_ks(public_ks, self.effective_length, token_axis=1, jnp=self.jnp)
        self.rsqrt_calls += 1
        return self.intrinsic.cdi_invert(
            operand,
            coefficients,
            routes,
            public_ks,
            operator="rsqrt",
        )

    def assert_complete(self) -> None:
        if (self.reciprocal is not None or self.allow_native) and self.reciprocal_calls != 12:
            raise ValueError(f"model consumed {self.reciprocal_calls} of 12 reciprocal sites")
        if (self.rsqrt is not None or self.allow_native) and self.rsqrt_calls != 25:
            raise ValueError(f"model consumed {self.rsqrt_calls} of 25 RSqrt sites")


def secure_softmax(value, *, axis, where, initial, jnp, intrinsic, runtime):

    maximum = jnp.max(value, axis, where=where, initial=initial, keepdims=True)
    exponentials = intrinsic.spu_neg_exp(value - maximum)
    divisor = jnp.sum(exponentials, axis, where=where, keepdims=True)
    if runtime is None:
        return exponentials / divisor
    return exponentials * runtime.reciprocal_inverse(divisor)


@contextmanager
def hijack_layernorm_rsqrt(layernorm_class, normalization_module, runtime):

    if runtime is None:
        yield
        return
    original_call = layernorm_class.__call__

    def cdi_layernorm_call(module, value, *args, **kwargs):
        original_rsqrt = normalization_module.lax.rsqrt
        calls = 0

        def cdi_rsqrt(value):
            nonlocal calls
            calls += 1
            return runtime.rsqrt_inverse(value)

        normalization_module.lax.rsqrt = cdi_rsqrt
        try:
            result = original_call(module, value, *args, **kwargs)
        finally:
            normalization_module.lax.rsqrt = original_rsqrt
        if calls != 1:
            raise ValueError("each BERT LayerNorm must invoke rsqrt exactly once")
        return result

    layernorm_class.__call__ = cdi_layernorm_call
    try:
        yield
    finally:
        layernorm_class.__call__ = original_call


@contextmanager
def native_nonlinear_shells(fnn, normalization, jnn, jnp, intrinsic, runtime):

    from transformers.models.bert import modeling_flax_bert as bert

    originals = jnn.gelu, fnn.gelu, jnn.softmax, fnn.softmax, bert.ACT2FN["gelu"]

    def gelu(value, approximate=False):
        if approximate:
            raise ValueError("only the frozen BERT GELU variant is supported")
        return intrinsic.spu_gelu(value)

    def softmax(value, axis=-1, where=None, initial=None):
        return secure_softmax(
            value,
            axis=axis,
            where=where,
            initial=initial,
            jnp=jnp,
            intrinsic=intrinsic,
            runtime=runtime,
        )

    jnn.gelu = fnn.gelu = bert.ACT2FN["gelu"] = gelu
    jnn.softmax = fnn.softmax = softmax
    try:
        with hijack_layernorm_rsqrt(fnn.LayerNorm, normalization, runtime):
            yield
    finally:
        jnn.gelu, fnn.gelu, jnn.softmax, fnn.softmax, bert.ACT2FN["gelu"] = originals


def _selected_ks(values):
    original = np.asarray(values)
    if original.ndim != 1 or not np.issubdtype(original.dtype, np.integer):
        raise ValueError("full Riblet accepts Selected K only")
    if np.any(original < 0) or np.any(original > 8):
        raise ValueError("public K outside 0..8")
    result = original.astype(np.int64)
    result.setflags(write=False)
    return result
