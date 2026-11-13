from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


Shape = tuple[int, ...]


class Lifetime(str, Enum):
    MODEL_STATIC = "model-static"
    REQUEST_IMMUTABLE = "request-immutable"
    APPEND_ONLY = "append-only"


def _shape(value: Iterable[int], *, name: str) -> Shape:
    result = tuple(int(dimension) for dimension in value)
    if not result or any(dimension <= 0 for dimension in result):
        raise ValueError(f"{name} must contain positive dimensions")
    return result


@dataclass(frozen=True)
class PublicLinearMapStep:
    kind: str
    domain_shape: Shape
    codomain_shape: Shape
    parameters: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "domain_shape": list(self.domain_shape),
            "codomain_shape": list(self.codomain_shape),
            "parameters": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.parameters
            },
        }


@dataclass(frozen=True)
class PublicLinearMap:
    domain_shape: Shape
    codomain_shape: Shape
    rank: int
    steps: tuple[PublicLinearMapStep, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "domain_shape", _shape(self.domain_shape, name="domain_shape"))
        object.__setattr__(self, "codomain_shape", _shape(self.codomain_shape, name="codomain_shape"))
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        expected = self.domain_shape
        for step in self.steps:
            if step.domain_shape != expected:
                raise ValueError("public linear map steps are dimension-inconsistent")
            expected = step.codomain_shape
        if expected != self.codomain_shape:
            raise ValueError("public linear map codomain does not match its steps")

    @classmethod
    def identity(cls, shape: Iterable[int], *, rank: int) -> "PublicLinearMap":
        normalized = _shape(shape, name="shape")
        return cls(normalized, normalized, int(rank))

    @classmethod
    def reshape(
        cls,
        domain_shape: Iterable[int],
        codomain_shape: Iterable[int],
        *,
        rank: int,
    ) -> "PublicLinearMap":
        domain = _shape(domain_shape, name="domain_shape")
        codomain = _shape(codomain_shape, name="codomain_shape")
        if math.prod(domain) != math.prod(codomain):
            raise ValueError("reshape must preserve the number of elements")
        step = PublicLinearMapStep("reshape", domain, codomain)
        return cls(domain, codomain, int(rank), (step,))

    @classmethod
    def transpose(
        cls,
        domain_shape: Iterable[int],
        permutation: Iterable[int],
        *,
        rank: int,
    ) -> "PublicLinearMap":
        domain = _shape(domain_shape, name="domain_shape")
        axes = tuple(int(axis) for axis in permutation)
        if sorted(axes) != list(range(len(domain))):
            raise ValueError("transpose permutation must contain every axis exactly once")
        codomain = tuple(domain[axis] for axis in axes)
        step = PublicLinearMapStep(
            "transpose",
            domain,
            codomain,
            (("permutation", axes),),
        )
        return cls(domain, codomain, int(rank), (step,))

    @classmethod
    def slice(
        cls,
        domain_shape: Iterable[int],
        *,
        starts: Iterable[int],
        limits: Iterable[int],
        rank: int,
        strides: Iterable[int] | None = None,
    ) -> "PublicLinearMap":
        domain = _shape(domain_shape, name="domain_shape")
        start_values = tuple(int(value) for value in starts)
        limit_values = tuple(int(value) for value in limits)
        stride_values = (
            tuple(1 for _ in domain)
            if strides is None
            else tuple(int(value) for value in strides)
        )
        if not (
            len(start_values)
            == len(limit_values)
            == len(stride_values)
            == len(domain)
        ):
            raise ValueError("slice parameters must match the domain rank")
        if any(stride <= 0 for stride in stride_values):
            raise ValueError("slice strides must be positive")
        if any(
            start < 0 or limit > size or start >= limit
            for start, limit, size in zip(start_values, limit_values, domain)
        ):
            raise ValueError("slice bounds must select a non-empty in-range view")
        codomain = tuple(
            (limit - start + stride - 1) // stride
            for start, limit, stride in zip(
                start_values, limit_values, stride_values
            )
        )
        step = PublicLinearMapStep(
            "slice",
            domain,
            codomain,
            (
                ("starts", start_values),
                ("limits", limit_values),
                ("strides", stride_values),
            ),
        )
        return cls(domain, codomain, int(rank), (step,))

    def then(self, following: "PublicLinearMap", *, rank: int | None = None) -> "PublicLinearMap":
        if self.codomain_shape != following.domain_shape:
            raise ValueError("public linear maps are dimension-inconsistent")
        return PublicLinearMap(
            domain_shape=self.domain_shape,
            codomain_shape=following.codomain_shape,
            rank=following.rank if rank is None else int(rank),
            steps=self.steps + following.steps,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_shape": list(self.domain_shape),
            "codomain_shape": list(self.codomain_shape),
            "rank": self.rank,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class ObjectContextSpec:
    logical_object_id: str
    lifetime: Lifetime
    epoch_kind: str
    object_shape: Shape
    joint_view: PublicLinearMap

    def __post_init__(self):
        if not self.logical_object_id:
            raise ValueError("logical_object_id must not be empty")
        object.__setattr__(self, "object_shape", _shape(self.object_shape, name="object_shape"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_object_id": self.logical_object_id,
            "lifetime": self.lifetime.value,
            "epoch_kind": self.epoch_kind,
            "object_shape": list(self.object_shape),
            "joint_view": self.joint_view.to_dict(),
        }


@dataclass(frozen=True)
class ConsumerContextSpec:
    consumer_id: str
    ordinal: int
    logical_object_id: str
    family: str
    native_view: PublicLinearMap
    lift: PublicLinearMap
    rhs_shape: Shape
    schedule: str
    role: str = ""
    object_operand: str = "lhs"

    def __post_init__(self):
        if not self.consumer_id:
            raise ValueError("consumer_id must not be empty")
        if self.ordinal < 0:
            raise ValueError("consumer ordinal must be non-negative")
        if not self.logical_object_id:
            raise ValueError("logical_object_id must not be empty")
        object.__setattr__(self, "rhs_shape", _shape(self.rhs_shape, name="rhs_shape"))
        if self.object_operand not in ("lhs", "rhs"):
            raise ValueError("object_operand must be lhs or rhs")

    @classmethod
    def dense(
        cls,
        *,
        consumer_id: str,
        ordinal: int = 0,
        logical_object_id: str,
        native_view: PublicLinearMap,
        lift: PublicLinearMap,
        rhs_shape: Iterable[int],
        schedule: str,
        role: str = "",
        object_operand: str = "lhs",
    ) -> "ConsumerContextSpec":
        return cls(
            consumer_id=consumer_id,
            ordinal=int(ordinal),
            logical_object_id=logical_object_id,
            family="dense",
            native_view=native_view,
            lift=lift,
            rhs_shape=tuple(rhs_shape),
            schedule=schedule,
            role=role,
            object_operand=object_operand,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "ordinal": self.ordinal,
            "logical_object_id": self.logical_object_id,
            "family": self.family,
            "native_view": self.native_view.to_dict(),
            "lift": self.lift.to_dict(),
            "rhs_shape": list(self.rhs_shape),
            "schedule": self.schedule,
            "role": self.role,
            "object_operand": self.object_operand,
        }


@dataclass(frozen=True)
class DynamicLinearCall:
    ordinal: int
    source_primitive: str
    source_ordinal: int
    kind: str
    logical_object_id: str
    lhs_shape: Shape
    rhs_shape: Shape
    batched: bool
    object_operand: str
    role: str

    def __post_init__(self):
        if self.ordinal < 0 or self.source_ordinal < 0:
            raise ValueError("dynamic call ordinals must be non-negative")
        if not self.source_primitive or not self.logical_object_id:
            raise ValueError("dynamic calls require source and object lineage")
        object.__setattr__(self, "lhs_shape", _shape(self.lhs_shape, name="lhs_shape"))
        object.__setattr__(self, "rhs_shape", _shape(self.rhs_shape, name="rhs_shape"))
        if self.object_operand not in ("lhs", "rhs"):
            raise ValueError("dynamic object_operand must be lhs or rhs")


@dataclass(frozen=True)
class PhysicalFamilySpec:
    physical_family_id: str
    kind: str
    logical_consumer_ids: tuple[str, ...]
    lhs_shape: Shape
    rhs_shape: Shape

    def __post_init__(self):
        if not self.physical_family_id or not self.logical_consumer_ids:
            raise ValueError("physical families require an ID and logical members")
        object.__setattr__(
            self, "logical_consumer_ids", tuple(self.logical_consumer_ids)
        )
        object.__setattr__(self, "lhs_shape", _shape(self.lhs_shape, name="lhs_shape"))
        object.__setattr__(self, "rhs_shape", _shape(self.rhs_shape, name="rhs_shape"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_family_id": self.physical_family_id,
            "kind": self.kind,
            "logical_consumer_ids": list(self.logical_consumer_ids),
            "lhs_shape": list(self.lhs_shape),
            "rhs_shape": list(self.rhs_shape),
        }


@dataclass(frozen=True)
class BoundarySpec:
    primitive: str
    before_object_id: str
    after_object_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "primitive": self.primitive,
            "before_object_id": self.before_object_id,
            "after_object_id": self.after_object_id,
        }


@dataclass(frozen=True)
class LinearObjectPlan:
    objects: tuple[ObjectContextSpec, ...]
    consumers: tuple[ConsumerContextSpec, ...]
    physical_families: tuple[PhysicalFamilySpec, ...] = ()
    boundaries: tuple[BoundarySpec, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "consumers", tuple(self.consumers))
        object.__setattr__(self, "physical_families", tuple(self.physical_families))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))

    def objects_for_epoch(self, epoch_kind: str) -> tuple[ObjectContextSpec, ...]:
        return tuple(obj for obj in self.objects if obj.epoch_kind == epoch_kind)

    def object(self, logical_object_id: str) -> ObjectContextSpec:
        matches = tuple(
            obj for obj in self.objects if obj.logical_object_id == logical_object_id
        )
        if len(matches) != 1:
            raise KeyError(logical_object_id)
        return matches[0]

    def consumers_of(self, logical_object_id: str) -> tuple[ConsumerContextSpec, ...]:
        return tuple(
            consumer
            for consumer in self.consumers
            if consumer.logical_object_id == logical_object_id
        )

    def consumers_by_family(self, family: str) -> tuple[ConsumerContextSpec, ...]:
        return tuple(
            consumer for consumer in self.consumers if consumer.family == family
        )

    def has_consumer(self, consumer_id: str) -> bool:
        return any(
            consumer.consumer_id == consumer_id for consumer in self.consumers
        )

    def physical_family(self, physical_family_id: str) -> PhysicalFamilySpec:
        matches = tuple(
            family
            for family in self.physical_families
            if family.physical_family_id == physical_family_id
        )
        if len(matches) != 1:
            raise KeyError(physical_family_id)
        return matches[0]

    def simultaneous_family(
        self, physical_family_id: str
    ) -> tuple[ConsumerContextSpec, ...]:
        family = self.physical_family(physical_family_id)
        consumers = {consumer.consumer_id: consumer for consumer in self.consumers}
        return tuple(consumers[consumer_id] for consumer_id in family.logical_consumer_ids)

    def objects_around(self, primitive: str) -> tuple[ObjectContextSpec, ObjectContextSpec]:
        matches = tuple(
            boundary for boundary in self.boundaries if boundary.primitive == primitive
        )
        if len(matches) != 1:
            raise KeyError(primitive)
        boundary = matches[0]
        return self.object(boundary.before_object_id), self.object(
            boundary.after_object_id
        )

    def merge(self, first_object_id: str, second_object_id: str):
        for boundary in self.boundaries:
            if {boundary.before_object_id, boundary.after_object_id} == {
                first_object_id,
                second_object_id,
            }:
                raise ValueError(
                    f"cannot fuse across boundary {boundary.primitive}"
                )
        raise ValueError("objects may merge only through a declared public view")

    @property
    def sum_r(self) -> int:
        return sum(consumer.native_view.rank for consumer in self.consumers)

    @property
    def rho(self) -> int:
        return sum(obj.joint_view.rank for obj in self.objects)

    @property
    def logical_consumer_count(self) -> int:
        return len(self.consumers)

    @property
    def physical_family_count(self) -> int:
        return len(self.physical_families)

    @property
    def physical_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for family in self.physical_families:
            counts[family.kind] = counts.get(family.kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_contexts": [obj.to_dict() for obj in self.objects],
            "consumer_contexts": [consumer.to_dict() for consumer in self.consumers],
            "physical_families": [
                family.to_dict() for family in self.physical_families
            ],
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


def _require_unique(values: Iterable[Any], *, name: str):
    values = tuple(values)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


def validate_plan(plan: LinearObjectPlan):
    _require_unique(
        (obj.logical_object_id for obj in plan.objects),
        name="logical object IDs",
    )
    _require_unique(
        (consumer.consumer_id for consumer in plan.consumers),
        name="consumer IDs",
    )
    _require_unique(
        (consumer.ordinal for consumer in plan.consumers),
        name="consumer ordinals",
    )
    _require_unique(
        (family.physical_family_id for family in plan.physical_families),
        name="physical family IDs",
    )

    objects = {obj.logical_object_id: obj for obj in plan.objects}
    for obj in plan.objects:
        if obj.lifetime is Lifetime.APPEND_ONLY:
            raise NotImplementedError(
                "append-only lifetime is deferred from v0.1"
            )
        expected_epoch = (
            "model" if obj.lifetime is Lifetime.MODEL_STATIC else "request"
        )
        if obj.epoch_kind != expected_epoch:
            raise ValueError(
                f"{obj.logical_object_id} lifetime requires {expected_epoch} epoch"
            )
        if obj.joint_view.domain_shape != obj.object_shape:
            raise ValueError("joint view domain must equal object shape")

    for consumer in plan.consumers:
        obj = objects.get(consumer.logical_object_id)
        if obj is None:
            raise ValueError(
                f"consumer {consumer.consumer_id} references an undeclared object"
            )
        if consumer.schedule not in ("simultaneous", "sequential"):
            raise ValueError("consumer schedule must be simultaneous or sequential")
        if consumer.native_view.domain_shape != obj.object_shape:
            raise ValueError("native view domain must equal object shape")
        try:
            lifted_joint_view = obj.joint_view.then(
                consumer.lift,
                rank=consumer.native_view.rank,
            )
        except ValueError as exc:
            raise ValueError("P_i must equal L_i U") from exc
        if lifted_joint_view != consumer.native_view:
            raise ValueError("P_i must equal L_i U")

    consumers = {consumer.consumer_id: consumer for consumer in plan.consumers}
    scheduled_consumers = []
    for family in plan.physical_families:
        for consumer_id in family.logical_consumer_ids:
            if consumer_id not in consumers:
                raise ValueError("physical family references an undeclared consumer")
            scheduled_consumers.append(consumer_id)
    if plan.physical_families:
        if len(scheduled_consumers) != len(set(scheduled_consumers)):
            raise ValueError("logical consumer appears in multiple physical families")
        if set(scheduled_consumers) != set(consumers):
            raise ValueError("physical schedule must cover every logical consumer")

    for boundary in plan.boundaries:
        if boundary.before_object_id not in objects or boundary.after_object_id not in objects:
            raise ValueError("boundary references an undeclared object")
        if boundary.before_object_id == boundary.after_object_id:
            raise ValueError("non-bilinear boundary must create a new object")


def hash_plan(plan: LinearObjectPlan) -> str:
    validate_plan(plan)
    return hashlib.sha256(plan.to_json().encode("utf-8")).hexdigest()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", ".", value).strip(".")


def dynamic_consumer_id(call: DynamicLinearCall) -> str:
    if call.kind == "embedding_row":
        row = call.role.removeprefix("row.")
        if not row.isdigit():
            raise ValueError("embedding row consumer requires a numeric row role")
        return f"word_embedding.row.{row}"
    if call.kind == "attention_matmul":
        return f"attention.{call.source_ordinal}.{call.role or call.ordinal}"
    raise ValueError(f"dynamic call {call.kind!r} has no runtime consumer context")


def _identity_object(
    logical_object_id: str,
    lifetime: Lifetime,
    shape: Shape,
    *,
    rank: int,
) -> ObjectContextSpec:
    epoch_kind = "model" if lifetime is Lifetime.MODEL_STATIC else "request"
    return ObjectContextSpec(
        logical_object_id=logical_object_id,
        lifetime=lifetime,
        epoch_kind=epoch_kind,
        object_shape=shape,
        joint_view=PublicLinearMap.identity(shape, rank=rank),
    )


def _dense_lineage_id(calls: tuple[Any, ...]) -> str:
    roles = tuple(call.scope_path.rsplit("/", 1)[-1] for call in calls)
    layer_matches = tuple(
        re.search(r"(?:^|/)layer/(\d+)(?:/|$)", call.scope_path)
        for call in calls
    )
    if roles == ("query", "key", "value") and all(layer_matches):
        layer_ids = {match.group(1) for match in layer_matches if match is not None}
        if len(layer_ids) == 1:
            return f"request.layer{next(iter(layer_ids))}.qkv_input"
    first = calls[0]
    return (
        f"request.dense_input.{first.input_key}."
        f"{_safe_id(first.scope_path)}"
    )


def _coerce_dynamic_call(value: DynamicLinearCall | dict[str, Any]) -> DynamicLinearCall:
    if isinstance(value, DynamicLinearCall):
        return value
    return DynamicLinearCall(
        ordinal=int(value["ordinal"]),
        source_primitive=str(value.get("source_primitive", value.get("kind", "unknown"))),
        source_ordinal=int(value.get("source_ordinal", value["ordinal"])),
        kind=str(value["kind"]),
        logical_object_id=str(value["logical_object_id"]),
        lhs_shape=tuple(value["lhs_shape"]),
        rhs_shape=tuple(value["rhs_shape"]),
        batched=bool(value.get("batched", False)),
        object_operand=str(value.get("object_operand", "lhs")),
        role=str(value.get("role", "")),
    )


def compile_linear_object_plan(
    dense_calls: Iterable[Any],
    dynamic_calls: Iterable[DynamicLinearCall | dict[str, Any]],
    *,
    boundaries: Iterable[BoundarySpec] = (),
) -> LinearObjectPlan:
    dense_calls = tuple(dense_calls)
    dynamic_calls = tuple(_coerce_dynamic_call(call) for call in dynamic_calls)
    if tuple(call.ordinal for call in dynamic_calls) != tuple(
        range(len(dynamic_calls))
    ):
        raise ValueError("dynamic call ordinals must be trace ordered")
    objects: list[ObjectContextSpec] = []
    consumers: list[ConsumerContextSpec] = []
    physical_families: list[PhysicalFamilySpec] = []

    dense_groups: dict[tuple[int, Shape], list[Any]] = {}
    dense_group_order: list[tuple[int, Shape]] = []
    for expected_ordinal, call in enumerate(dense_calls):
        if call.ordinal != expected_ordinal:
            raise ValueError("Dense call ordinals must be trace ordered")
        key = (int(call.input_key), tuple(call.input_shape))
        if key not in dense_groups:
            dense_groups[key] = []
            dense_group_order.append(key)
        dense_groups[key].append(call)

    for key in dense_group_order:
        calls = tuple(dense_groups[key])
        logical_object_id = _dense_lineage_id(calls)
        input_shape = tuple(calls[0].input_shape)
        rank = input_shape[-1]
        obj = _identity_object(
            logical_object_id,
            Lifetime.REQUEST_IMMUTABLE,
            input_shape,
            rank=rank,
        )
        objects.append(obj)
        roles = tuple(call.scope_path.rsplit("/", 1)[-1] for call in calls)
        is_qkv = roles == ("query", "key", "value")
        group_consumers = []
        for call, role in zip(calls, roles):
            consumer = ConsumerContextSpec.dense(
                consumer_id=call.scope_path,
                ordinal=len(consumers),
                logical_object_id=logical_object_id,
                native_view=PublicLinearMap.identity(input_shape, rank=rank),
                lift=PublicLinearMap.identity(input_shape, rank=rank),
                rhs_shape=tuple(call.kernel_shape),
                schedule="simultaneous" if is_qkv else "sequential",
                role=role,
                object_operand="lhs",
            )
            consumers.append(consumer)
            group_consumers.append(consumer)
        if is_qkv:
            physical_families.append(
                PhysicalFamilySpec(
                    physical_family_id=logical_object_id,
                    kind="qkv_direct_sum",
                    logical_consumer_ids=tuple(
                        consumer.consumer_id for consumer in group_consumers
                    ),
                    lhs_shape=input_shape,
                    rhs_shape=(
                        calls[0].kernel_shape[0],
                        sum(call.kernel_shape[1] for call in calls),
                    ),
                )
            )
        else:
            for call, consumer in zip(calls, group_consumers):
                physical_families.append(
                    PhysicalFamilySpec(
                        physical_family_id=f"dense.family.{call.ordinal}",
                        kind="other_dense",
                        logical_consumer_ids=(consumer.consumer_id,),
                        lhs_shape=input_shape,
                        rhs_shape=tuple(call.kernel_shape),
                    )
                )

    embedding_groups: dict[str, list[DynamicLinearCall]] = {}
    attention_calls: list[DynamicLinearCall] = []
    for call in dynamic_calls:
        if call.kind == "embedding_row":
            embedding_groups.setdefault(call.logical_object_id, []).append(call)
        elif call.kind == "attention_matmul":
            attention_calls.append(call)
        elif call.kind == "public_gather":
            continue
        else:
            raise ValueError(f"unsupported dynamic linear kind {call.kind!r}")

    for logical_object_id, row_calls in embedding_groups.items():
        row_calls = sorted(row_calls, key=lambda call: call.ordinal)
        first = row_calls[0]
        if any(
            call.rhs_shape != first.rhs_shape
            or call.lhs_shape != first.lhs_shape
            or call.object_operand != "rhs"
            for call in row_calls
        ):
            raise ValueError("embedding rows must share one table and selector shape")
        table_shape = first.rhs_shape
        obj = _identity_object(
            logical_object_id,
            Lifetime.MODEL_STATIC,
            table_shape,
            rank=math.prod(table_shape),
        )
        objects.append(obj)
        row_consumers = []
        for row_index, call in enumerate(row_calls):
            consumer = ConsumerContextSpec(
                consumer_id=f"word_embedding.row.{row_index}",
                ordinal=len(consumers),
                logical_object_id=logical_object_id,
                family="MatMulVVS",
                native_view=PublicLinearMap.identity(
                    table_shape, rank=obj.joint_view.rank
                ),
                lift=PublicLinearMap.identity(
                    table_shape, rank=obj.joint_view.rank
                ),
                rhs_shape=call.lhs_shape,
                schedule="simultaneous",
                role=call.role,
                object_operand="rhs",
            )
            consumers.append(consumer)
            row_consumers.append(consumer)
        selector_width = first.lhs_shape[-1]
        physical_families.append(
            PhysicalFamilySpec(
                physical_family_id=(
                    "request.word_embedding_batch"
                    if logical_object_id == "model.word_embeddings"
                    else f"request.embedding_batch.{_safe_id(logical_object_id)}"
                ),
                kind="embedding_batch",
                logical_consumer_ids=tuple(
                    consumer.consumer_id for consumer in row_consumers
                ),
                lhs_shape=(len(row_calls), selector_width),
                rhs_shape=table_shape,
            )
        )

    for call in sorted(attention_calls, key=lambda item: item.ordinal):
        object_shape = call.lhs_shape if call.object_operand == "lhs" else call.rhs_shape
        other_shape = call.rhs_shape if call.object_operand == "lhs" else call.lhs_shape
        rank = object_shape[-1]
        obj = _identity_object(
            call.logical_object_id,
            Lifetime.REQUEST_IMMUTABLE,
            object_shape,
            rank=rank,
        )
        objects.append(obj)
        consumer_id = dynamic_consumer_id(call)
        consumer = ConsumerContextSpec(
            consumer_id=consumer_id,
            ordinal=len(consumers),
            logical_object_id=call.logical_object_id,
            family="BatchMatMul" if call.batched else "MatMul",
            native_view=PublicLinearMap.identity(object_shape, rank=rank),
            lift=PublicLinearMap.identity(object_shape, rank=rank),
            rhs_shape=other_shape,
            schedule="sequential",
            role=call.role,
            object_operand=call.object_operand,
        )
        consumers.append(consumer)
        physical_families.append(
            PhysicalFamilySpec(
                physical_family_id=f"attention.family.{call.ordinal}",
                kind="attention_matmul",
                logical_consumer_ids=(consumer_id,),
                lhs_shape=call.lhs_shape,
                rhs_shape=call.rhs_shape,
            )
        )

    plan = LinearObjectPlan(
        objects=tuple(objects),
        consumers=tuple(consumers),
        physical_families=tuple(physical_families),
        boundaries=tuple(boundaries),
    )
    validate_plan(plan)
    return plan
