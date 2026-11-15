from __future__ import annotations
import math
from typing import Any
from vit_dense_plan import DensePlanExecutor, hijack_dense_plan
STATIC_KINDS={'qkv_direct_sum','other_dense'}
DYNAMIC_KINDS={'attention_matmul'}

def make_full_evaluate(*,mode,model_apply,dense_class,dense_calls,runtime_scope,
                      intrinsic,plan_sha256,model_epoch,request_epoch,
                      group_request_ids=(),release_model=False):
    if mode!='ours-object-context':raise ValueError('this release provides only full Riblet')
    if len(group_request_ids)!=49:raise ValueError('49 fresh Dense request identifiers required')
    def evaluate(params,image,*handles):
        if len(handles)!=49:raise ValueError('49 Dense handles required')
        indices={id(h):i for i,h in enumerate(handles)}
        def projection(inputs,handle,*,output_widths):
            index=indices[id(handle)]
            return intrinsic.mjv_linear_group_online(inputs,handle,model_id=model_epoch+index,
                request_id=group_request_ids[index],output_widths=tuple(output_widths))
        executor=DensePlanExecutor(dense_calls,handles=handles,grouped_projection=projection)
        image=intrinsic.mjv_epoch_begin(image,plan_sha256=plan_sha256,model_epoch=model_epoch,request_epoch=request_epoch)
        image=intrinsic.mjv_linear_begin(image,request_id=request_epoch,manifest_hash=plan_sha256,
            backend='ours',common_fallback_prefix_calls=1,allowed_online_dot_ole_calls=1,allowed_online_batch_dot_ole_calls=0)
        with runtime_scope(),hijack_dense_plan(dense_class,executor):logits=model_apply(params,image)
        logits=intrinsic.mjv_linear_end(logits,request_id=request_epoch)
        audit=intrinsic.mjv_audit_snapshot(logits,plan_sha256=plan_sha256,model_epoch=model_epoch,request_epoch=request_epoch)
        logits=intrinsic.mjv_epoch_end(logits,audit,plan_sha256=plan_sha256,model_epoch=model_epoch,
                                      request_epoch=request_epoch,release_model_epoch=release_model)
        return logits,audit
    return evaluate
def select_runtime_families(schedule: dict[str, Any]):
    families = tuple(schedule.get("physical_families", ()))
    contexts = {
        context["physical_family_id"]: context
        for context in schedule.get("object_contexts", ())
    }

    def bind(family):
        try:
            context = contexts[family["physical_family_id"]]
        except KeyError as error:
            raise ValueError("runtime family lacks its object context") from error
        return {**family, "runtime_context": context}

    static = tuple(
        bind(family) for family in families if family.get("kind") in STATIC_KINDS
    )
    dynamic = tuple(
        bind(family) for family in families if family.get("kind") in DYNAMIC_KINDS
    )
    if len(static) != 49 or len(dynamic) != 24:
        raise ValueError("ViT runtime requires 49 static and 24 dynamic families")
    return static, dynamic

def _runtime_metadata(family: dict[str, Any]) -> dict[str, Any]:
    consumers = tuple(family.get("logical_consumer_ids", ()))
    lhs_shape = tuple(int(value) for value in family.get("lhs_shape", ()))
    rhs_shape = tuple(int(value) for value in family.get("rhs_shape", ()))
    if not consumers or not lhs_shape or not rhs_shape:
        raise ValueError("runtime family lacks readable consumers or shapes")
    context = family.get("runtime_context")
    if not isinstance(context, dict):
        raise ValueError("runtime family has no readable object context")
    codomain = tuple(
        int(value) for value in context.get("joint_view", {}).get("codomain_shape", ())
    )
    if codomain == lhs_shape:
        operand = "lhs"
        shared_shape, right_shape = lhs_shape, rhs_shape
    elif codomain == rhs_shape:
        operand = "rhs"
        shared_shape, right_shape = rhs_shape, lhs_shape
    else:
        raise ValueError("object-context joint view does not bind a family operand")
    if tuple(context.get("logical_consumer_ids", ())) != consumers:
        raise ValueError("object-context consumers differ from the physical family")
    return {
        "physical_family_id": family["physical_family_id"],
        "logical_object_id": context["logical_object_id"],
        "logical_members": len(consumers),
        "object_lifetime": context["lifetime"],
        "object_operand": operand,
        "family_kind": family["kind"],
        "shared_shape": shared_shape,
        "right_shape": right_shape,
    }

def _require_dense_alignment(groups, static_families) -> None:
    if len(groups) != 49 or len(static_families) != 49:
        raise ValueError("Dense groups and static families must both contain 49 entries")
    for group, family in zip(groups, static_families):
        rhs_shape = tuple(int(value) for value in family["rhs_shape"])
        lhs_shape = tuple(int(value) for value in family["lhs_shape"])
        if sum(group.output_widths) != rhs_shape[-1]:
            raise ValueError("Dense output width differs from its family")
        if group.input_shape[-1] != rhs_shape[0] or (
            math.prod(group.input_shape[:-1]) != math.prod(lhs_shape[:-1])
        ):
            raise ValueError("Dense input shape differs from its family")
        if len(group.scope_paths) != len(family["logical_consumer_ids"]):
            raise ValueError("Dense group consumer count differs from its family")

def build_ours_setup_specs(
    groups,
    static_families,
    *,
    plan_sha256: str,
    model_epoch: int,
) -> tuple[dict[str, Any], ...]:
    groups = tuple(groups)
    static_families = tuple(static_families)
    _require_dense_alignment(groups, static_families)
    return tuple(
        {
            "model_id": int(model_epoch) + index,
            "output_widths": tuple(group.output_widths),
            "sum_r": int(group.input_shape[-1]) * len(group.scope_paths),
            "linear_family": "matmul_aa",
            "plan_sha256": plan_sha256,
            "model_epoch": int(model_epoch),
            **_runtime_metadata(family),
        }
        for index, (group, family) in enumerate(zip(groups, static_families))
    )

def build_ours_preprocess_specs(
    groups,
    static_families,
    dynamic_families,
    *,
    plan_sha256: str,
    model_epoch: int,
    request_epoch: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    groups = tuple(groups)
    static_families = tuple(static_families)
    dynamic_families = tuple(dynamic_families)
    _require_dense_alignment(groups, static_families)
    if len(dynamic_families) != 24:
        raise ValueError("ViT requires 24 dynamic attention families")
    request_epoch = int(request_epoch)
    static = tuple(
        {
            "model_id": int(model_epoch) + index,
            "request_id": request_epoch * 100 + index,
            "rows": math.prod(int(value) for value in group.input_shape[:-1]),
            "plan_sha256": plan_sha256,
            "model_epoch": int(model_epoch),
            "request_epoch": request_epoch,
            "physical_family_id": family["physical_family_id"],
        }
        for index, (group, family) in enumerate(zip(groups, static_families))
    )
    dynamic = tuple(
        {
            "plan_sha256": plan_sha256,
            "model_epoch": int(model_epoch),
            "request_epoch": request_epoch,
            "consumer_id": family["logical_consumer_ids"][0],
            "lhs_shape": tuple(int(value) for value in family["lhs_shape"]),
            "rhs_shape": tuple(int(value) for value in family["rhs_shape"]),
            **{
                key: value
                for key, value in _runtime_metadata(family).items()
                if key not in {"shared_shape", "right_shape"}
            },
        }
        for family in dynamic_families
    )
    return static, dynamic

def make_ours_setup(*, intrinsic, specs):
    specs = tuple(specs)

    def setup(*combined_weights):
        if len(combined_weights) != len(specs):
            raise ValueError("Ours setup weights and specs differ")
        weights = list(combined_weights)
        weights[0] = intrinsic.mjv_model_setup_batch(
            weights[0],
            plan_sha256=specs[0]["plan_sha256"],
            model_epoch=specs[0]["model_epoch"],
        )
        return tuple(
            intrinsic.mjv_model_setup(weight, **spec)
            for weight, spec in zip(weights, specs)
        )

    return setup

def make_ours_preprocess(*, intrinsic, static_specs, dynamic_specs):
    static_specs = tuple(static_specs)
    dynamic_specs = tuple(dynamic_specs)

    def preprocess(*model_handles):
        if len(model_handles) != len(static_specs):
            raise ValueError("Ours preprocess requires all model handles")
        handles = list(model_handles)
        handles[0] = intrinsic.mjv_request_preprocess_batch(
            handles[0],
            plan_sha256=static_specs[0]["plan_sha256"],
            model_epoch=static_specs[0]["model_epoch"],
            request_epoch=static_specs[0]["request_epoch"],
        )
        request_handles = tuple(
            intrinsic.mjv_request_preprocess(handle, **spec)
            for handle, spec in zip(handles, static_specs)
        )
        for spec in dynamic_specs:
            intrinsic.mjv_consumer_preprocess(**spec)
        return request_handles

    return preprocess
