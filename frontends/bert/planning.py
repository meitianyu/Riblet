import hashlib, json, math
from contextlib import contextmanager
from linear_execution_plan import select_complete_bert_dynamic_consumers
from linear_object_plan import hash_plan


def extract_word_embedding_kernel(params):
    return params["bert"]["embeddings"]["word_embeddings"]["embedding"]


def select_complete_bert_plan_dynamic_consumers(
    dense_calls,
    dynamic_calls,
    *,
    input_ids,
    model_params,
):
    embedding_kernel = extract_word_embedding_kernel(model_params)

    dynamic_calls = tuple(
        call
        for call in dynamic_calls
        if not (
            call.source_primitive == "gather"
            and call.kind == "embedding_row"
            and call.lhs_shape == (1, 2)
            and call.rhs_shape == (2, 768)
        )
    )
    selected = select_complete_bert_dynamic_consumers(
        dense_calls,
        dynamic_calls,
        embedding_row_count=math.prod(tuple(input_ids.shape)),
        embedding_table_shape=tuple(embedding_kernel.shape),
    )
    return selected, embedding_kernel


@contextmanager
def hijack_word_embedding_batch(
    embed_class,
    *,
    one_hot,
    projection,
    target_name="word_embeddings",
):
    original_call = embed_class.__call__

    def batched_call(module, inputs):
        if module.name != target_name:
            return original_call(module, inputs)
        embedding = module.variables["params"]["embedding"]

        selectors = one_hot(inputs, embedding.shape[0]).astype(embedding.dtype)
        flat_selectors = selectors.reshape((-1, embedding.shape[0]))
        rows = projection(flat_selectors, embedding)
        return rows.reshape(selectors.shape[:-1] + (embedding.shape[1],))

    embed_class.__call__ = batched_call
    try:
        yield
    finally:
        embed_class.__call__ = original_call


def runtime_family_metadata(manifest, physical_family_id):
    families = tuple(
        family
        for family in manifest.get("physical_families", ())
        if family.get("physical_family_id") == physical_family_id
    )
    if len(families) != 1:
        raise ValueError(f"runtime family must resolve exactly once: {physical_family_id}")
    family = families[0]

    consumers_by_id = {
        consumer["consumer_id"]: consumer for consumer in manifest.get("consumer_contexts", ())
    }
    member_ids = tuple(family.get("logical_consumer_ids", ()))
    try:
        members = tuple(consumers_by_id[consumer_id] for consumer_id in member_ids)
    except KeyError as exc:
        raise ValueError("runtime family references an unknown consumer") from exc
    if not members:
        raise ValueError("runtime family must have logical consumers")

    object_ids = {member.get("logical_object_id") for member in members}
    operands = {member.get("object_operand") for member in members}
    if len(object_ids) != 1 or None in object_ids:
        raise ValueError("runtime family must share one logical object")
    if len(operands) != 1 or operands - {"lhs", "rhs"}:
        raise ValueError("runtime family must share one object orientation")
    logical_object_id = next(iter(object_ids))
    object_operand = next(iter(operands))

    objects = tuple(
        obj
        for obj in manifest.get("object_contexts", ())
        if obj.get("logical_object_id") == logical_object_id
    )
    if len(objects) != 1:
        raise ValueError("runtime family object must resolve exactly once")
    lifetime = objects[0].get("lifetime")
    if lifetime not in ("model-static", "request-immutable"):
        raise ValueError(f"unsupported runtime object lifetime: {lifetime}")

    lhs_shape = tuple(family.get("lhs_shape", ()))
    rhs_shape = tuple(family.get("rhs_shape", ()))
    if not lhs_shape or not rhs_shape:
        raise ValueError("runtime family physical shapes are required")
    shared_shape, right_shape = (
        (lhs_shape, rhs_shape) if object_operand == "lhs" else (rhs_shape, lhs_shape)
    )
    declared_shape = tuple(objects[0].get("object_shape", ()))
    if declared_shape and declared_shape != shared_shape:
        raise ValueError("runtime family shape differs from its logical object")

    return {
        "physical_family_id": physical_family_id,
        "logical_object_id": logical_object_id,
        "logical_members": len(members),
        "object_lifetime": lifetime,
        "object_operand": object_operand,
        "family_kind": family.get("kind"),
        "shared_shape": shared_shape,
        "right_shape": right_shape,
    }


def bind_model_runtime_family(
    spec,
    manifest,
    physical_family_id,
    *,
    plan_sha256,
    model_epoch,
):
    return {
        **dict(spec),
        "plan_sha256": plan_sha256,
        "model_epoch": model_epoch,
        **runtime_family_metadata(manifest, physical_family_id),
    }


def bind_request_runtime_family(
    spec,
    physical_family_id,
    *,
    plan_sha256,
    model_epoch,
    request_epoch,
):
    if not physical_family_id:
        raise ValueError("request runtime family ID must not be empty")
    return {
        **dict(spec),
        "plan_sha256": plan_sha256,
        "model_epoch": model_epoch,
        "request_epoch": request_epoch,
        "physical_family_id": physical_family_id,
    }


def bind_dynamic_runtime_family(
    spec,
    manifest,
    physical_family_id,
    *,
    model_epoch,
):
    return {
        **dict(spec),
        "model_epoch": model_epoch,
        **runtime_family_metadata(manifest, physical_family_id),
    }


def _runtime_families(manifest, kinds):
    return tuple(
        family for family in manifest.get("physical_families", ()) if family.get("kind") in kinds
    )


def bind_model_runtime_schedule(
    specs,
    manifest,
    *,
    plan_sha256,
    model_epoch,
):
    specs = tuple(specs)
    families = _runtime_families(manifest, {"qkv_direct_sum", "other_dense", "embedding_batch"})
    if len(specs) != len(families):
        raise ValueError("model setup specs do not match the physical schedule")
    return tuple(
        bind_model_runtime_family(
            spec,
            manifest,
            family["physical_family_id"],
            plan_sha256=plan_sha256,
            model_epoch=model_epoch,
        )
        for spec, family in zip(specs, families)
    )


def bind_request_runtime_schedule(
    static_specs,
    dynamic_specs,
    manifest,
    *,
    plan_sha256,
    model_epoch,
    request_epoch,
):
    static_specs = tuple(static_specs)
    dynamic_specs = tuple(dynamic_specs)
    static_families = _runtime_families(
        manifest, {"qkv_direct_sum", "other_dense", "embedding_batch"}
    )
    if len(static_specs) != len(static_families):
        raise ValueError("static request specs do not match the physical schedule")
    bound_static = tuple(
        bind_request_runtime_family(
            spec,
            family["physical_family_id"],
            plan_sha256=plan_sha256,
            model_epoch=model_epoch,
            request_epoch=request_epoch,
        )
        for spec, family in zip(static_specs, static_families)
    )

    dynamic_families = _runtime_families(manifest, {"attention_matmul"})
    family_by_consumer = {}
    for family in dynamic_families:
        for consumer_id in family.get("logical_consumer_ids", ()):
            if consumer_id in family_by_consumer:
                raise ValueError("dynamic consumer appears in multiple runtime families")
            family_by_consumer[consumer_id] = family
    dynamic_ids = tuple(spec.get("consumer_id") for spec in dynamic_specs)
    if set(dynamic_ids) != set(family_by_consumer) or len(dynamic_ids) != len(family_by_consumer):
        raise ValueError("dynamic request specs do not match the physical schedule")
    bound_dynamic = tuple(
        bind_dynamic_runtime_family(
            spec,
            manifest,
            family_by_consumer[spec["consumer_id"]]["physical_family_id"],
            model_epoch=model_epoch,
        )
        for spec in dynamic_specs
    )
    return bound_static, bound_dynamic


def build_linear_path_manifest(dense_calls, dynamic_calls, object_plan=None):
    manifest = {
        "schema_version": 1,
        "static_dense_calls": [
            {
                "ordinal": call.ordinal,
                "scope_path": call.scope_path,
                "input_shape": list(call.input_shape),
                "kernel_shape": list(call.kernel_shape),
                "baseline_family": "source-derived-at-online-scope",
            }
            for call in dense_calls
        ],
        "source_dynamic_consumers": [
            {
                "ordinal": call["ordinal"],
                "consumer_id": call["consumer_id"],
                "logical_object_id": call["logical_object_id"],
                "source_primitive": call["source_primitive"],
                "source_ordinal": call["source_ordinal"],
                "lhs_shape": list(call["lhs_shape"]),
                "rhs_shape": list(call["rhs_shape"]),
                "batched": bool(call["batched"]),
            }
            for call in dynamic_calls
        ],
    }
    if object_plan is not None:
        manifest.update(object_plan.to_dict())
        manifest["logical_consumer_count"] = object_plan.logical_consumer_count
        manifest["physical_family_count"] = object_plan.physical_family_count
        manifest["physical_family_counts"] = object_plan.physical_family_counts
        manifest["object_context_plan_sha256"] = hash_plan(object_plan)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return manifest, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def derive_object_context_opening_expectations(manifest):
    objects = manifest.get("object_contexts", ())
    consumers = manifest.get("consumer_contexts", ())
    families = manifest.get("physical_families", ())

    objects_by_id = {}
    for obj in objects:
        logical_object_id = obj.get("logical_object_id")
        if not logical_object_id or logical_object_id in objects_by_id:
            raise ValueError("logical object IDs must be non-empty and unique")
        lifetime = obj.get("lifetime")
        if lifetime not in ("model-static", "request-immutable"):
            raise ValueError(f"unsupported object lifetime: {lifetime}")
        objects_by_id[logical_object_id] = obj

    consumers_by_id = {}
    for consumer in consumers:
        consumer_id = consumer.get("consumer_id")
        if not consumer_id or consumer_id in consumers_by_id:
            raise ValueError("consumer IDs must be non-empty and unique")
        logical_object_id = consumer.get("logical_object_id")
        if logical_object_id not in objects_by_id:
            raise ValueError(f"consumer references unknown object: {logical_object_id}")
        consumers_by_id[consumer_id] = consumer

    family_members = {}
    for family in families:
        family_id = family.get("physical_family_id")
        if not family_id or family_id in family_members:
            raise ValueError("physical family IDs must be non-empty and unique")
        members = tuple(family.get("logical_consumer_ids", ()))
        if not members:
            raise ValueError("physical family must contain logical consumers")
        for consumer_id in members:
            if consumer_id not in consumers_by_id:
                raise ValueError(f"physical family references unknown consumer: {consumer_id}")
            if any(consumer_id in existing for existing in family_members.values()):
                raise ValueError(f"consumer appears in multiple families: {consumer_id}")
        family_members[family_id] = members

    scheduled_consumers = {
        consumer_id for members in family_members.values() for consumer_id in members
    }
    if scheduled_consumers != set(consumers_by_id):
        raise ValueError("physical schedule must cover every logical consumer exactly once")

    referenced_object_ids = {consumer["logical_object_id"] for consumer in consumers_by_id.values()}
    model_objects = {
        object_id
        for object_id in referenced_object_ids
        if objects_by_id[object_id]["lifetime"] == "model-static"
    }
    request_objects = referenced_object_ids - model_objects

    model_family_kinds = {"qkv_direct_sum", "other_dense"}
    request_family_kinds = {"embedding_batch", "attention_matmul"}
    unknown_family_kinds = (
        {family.get("kind") for family in families} - model_family_kinds - request_family_kinds
    )
    if unknown_family_kinds:
        raise ValueError(f"unsupported physical family kinds: {sorted(unknown_family_kinds)}")

    def logical_members_for(kinds):
        return sum(
            len(family_members[family["physical_family_id"]])
            for family in families
            if family["kind"] in kinds
        )

    def family_count_for(kinds):
        return sum(1 for family in families if family["kind"] in kinds)

    return {
        "model_required_shared_object_contexts": len(model_objects),
        "request_required_shared_object_contexts": len(request_objects),
        "model_setup_right_logical_openings": logical_members_for(model_family_kinds),
        "model_setup_right_physical_collectives": family_count_for(model_family_kinds),
        "request_online_right_logical_openings": logical_members_for(request_family_kinds),
        "request_online_right_physical_collectives": family_count_for(request_family_kinds),
        "logical_consumer_count": len(consumers_by_id),
        "physical_family_count": len(families),
    }


def _hash_manifest_section(manifest, fields):
    payload = {field: manifest.get(field, ()) for field in fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compile_execution_certificate(manifest):
    expectations = derive_object_context_opening_expectations(manifest)
    family_counts = {}
    for family in manifest.get("physical_families", ()):
        kind = family["kind"]
        family_counts[kind] = family_counts.get(kind, 0) + 1
    physical_family_count = expectations["physical_family_count"]
    certificate = {
        "linear_mode": "ours-object-context",
        "logical_consumer_manifest_sha256": _hash_manifest_section(
            manifest, ("object_contexts", "consumer_contexts")
        ),
        "physical_fusion_schedule_sha256": _hash_manifest_section(manifest, ("physical_families",)),
        "logical_consumer_count": expectations["logical_consumer_count"],
        "physical_family_count": physical_family_count,
        "physical_family_counts": family_counts,
        "secure_invoke_rpc_calls": 1,
        "protocol_matrix_calls": physical_family_count,
    }
    certificate.update(online_dot_ole_calls=0, online_batch_dot_ole_calls=0)
    return certificate
