from contextlib import contextmanager
import jax.numpy as jnp, jax.nn as jnn
import flax.linen as fnn
import flax.linen.normalization as fnormalization
import spu.intrinsic as intrinsic
import cdi_bert_integration
from linear_execution_plan import DensePlanExecutor, hijack_dense_plan
from planning import hijack_word_embedding_batch


def make_mjv_all_evaluate(
    ctx,
    group_request_ids,
    scope_request_id,
    manifest_hash,
    backend,
    fair_schedule=False,
    object_context_scope=False,
    model_epoch=None,
    release_model_epoch=False,
):
    args = ctx.args
    model = ctx.model
    dense_groups = ctx.groups
    dense_trace = ctx.trace
    model_ids = ctx.model_ids
    static_group_count = len(dense_groups) + 1
    effective_length = ctx.effective_length
    physical_length = 128
    cdi_execution_plan = ctx.cdi_plan
    if backend != "ours":
        raise ValueError("only the full Riblet linear implementation is available")

    @contextmanager
    def hijack(runtime=None):
        with cdi_bert_integration.native_nonlinear_shells(
            fnn, fnormalization, jnn, jnp, intrinsic, runtime
        ):
            yield

    def evaluate(
        params,
        request_input_ids,
        request_attention_mask,
        request_token_type_ids,
        *request_handles,
    ):
        timed = args.cdi_nonlinear_mode != "disabled"
        if timed:
            if not hasattr(intrinsic, "cdi_scope"):
                raise ValueError("compiled CDI scope ABI missing")
            anchored = intrinsic.cdi_scope(
                (
                    request_input_ids,
                    request_attention_mask,
                    request_token_type_ids,
                    *request_handles,
                ),
                scope="inference",
                site=scope_request_id,
                action="begin",
            )
            request_input_ids, request_attention_mask, request_token_type_ids = anchored[:3]
            request_handles = anchored[3:]
        linear_handles = request_handles
        cdi_context = None
        if object_context_scope:
            from riblet.bert_tables import runtime_arguments

            if len(request_handles) != static_group_count + 2:
                raise ValueError("expected linear handles and two secret coefficient components")
            linear_handles = request_handles[:static_group_count]
            cdi_context = cdi_execution_plan.bind_runtime_arguments(
                jnp,
                intrinsic,
                *runtime_arguments(jnp, ctx.schedule, *request_handles[-2:], effective_length),
            )
        if timed:
            if cdi_context is None:
                cdi_context = cdi_bert_integration.CdiBertRuntimeContext(
                    jnp=jnp, intrinsic=intrinsic
                )
            cdi_context.enable_execution(
                native_rsqrt=fnormalization.lax.rsqrt,
                effective_length=effective_length,
                physical_length=physical_length,
            )
        native_segments = cdi_bert_integration.native_segment_prefix(
            params, request_token_type_ids, jnp=jnp
        )
        scope_anchor = native_segments
        if object_context_scope:
            scope_anchor = intrinsic.mjv_epoch_begin(
                scope_anchor,
                plan_sha256=manifest_hash,
                model_epoch=model_epoch,
                request_epoch=scope_request_id,
            )
        scope_anchor = intrinsic.mjv_linear_begin(
            scope_anchor,
            request_id=scope_request_id,
            manifest_hash=manifest_hash,
            backend=backend,
        )
        request_input_ids, native_segments = intrinsic.cdi_scope(
            (request_input_ids, scope_anchor),
            scope="barrier",
            site=scope_request_id,
            action="barrier",
        )
        with hijack(cdi_context), cdi_bert_integration.use_native_segment_prefix(
            fnn.Embed, native_segments
        ):
            if fair_schedule:
                dense_handles = linear_handles[: len(dense_groups)]
                handle_indices = {id(handle): index for index, handle in enumerate(dense_handles)}

                def grouped_projection(inputs, handle, *, output_widths):
                    group = handle_indices[id(handle)]
                    return intrinsic.mjv_linear_group_online(
                        inputs,
                        handle,
                        model_id=model_ids[group],
                        request_id=group_request_ids[group],
                        output_widths=output_widths,
                    )

                executor = DensePlanExecutor(
                    dense_trace.calls,
                    handles=dense_handles,
                    grouped_projection=grouped_projection,
                )

                def embedding_projection(selectors, embedding):
                    return intrinsic.mjv_linear_group_online(
                        selectors,
                        linear_handles[len(dense_groups)],
                        model_id=model_ids[len(dense_groups)],
                        request_id=group_request_ids[len(dense_groups)],
                        output_widths=(embedding.shape[1],),
                        logical_row_members=selectors.shape[0],
                    )[0]

                with hijack_dense_plan(fnn.Dense, executor), hijack_word_embedding_batch(
                    fnn.Embed,
                    one_hot=jnn.one_hot,
                    projection=embedding_projection,
                ):
                    logits = model(
                        request_input_ids,
                        request_attention_mask,
                        token_type_ids=request_token_type_ids,
                        params=params,
                    )[0]
            else:
                handle_indices = {id(handle): index for index, handle in enumerate(linear_handles)}

                def grouped_projection(inputs, handle, *, output_widths):
                    group = handle_indices[id(handle)]
                    return intrinsic.mjv_linear_group_online(
                        inputs,
                        handle,
                        model_id=model_ids[group],
                        request_id=group_request_ids[group],
                        output_widths=output_widths,
                    )

                executor = DensePlanExecutor(
                    dense_trace.calls,
                    handles=linear_handles,
                    grouped_projection=grouped_projection,
                )
                with hijack_dense_plan(fnn.Dense, executor):
                    logits = model(
                        request_input_ids,
                        request_attention_mask,
                        token_type_ids=request_token_type_ids,
                        params=params,
                    )[0]

        if cdi_context is not None:
            cdi_context.assert_complete()
        logits = intrinsic.mjv_linear_end(logits, request_id=scope_request_id)
        if timed:
            logits = intrinsic.cdi_scope(
                logits, scope="inference", site=scope_request_id, action="end"
            )
        if not object_context_scope:
            return logits
        metadata = {
            "plan_sha256": manifest_hash,
            "model_epoch": model_epoch,
            "request_epoch": scope_request_id,
        }
        audit = intrinsic.mjv_audit_snapshot(logits, **metadata)
        expected_width = len(intrinsic.RUNTIME_OBJECT_AUDIT_V1_FIELDS)
        if len(audit) != expected_width:
            raise ValueError(f"runtime audit width mismatch: {len(audit)} != {expected_width}")
        logits = intrinsic.mjv_epoch_end(
            logits,
            audit,
            release_model_epoch=release_model_epoch,
            **metadata,
        )
        return logits, audit

    return evaluate
