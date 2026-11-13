from pathlib import Path
from types import SimpleNamespace
import hashlib, json, math
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

if not hasattr(jnp, "DeviceArray"):
    jnp.DeviceArray = jax.Array
import flax.linen as nn
from flax.serialization import msgpack_restore
from transformers import BertConfig, FlaxBertForSequenceClassification
from spu import spu_pb2 as pb
from spu.utils import frontend
import spu.intrinsic as intrinsic
from riblet.assets import verify_model
from riblet.packages import load_model_package, digest, graph_binding

from linear_execution_plan import (
    capture_dense_trace,
    group_dense_calls,
    combine_group_kernels,
    collect_dynamic_linear_calls,
)
from linear_object_plan import compile_linear_object_plan, dynamic_consumer_id
from forward import make_mjv_all_evaluate
from planning import (
    select_complete_bert_plan_dynamic_consumers,
    build_linear_path_manifest,
    bind_model_runtime_schedule,
    bind_request_runtime_schedule,
    compile_execution_certificate,
)
from native_admission import fingerprint_text, geometry_sha
from cdi_bert_integration import CdiBertExecutionPlan


from riblet.config import runtime_config, public_audit


def get_kernel(params, path):
    value = params
    for part in path.split("/"):
        value = value[part]
    return value["kernel"]


def build(
    model_path,
    parameter_package,
    output,
    *,
    effective_length,
    model_epoch,
    request_epoch,
    preprocessor_path=None,
):
    if type(effective_length) is not int or not 1 <= effective_length <= 128:
        raise ValueError("public effective length outside 1..128")
    for value in (model_epoch, request_epoch):
        if type(value) is not int or not 0 < value < 2**52:
            raise ValueError("bounded fresh public epoch required")
    model_path = Path(model_path)
    out = Path(output)
    if out.exists():
        raise FileExistsError("compiled bundle is immutable")
    package = load_model_package(parameter_package, rank=0)
    if not package["model"].startswith("bert-"):
        raise ValueError("BERT parameter package required")
    verify_model(
        package["model"], model_path, bound=package["model_files"], preprocessor=preprocessor_path
    )
    model = FlaxBertForSequenceClassification(
        BertConfig.from_pretrained(str(model_path), local_files_only=True), _do_init=False
    )
    params = msgpack_restore((model_path / "flax_model.msgpack").read_bytes())
    inputs = tuple(jax.ShapeDtypeStruct((1, 128), np.int64) for _ in range(3))
    trace = capture_dense_trace(
        nn.Dense,
        lambda: jax.eval_shape(
            lambda p, i, a, t: model(i, a, token_type_ids=t, params=p, train=False)[0],
            params,
            *inputs,
        ),
    )
    groups = group_dense_calls(trace.calls)
    if len(trace.calls) != 74 or len(groups) != 50:
        raise ValueError("BERT Dense consumer organization changed")
    kernels = [get_kernel(params, call.scope_path) for call in trace.calls]
    combined = combine_group_kernels(trace.calls, kernels)
    schedule = package["schedule"]
    groups_by_op = {
        op: [r for r in schedule if r["operator"] == op] for op in ("reciprocal", "rsqrt")
    }
    cdi_plan = CdiBertExecutionPlan.from_public_groups(
        "joint",
        {
            op: (tuple(r["site"] for r in rows), np.asarray([r["K"] or 0 for r in rows], np.int64))
            for op, rows in groups_by_op.items()
        },
    )
    model_ids = tuple(model_epoch + i for i in range(51))
    request_ids = tuple(request_epoch * 100 + i for i in range(51))
    ctx = SimpleNamespace(
        model=model,
        groups=groups,
        trace=trace,
        model_ids=model_ids,
        effective_length=effective_length,
        schedule=schedule,
        cdi_plan=cdi_plan,
        args=SimpleNamespace(
            cdi_nonlinear_mode="joint",
        ),
    )
    dummy = tuple(
        jax.ShapeDtypeStruct((math.prod(g.input_shape[:-1]), g.input_shape[-1]), np.float32)
        for g in groups
    )
    trace_fn = make_mjv_all_evaluate(ctx, request_ids[:50], request_epoch, "0" * 64, "ours")
    jaxpr = jax.make_jaxpr(trace_fn)(params, *inputs, *dummy)
    dynamic = collect_dynamic_linear_calls(jaxpr)
    selected, embedding = select_complete_bert_plan_dynamic_consumers(
        trace.calls, dynamic, input_ids=inputs[0], model_params=params
    )
    dynamic_rows = [
        dict(
            ordinal=c.ordinal,
            consumer_id=dynamic_consumer_id(c),
            logical_object_id=c.logical_object_id,
            source_primitive=c.source_primitive,
            source_ordinal=c.source_ordinal,
            lhs_shape=c.lhs_shape,
            rhs_shape=c.rhs_shape,
            batched=c.batched,
        )
        for c in selected
    ]
    runtime_dynamic = [
        dict(
            consumer_id=dynamic_consumer_id(c),
            source_ordinal=c.ordinal,
            lhs_shape=c.lhs_shape,
            rhs_shape=c.rhs_shape,
            batched=c.batched,
        )
        for c in dynamic
        if c.source_primitive == "dot_general"
    ]
    objects = compile_linear_object_plan(trace.calls, selected)
    manifest, _ = build_linear_path_manifest(trace.calls, dynamic_rows, objects)
    manifest.update(model_rank=0, data_rank=1)
    ph = digest(manifest)
    specs = [
        dict(
            model_id=model_ids[i],
            output_widths=g.output_widths,
            sum_r=g.input_shape[-1] * len(g.scope_paths),
        )
        for i, g in enumerate(groups)
    ]
    specs.append(
        dict(
            model_id=model_ids[-1],
            output_widths=(embedding.shape[1],),
            sum_r=embedding.shape[0],
            linear_family="matmul_vvs",
        )
    )
    specs = bind_model_runtime_schedule(specs, manifest, plan_sha256=ph, model_epoch=model_epoch)
    prep = [
        dict(
            model_id=model_ids[i],
            request_id=request_ids[i],
            rows=math.prod(groups[i].input_shape[:-1]) if i < 50 else 128,
        )
        for i in range(51)
    ]
    prep, dyn = bind_request_runtime_schedule(
        prep,
        runtime_dynamic,
        manifest,
        plan_sha256=ph,
        model_epoch=model_epoch,
        request_epoch=request_epoch,
    )
    weights = tuple(combined) + (embedding,)
    ws = tuple(jax.ShapeDtypeStruct(w.shape, np.float32) for w in weights)
    wn = [f"w{i}" for i in range(51)]
    mh = [f"mh{i}" for i in range(51)]
    rh = [f"rh{i}" for i in range(51)]
    out.mkdir(parents=True)
    opts = pb.CompilerOptions()
    opts.enable_optimize_denominator_with_broadcast = True

    def save(name, fn, args, ins, outs, public=False):
        leaves, _ = jax.tree_util.tree_flatten(args)
        exe, _ = frontend.compile(
            frontend.Kind.JAX,
            fn,
            args,
            {},
            ins,
            [pb.VIS_PUBLIC if public else pb.VIS_SECRET] * len(leaves),
            lambda _: outs,
            copts=opts,
        )
        code = exe.code.decode()
        if name == "request":
            code = public_audit(code)
            exe.code = code.encode()
        (out / (name + ".exec")).write_bytes(exe.SerializeToString())
        (out / (name + ".pphlo")).write_text(code)
        return code

    def setup(*values):
        values = list(values)
        values[0] = intrinsic.mjv_model_setup_batch(
            values[0], plan_sha256=ph, model_epoch=model_epoch
        )
        return tuple(intrinsic.mjv_model_setup(v, **s) for v, s in zip(values, specs))

    save("setup", setup, ws, wn, mh)

    def register(anchor):
        return intrinsic.mjv_linear_manifest(
            anchor, plan_sha256=ph, records=fingerprint_text(manifest)
        )

    save(
        "register",
        register,
        (jax.ShapeDtypeStruct((1,), np.int64),),
        ["anchor"],
        ["registered"],
        public=True,
    )

    def preprocess(*handles):
        handles = list(handles)
        handles[0] = intrinsic.mjv_request_preprocess_batch(
            handles[0], plan_sha256=ph, model_epoch=model_epoch, request_epoch=request_epoch
        )
        result = tuple(intrinsic.mjv_request_preprocess(h, **s) for h, s in zip(handles, prep))
        fields = (
            "model_epoch",
            "physical_family_id",
            "logical_object_id",
            "logical_members",
            "object_lifetime",
            "object_operand",
            "family_kind",
        )
        for row in dyn:
            intrinsic.mjv_consumer_preprocess(
                plan_sha256=ph,
                request_epoch=request_epoch,
                consumer_id=row["consumer_id"],
                lhs_shape=row["lhs_shape"],
                rhs_shape=row["rhs_shape"],
                **{k: row[k] for k in fields},
            )
        return result

    save("preprocess", preprocess, ws, mh, rh)
    leaves, _ = jax.tree_util.tree_flatten(params)
    pn = [f"p{i}" for i in range(len(leaves))]
    handles = dummy + (jax.ShapeDtypeStruct((128, embedding.shape[0]), np.float32),)
    evaluate = make_mjv_all_evaluate(
        ctx,
        request_ids,
        request_epoch,
        ph,
        "ours",
        fair_schedule=True,
        object_context_scope=True,
        model_epoch=model_epoch,
        release_model_epoch=False,
    )
    code = save(
        "request",
        evaluate,
        (
            params,
            *inputs,
            *handles,
            jax.ShapeDtypeStruct((169, 2), np.float32),
            jax.ShapeDtypeStruct((169, 2), np.float32),
        ),
        pn + ["input_ids", "attention_mask", "token_type_ids"] + rh + ["coeff_hi", "coeff_lo"],
        ["logits", "audit"],
    )
    counts = {
        tag: code.count(tag)
        for tag in (
            "mjv_group_online:",
            "cdi_invert:reciprocal",
            "cdi_invert:rsqrt",
        )
    }
    if list(counts.values()) != [51, 12, 25]:
        raise ValueError("complete BERT contribution call mismatch: " + str(counts))
    if any(x in code.lower() for x in ("nimbus", "east_", "fair_fused", "dbg_print")):
        raise ValueError("foreign/diagnostic path")
    certificate = dict(
        physical_families=75,
        reciprocal_calls=12,
        rsqrt_calls=25,
        logical=compile_execution_certificate(manifest),
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
    )
    (out / "config.pb").write_bytes(runtime_config().SerializeToString())
    (out / "LINEAR_PLAN.json").write_text(json.dumps(manifest, indent=2))
    document = dict(
        method="riblet-full",
        family="bert",
        model=package["model"],
        model_rank=0,
        data_rank=1,
        layout="pad128",
        input_dtype="int64",
        effective_length=effective_length,
        **graph_binding(package),
        model_epoch=model_epoch,
        request_epoch=request_epoch,
        plan_sha256=ph,
        geometry_sha256=geometry_sha(manifest),
        runtime={
            "max_concurrency": 64,
            "field": "FM64",
            "fxp_fraction_bits": 18,
            "intra_op_parallel": False,
        },
        model_schema=[
            dict(name=n, shape=list(v.shape), dtype=str(v.dtype)) for n, v in zip(pn, leaves)
        ],
        weight_schema=[
            dict(name=n, shape=list(v.shape), dtype=str(v.dtype)) for n, v in zip(wn, weights)
        ],
        kernel_paths=[c.scope_path for c in trace.calls],
        groups=[list(g.consumer_ordinals) for g in groups],
        model_handles=mh,
        request_handles=rh,
        certificate=certificate,
        graphs={
            name: hashlib.sha256((out / (name + ".exec")).read_bytes()).hexdigest()
            for name in ("setup", "register", "preprocess", "request")
        },
        plaintext_forward_executed=False,
        model_initialization_forward_executed=False,
    )
    (out / "GRAPH.json").write_text(json.dumps(document, indent=2))
    return document
