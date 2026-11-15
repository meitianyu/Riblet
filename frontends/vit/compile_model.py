from pathlib import Path
import hashlib, json, math
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import flax.linen as nn
from flax.serialization import msgpack_restore
from transformers import ViTConfig, FlaxViTForImageClassification
from spu import spu_pb2 as pb
import spu.intrinsic as intrinsic
from spu.utils import frontend
from riblet.assets import verify_model
from riblet.packages import load_model_package, graph_binding
from riblet.vit_parameters import point_shape
from vit_dense_plan import capture_dense_trace, group_dense_calls, combine_group_kernels
from vit_full_secure_entry import (
    select_runtime_families,
    build_ours_setup_specs,
    build_ours_preprocess_specs,
    make_ours_setup,
    make_ours_preprocess,
)
from vit_native_policy import plan_hash, fingerprint_text, load_schedule
from ours_frontend import make_forward


from riblet.config import runtime_config, public_audit


def get_kernel(params, path):
    value = params
    for part in path.split("/"):
        value = value[part]
    return value["kernel"]


def certify(code):
    expected = {
        "mjv_group_online:": 49,
        "ours_vit_inverse_v1:reciprocal": 12,
        "ours_vit_inverse_v1:rsqrt": 25,
        "spu.gelu": 12,
        "spu.neg_exp": 12,
    }
    for tag, n in expected.items():
        if code.count(tag) != n:
            raise ValueError("complete ViT graph call mismatch: " + tag)
    if any(x in code.lower() for x in ("nimbus", "ops_select", "fair_fused", "dbg_print", "erf")):
        raise ValueError("foreign or diagnostic graph path")
    return dict(calls=expected, code_sha256=hashlib.sha256(code.encode()).hexdigest())


def build(
    model_path, parameter_package, output, *, model_epoch, request_epoch, preprocessor_path=None
):
    model_path = Path(model_path)
    out = Path(output)
    if out.exists():
        raise FileExistsError("compiled bundle is immutable")
    package = load_model_package(parameter_package, rank=0)
    verify_model(
        "vit-base-patch16-224",
        model_path,
        bound=package["model_files"],
        preprocessor=preprocessor_path,
    )
    if package["model"] != "vit-base-patch16-224":
        raise ValueError("wrong model parameters")
    for v in (model_epoch, request_epoch):
        if type(v) is not int or not 0 < v < 2**52:
            raise ValueError("bounded public epoch required")
    config = ViTConfig.from_pretrained(str(model_path), local_files_only=True)
    model = FlaxViTForImageClassification(config, _do_init=False)
    params = msgpack_restore((model_path / "flax_model.msgpack").read_bytes())
    image = jax.ShapeDtypeStruct((1, 3, 224, 224), np.float32)
    trace = capture_dense_trace(
        nn.Dense,
        lambda: jax.eval_shape(lambda p, x: model(x, params=p, train=False).logits, params, image),
    )
    groups = group_dense_calls(trace.calls)
    weights = combine_group_kernels(
        trace.calls, [get_kernel(params, c.scope_path) for c in trace.calls]
    )
    schedule = load_schedule()
    static, dynamic = select_runtime_families(schedule)
    ph = plan_hash()
    setup_specs = build_ours_setup_specs(groups, static, plan_sha256=ph, model_epoch=model_epoch)
    stat, dyn = build_ours_preprocess_specs(
        groups,
        static,
        dynamic,
        plan_sha256=ph,
        model_epoch=model_epoch,
        request_epoch=request_epoch,
    )
    out.mkdir(parents=True)
    opts = pb.CompilerOptions()
    opts.enable_optimize_denominator_with_broadcast = True

    def save(name, fn, args, inputs, outputs):
        leaves, _ = jax.tree_util.tree_flatten(args)
        exe, _ = frontend.compile(
            frontend.Kind.JAX,
            fn,
            args,
            {},
            inputs,
            [pb.VIS_SECRET] * len(leaves),
            lambda _: outputs,
            copts=opts,
        )
        if name == "request":
            exe.code = public_audit(exe.code.decode()).encode()
        (out / (name + ".exec")).write_bytes(exe.SerializeToString())
        (out / (name + ".pphlo")).write_bytes(exe.code)
        return exe.code.decode()

    ws = tuple(jax.ShapeDtypeStruct(w.shape, np.float32) for w in weights)
    wn = [f"w{i}" for i in range(49)]
    mh = [f"mh{i}" for i in range(49)]
    rh = [f"rh{i}" for i in range(49)]
    original_setup = make_ours_setup(intrinsic=intrinsic, specs=setup_specs)

    def setup(*values):
        handles = original_setup(*values)
        return (
            intrinsic.vit_linear_manifest(handles[0], plan_sha256=ph, records=fingerprint_text()),
        ) + tuple(handles[1:])

    save("setup", setup, ws, wn, mh)
    save(
        "preprocess",
        make_ours_preprocess(intrinsic=intrinsic, static_specs=stat, dynamic_specs=dyn),
        ws,
        mh,
        rh,
    )
    leaves, _ = jax.tree_util.tree_flatten(params)
    pn = [f"p{i}" for i in range(len(leaves))]
    handles = tuple(
        jax.ShapeDtypeStruct((math.prod(g.input_shape[:-1]), g.input_shape[-1]), np.float32)
        for g in groups
    )
    coefficients = tuple(jax.ShapeDtypeStruct(point_shape(i) + (2,), np.float32) for i in range(37))
    tn = [f"theta{i}" for i in range(37)]
    schedules = [
        (
            np.asarray(row["route"], np.int64).reshape(point_shape(i)),
            np.asarray(row["K"], np.int64).reshape(point_shape(i)),
        )
        for i, row in enumerate(package["schedule"])
    ]
    evaluate = make_forward(
        model,
        trace.calls,
        [s["request_id"] for s in stat],
        schedules,
        plan_sha256=ph,
        model_epoch=model_epoch,
        request_epoch=request_epoch,
        diagnostics=False,
    )
    code = save(
        "request",
        evaluate,
        (params, image, coefficients, *handles),
        pn + ["image"] + tn + rh,
        ["logits", "audit"],
    )
    (out / "config.pb").write_bytes(runtime_config().SerializeToString())
    document = dict(
        method="riblet-full",
        family="vit",
        model=package["model"],
        model_rank=0,
        data_rank=1,
        layout="vit-standard",
        **graph_binding(package),
        model_epoch=model_epoch,
        request_epoch=request_epoch,
        plan_sha256=ph,
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
        coefficients=tn,
        certificate=certify(code),
        graphs={
            n: hashlib.sha256((out / (n + ".exec")).read_bytes()).hexdigest()
            for n in ("setup", "preprocess", "request")
        },
        plaintext_forward_executed=False,
        model_initialization_forward_executed=False,
    )
    (out / "GRAPH.json").write_text(json.dumps(document, indent=2))
    return document
