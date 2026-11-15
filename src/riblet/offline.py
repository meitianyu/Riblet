import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .config import model_spec
from .assets import file_hash as _sha

ROOT = Path(__file__).resolve().parents[2]


def source_root(model):
    if model not in ("bert", "vit"):
        raise ValueError("only the two supported construction graphs are available")
    return ROOT / "offline" / model


def construction_sources(family):

    files = [p for p in source_root(family).rglob('*') if p.suffix in ('.py', '.cc', '.json')]
    shared = ('assets', 'config', 'offline', 'development', 'preprocessing',
              'bert_tables' if family == 'bert' else 'vit_parameters')
    files += [ROOT / 'src/riblet' / (name + '.py') for name in shared]
    return {p.relative_to(ROOT).as_posix(): _sha(p) for p in sorted(files)}


def cpu_environment(model):
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(source_root(model)), str(Path(__file__).resolve().parents[1]))
        ),
        "CDI_ARRAY_BACKEND": "numpy",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "BERT_CDI_LAYOUT": "pad128",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def fit_bert_point(operands, output, *, point, task, population=None, workers=16):
    model_spec("bert-" + task)
    if type(point) is not int or not 0 <= point < 37:
        raise ValueError("invalid causal point")
    operands, output = Path(operands).resolve(), Path(output).resolve()
    meta = json.loads(operands.with_name("input.json").read_text())
    lineage = "public-site-pad128"
    digest = _sha(operands)
    if (meta["task"], meta["point"], meta["lineage"], meta["sha256"]) != (
        task,
        point,
        lineage,
        digest,
    ):
        raise ValueError("point identity or operands changed")
    population = population or meta["population_sha256"]
    result_file = output / "result.json"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.with_name(output.name + ".log")
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(source_root("bert") / "fit_point.py"),
                    "--workers",
                    str(workers),
                    *(["--population-id", population] if population else []),
                    "--operands",
                    str(operands),
                    "--output",
                    str(output),
                    "--point",
                    str(point),
                    "--lineage",
                    lineage,
                    "--task",
                    task,
                ],
                env=cpu_environment("bert"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            raise RuntimeError(
                "point fitter failed; log: "
                + str(log)
                + "\n"
                + log.read_text(encoding="utf-8")[-2400:]
            )
    if not result_file.is_file():
        raise ValueError("incomplete point output; inspect failure before explicit recovery")
    result = json.loads(result_file.read_text())
    expected = {
        "point": point,
        "task": task,
        "lineage": lineage,
        "status": "point_fit_complete",
        "operands_sha256": digest,
        "model_hash": meta["model_hash"],
        "upstream_profiles": meta["upstream_profiles"],
        "table_sha256": _sha(output / "tables.json"),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("resumed fit does not match current point, prefix or table")
    return result


def validate_bert_checkpoint(folder, task):
    expected = model_spec("bert-" + task)
    config = json.loads((Path(folder) / "config.json").read_text())
    architecture = dict(
        model_type="bert",
        hidden_act="gelu",
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        vocab_size=30522,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
    )
    labels = config.get("num_labels", len(config["id2label"]) if "id2label" in config else None)
    if (
        any(config.get(k) != v for k, v in architecture.items())
        or config.get("max_position_embeddings", 0) < 128
    ):
        raise ValueError("checkpoint architecture differs from the supported BERT graph")
    if labels is not None and labels != expected["output_shape"][-1]:
        raise ValueError("checkpoint head does not match the task")

    return {
        **config,
        "num_labels": expected["output_shape"][-1],
        "head_metadata_present": labels is not None,
    }


def load_bert_weights(folder, task, *, bound=None, preprocessor=None):

    config = validate_bert_checkpoint(folder, task)
    from .assets import verify_model

    verify_model("bert-" + task, folder, bound=bound, preprocessor=preprocessor)
    import numpy as np
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from flax.serialization import msgpack_restore

    if jax.default_backend() != "cpu":
        raise ValueError("this construction adapter requires the CPU reference backend")
    raw = msgpack_restore((Path(folder) / "flax_model.msgpack").read_bytes())
    if np.asarray(raw["classifier"]["bias"]).shape != (config["num_labels"],):
        raise ValueError("checkpoint parameter head differs from its metadata")
    if set(raw["bert"]["encoder"]["layer"]) != {str(i) for i in range(12)}:
        raise ValueError("incomplete encoder layers")
    leaves, tree = jax.tree_util.tree_flatten(raw)
    digest = hashlib.sha256()
    encoded = []
    for leaf in leaves:
        value = np.asarray(leaf)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
        word = np.trunc(value.astype(np.float64) * (1 << 18))
        if not np.isfinite(word).all() or np.any(np.abs(word) >= 2**63):
            raise ValueError("model tensor cannot be encoded in FM64/f18")
        encoded.append(jnp.asarray(word.astype(np.int64)))
    return jax.tree_util.tree_unflatten(tree, encoded), digest.hexdigest()


def prepare_parameters(c, resume):

    from .assets import verify_model
    from .packages import (
        load_model_package,
        pack_frozen_bert,
        pack_frozen_vit,
        pack_constructed_bert,
        pack_constructed_vit,
    )
    from .development import custom_inputs, prepare_construction_inputs

    family, model, root = c["family"], c["model"], c["parameters"]
    if c["source"] == "frozen":
        if c["profile"] == "custom":
            pkg = load_model_package(root, rank=0)
            if pkg.get("profile") != "custom" or pkg["model"] != model:
                raise ValueError("expected a complete custom package for this model")
            verify_model(
                model, c["checkpoint"], bound=pkg["model_files"], preprocessor=c["processor"]
            )
            return pkg
        verify_model(model, c["checkpoint"], preprocessor=c["processor"])
        return (
            pack_frozen_bert(c["frozen"], model, root)
            if family == "bert"
            else pack_frozen_vit(c["frozen"], root)
        )
    if root.exists():
        raise FileExistsError("parameter output exists; use a fresh destination")
    off = c["construction"]
    base = c["construction_root"]
    if base.exists() and not resume:
        raise FileExistsError("construction exists; use --resume for the same inputs")
    base.parent.mkdir(parents=True, exist_ok=True)
    population = base / ("population.parquet" if family == "bert" else "population")
    raw = c["directory"] / off["data_root"]
    models = (
        c["directory"] / off["models_root"]
        if family == "bert" and (c["profile"] == "paper" or off["selection"].get("scope")=="all-paper-tasks")
        else c["checkpoint"].parent
    )
    out = base / "fit"
    from .assets import file_hash

    from .config import construction_policy

    policy = construction_policy(c)
    if c["profile"] == "custom":
        from .assets import verify_preprocessor

        bound = verify_model(
            model, c["checkpoint"], custom_weights=True, preprocessor=c["processor"]
        )
        verify_preprocessor(model, c["processor"], bound=bound)
        if family=="bert" and off["selection"].get("scope")=="all-paper-tasks":
            from .development import task_counts
            checkpoints={t:verify_model("bert-"+t,models/t,custom_weights=True) for t in task_counts(off["selection"]["count"])}
            if checkpoints[model[5:]]!=bound:raise ValueError("selected checkpoint differs from all-task model root")
            policy["model_files"]=checkpoints
        else:
            policy["model_files"] = bound
    state = base / "INPUTS.json"
    if off["selection"]["mode"] == "manifest":
        policy["membership_file_sha256"] = file_hash(c["directory"] / off["selection"]["path"])
    if c["profile"] == "custom" and family == "vit":
        policy["candidate_file_sha256"] = file_hash(c["directory"] / off["candidates"])
    if not resume:
        with tempfile.TemporaryDirectory(prefix=".riblet-inputs-", dir=base.parent) as temporary:
            stage = Path(temporary) / "construction"
            stage.mkdir()
            staged_population = stage / population.name
            if c["profile"] == "paper":
                prepare_construction_inputs(
                    family,
                    c["directory"] / off["membership"],
                    raw,
                    staged_population,
                    preprocessors=models,
                )
            else:
                custom_inputs(c, staged_population)
            with (stage / "INPUTS.json").open("x") as f:
                json.dump(
                    dict(
                        policy=policy,
                        artifact=file_hash(
                            staged_population
                            if family == "bert"
                            else staged_population / "manifest.json"
                        ),
                    ),
                    f,
                )
            stage.rename(base)
    elif json.loads(state.read_text()) != dict(
        policy=policy,
        artifact=file_hash(population if family == "bert" else population / "manifest.json"),
    ):
        raise ValueError("population or selection changed; start in a new directory")
    if family == "bert":
        args = ["--population", population, "--models", models, "--output", out]
        if c["profile"] == "custom" and off["selection"].get("scope")=="all-paper-tasks":
            args += ["--custom-all"]
        elif c["profile"] == "custom":
            args += [
                "--custom-task",
                model[5:],
                "--checkpoint",
                c["checkpoint"],
                "--preprocessor",
                c["processor"],
            ]
    else:
        args = [
            "--root",
            out,
            "--model",
            c["checkpoint"],
            "--preprocessor",
            c["processor"],
            "--data",
            population,
        ]
        if c["profile"] == "custom":
            args += ["--custom"]
    args += ["--workers", str(off["workers"])]
    if resume:
        args += ["--resume"]
    _construct(family, args)
    if family == "vit" and not (out / "constructed-joint/READY.json").is_file():
        raise ValueError("ViT numerical construction is incomplete")
    return (
        pack_constructed_bert(
            out / "private-sites.json", model, root, out / "CONSTRUCTION_RESULT.json"
        )
        if family == "bert"
        else pack_constructed_vit(out / "constructed-joint", root)
    )


def _construct(family, arguments):
    from .lifecycle import host_lease

    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "CDI_ARRAY_BACKEND": "jax",
        "PYTHONPATH": os.pathsep.join(map(str, (ROOT / "src", ROOT / "offline" / family))),
    }
    env.pop("RIBLET_COST_ENVIRONMENT", None)
    with host_lease():
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "offline" / family / "construct.py"), *arguments],
            env=env,
            check=True,
        )
    return {"status": "construction-command-finished", "family": family}
