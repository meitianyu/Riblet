"""Domain and Coefficient Construction, Cryptographic preprocessing, and Online inference."""

import argparse, json, os, secrets, sys
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu"
)
from .config import rank_for, load_run_config
from .offline import prepare_parameters as _prepare_model

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["resources", "inputs", "prepare-model", "describe", "compile", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=["model", "data"])
    parser.add_argument(
        "--resume", action="store_true", help="resume an unchanged numerical construction"
    )
    args = parser.parse_args(argv)
    config = load_run_config(args.config, args.role, args.command)
    if args.resume and (args.command != "prepare-model" or config["source"] != "construct"):
        raise ValueError("--resume is only for Domain and Coefficient Construction")
    if (args.command in ("prepare-model", "compile") and args.role != "model") or (
        args.command == "describe" and args.role != "data"
    ):
        raise PermissionError("command is not available to this role")
    if args.command == "prepare-model":
        return _prepare_model(config, args.resume)
    handlers = {"resources": _resources, "inputs": _inputs, "describe": _describe, "compile": _compile, "run": _run}
    return handlers[args.command](config)


def _describe(config):
    from .inputs import load_input
    from .config import model_spec
    from .packages import load_public
    from .paper_inputs import resolve_paper_input

    input_path, _ = resolve_paper_input(config)

    model = config["model"]
    package = load_public(config["parameters"]) if config["profile"] == "custom" else None
    if package and (package["model"] != model or package.get("profile") != "custom"):
        raise ValueError("custom description requires corresponding public parameters")
    _, description = load_input(
        1,
        model,
        input_path,
        config["processor"],
        bound=package["preprocessor_files"] if package else None,
    )
    if package:
        description["parameter_package"] = package["package_id"]
    description.update(
        model=model, layout=model_spec(model)["layout"], request=config["request_id"]
    )
    config["description"].parent.mkdir(parents=True, exist_ok=True)
    with config["description"].open("x") as stream:
        json.dump(description, stream, indent=2)
    return description


def _activate_deployment(config):
    from .packages import load_public
    from .environment import activate

    package = load_public(config["parameters"])
    if (
        package["model"] != config["model"]
        or package.get("profile", "paper") != config["profile"]
        or (config["source"] == "construct" and package["kind"] != "constructed-deployment")
    ):
        raise ValueError("configured model/profile does not match the parameter package")
    activate(config, config["family"])
    return package


def _compile(config):
    from .lifecycle import host_lease

    package = _activate_deployment(config)
    sys.path.insert(0, str(ROOT / "frontends" / config["family"]))
    from compile_model import build

    epochs = dict(model_epoch=secrets.randbits(48) + 1, request_epoch=secrets.randbits(48) + 1)
    if config["family"] == "bert":
        description = json.loads(config["description"].read_text())
        if (description["model"], description["layout"], description["request"]) != (
            config["model"],
            "pad128",
            config["request_id"],
        ):
            raise ValueError("public request description mismatch")
        if (
            config["profile"] == "custom"
            and description.get("parameter_package") != package["package_id"]
        ):
            raise ValueError("description generated with another custom preprocessor/package")
        epochs["effective_length"] = description["effective_length"]
    config["graph"].parent.mkdir(parents=True, exist_ok=True)
    with host_lease():
        return build(
            config["checkpoint"],
            config["parameters"],
            config["graph"],
            preprocessor_path=config["processor"],
            **epochs,
        )


def _run(config):
    from .paper_inputs import resolve_paper_input

    model_host = config["role"] == "model"
    input_path, binding = (None, None) if model_host else resolve_paper_input(config)
    _activate_deployment(config)
    from .party import run_full

    result_existed = config["result"].exists()
    status = "failed"
    record = None
    try:
        record = run_full(
            rank=rank_for(config["model"], config["role"]),
            parties=config["parties"],
            graph_dir=config["graph"],
            public_parameters=config["parameters"],
            state_dir=config["state_dir"],
            result_dir=config["result"],
            model=config["checkpoint"] if model_host else None,
            private_parameters=config["parameters"] if model_host else None,
            client_input=input_path,
            preprocessor=None if model_host else config["processor"],
            model_preprocessor=config["processor"] if model_host else None,
            network=config["network"],
            request_alias=config["request_id"],
        )
        status = "complete"
        return record
    finally:

        if binding is not None and not result_existed and config["result"].is_dir():
            export_status = "complete"
            try:
                (config["result"] / "paper-input.json").write_text(
                    json.dumps(dict(binding, execution_status=status), indent=2)
                )
            except OSError as error:
                export_status = "failed"
                print(
                    f"Warning: input provenance export failed: {error}. "
                    f"Protocol status remains {status}; inspect result.json or FAILED.json "
                    "before retrying. Do not rerun completed MPC solely for this export.",
                    file=sys.stderr,
                )
            if record is not None:
                record["provenance_export_status"] = export_status


def _inputs(config):
    from .paper_inputs import prepare_paper_inputs

    return prepare_paper_inputs(config)


def _resources(c):
    import subprocess

    r = c["resources"]
    if c["profile"] == "custom":
        raise ValueError("provide your own custom assets locally; do not fetch paper tables")
    cmd = [
        sys.executable,
        str(ROOT / "tools/fetch_resources.py"),
        "--model",
        c["model"],
        "--role",
        c["role"],
        "--route",
        c["source"],
        "--manifest",
        c["directory"] / r["catalogue"],
        "--cache",
        c["directory"] / r["cache"],
        "--unpack-to",
        c["directory"] / r["directory"],
    ]
    if r.get("local_dir"):
        cmd += ["--local-directory", c["directory"] / r["local_dir"]]
    subprocess.run(cmd, check=True)
    return {"status": "resources-verified"}


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in ("status", "kind", "model", "family", "package_id", "model_rank", "data_rank", "provenance_export_status")
            },
            ensure_ascii=False,
        )
    )
