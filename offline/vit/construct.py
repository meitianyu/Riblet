"""CPU Domain and Coefficient Construction for ViT."""

from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, os, sys

os.environ.setdefault("CDI_ARRAY_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from riblet.assets import verify_model
from riblet.offline import construction_sources
from io_utils import manifest, atomic_json
import abi


def prepare_run(root, model, data, *, resume=False, workers=28, custom=False, preprocessor=None):
    root, model, data = map(Path, (root, model, data))
    rows, population = manifest(data, custom)
    model_files = verify_model(
        "vit-base-patch16-224", model, custom_weights=custom, preprocessor=preprocessor
    )
    identity = dict(
        profile="custom" if custom else "paper",
        population=population,
        model_files=model_files,
        requests=len(rows),
        construction_policy=abi.CONSTRUCTION_POLICY,
        population_artifact_sha256=hashlib.sha256(
            (data / "manifest.json").read_bytes()
        ).hexdigest(),
        source_hashes=construction_sources('vit'),
    )
    if root.exists() and not resume:
        raise FileExistsError("explicit construction resume required")
    binding = root / "CONSTRUCTION_BINDING.json"
    if root.exists() and not binding.exists():
        raise ValueError("unbound existing construction directory")
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("construction source/data/model/numerical policy changed")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(binding, identity)
    from local_point import LocalPointService

    service = LocalPointService(root, data / "manifest.json", workers=workers)
    policy = dict(mode=abi.CONSTRUCTION_POLICY)
    path = root / "CONSTRUCTION_POLICY.json"
    if path.exists() and json.loads(path.read_text()) != policy:
        raise ValueError("construction policy changed")
    atomic_json(path, policy)
    return SimpleNamespace(
        root=str(root),
        model=str(model),
        data=str(data),
        audit_after=False,
        point_service=service,
        identity=identity,
        preprocessor=str(preprocessor or model),
    )


def run_local(*args, **kwargs):
    import fcntl

    prepared = prepare_run(*args, **kwargs)
    root = Path(prepared.root)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / "CONSTRUCTION_DONE.json").exists():
            import causal

            causal.run(prepared)
        if (root / "CONSTRUCTION_DONE.json").exists():
            if not (root / "constructed-joint/READY.json").exists():
                from audit import audit

                audit(root)
            return dict(
                status="audited_construction_complete",
                parameter_source=str(root / "constructed-joint"),
                MPC_measurements_started_by_controller=False,
            )
        raise ValueError("numerical construction did not complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=28)
    p.add_argument("--custom", action="store_true")
    p.add_argument("--preprocessor")
    a = p.parse_args()
    print(
        json.dumps(
            run_local(
                a.root,
                a.model,
                a.data,
                resume=a.resume,
                workers=a.workers,
                custom=a.custom,
                preprocessor=a.preprocessor,
            )
        )
    )
