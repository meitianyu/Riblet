"""CPU Domain and Coefficient Construction for BERT."""

from pathlib import Path
import argparse, gc, hashlib, json, os, sys
from types import SimpleNamespace

os.environ.setdefault("CDI_ARRAY_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ["BERT_CDI_LAYOUT"] = "pad128"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from collections import Counter
from riblet.assets import verify_model, file_hash
from riblet.offline import fit_bert_point, load_bert_weights, construction_sources
from baseline_io import sha, write_json
from population import TASK_COUNTS, validate_rows
import causal

from riblet.development import MEMBERSHIP_SHA, content_hash, FIELDS


def load_population(path, custom=False):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if custom:
        counts = dict(Counter(r["task"] for r in rows))
        if not rows or len(rows) >= 2**47 or not set(counts) <= set(TASK_COUNTS):
            raise ValueError("invalid custom BERT population")
        validate_rows(
            rows, task_counts=counts, token_count=sum(r["effective_length"] for r in rows)
        )
    else:
        validate_rows(rows)
    encoded = json.dumps(
        [{k: r[k] for k in FIELDS} for r in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    identity = hashlib.sha256(encoded).hexdigest()
    if not custom:
        membership = (
            Path(__file__).resolve().parents[2]
            / "assets/reproduction/construction-bert/bert-development-membership.json"
        )
        if file_hash(membership) != MEMBERSHIP_SHA:
            raise ValueError("paper membership changed")
        members = json.loads(membership.read_bytes())["rows"]
        for member, row in zip(members, rows, strict=True):
            if any(row[k] != v for k, v in member.items()):
                raise ValueError("paper member or fold changed")
            if (
                content_hash(
                    member, row["effective_length"], row["input_ids"], row["token_type_ids"]
                )
                != member["content_sha256"]
            ):
                raise ValueError("training input content changed")
    return rows, identity


class LocalFitter:
    def __init__(self, population, workers):
        self.population, self.workers = population, workers

    def fit(self, point_folder, mode, task, point):
        if mode != "joint":
            raise ValueError("only complete Joint construction is supported")
        result = fit_bert_point(
            point_folder / "operands.npz",
            point_folder / "fit",
            point=point,
            task=task,
            population=self.population,
            workers=self.workers,
        )
        return causal.load_profile(point_folder / "fit/tables.json"), result


def run_local(
    population,
    model_root,
    output,
    *,
    resume=False,
    fixture_task=None,
    fixture_requests=None,
    stop_after_point=36,
    custom_task=None,
    custom_all=False,
    checkpoint=None,
    workers=16,
    preprocessor=None
):
    import fcntl
    import runtime

    root = Path(output)
    model_root = Path(model_root)
    fixture = fixture_task is not None
    if type(stop_after_point) is not int or not 0 <= stop_after_point <= 36:
        raise ValueError("invalid causal boundary")
    if not fixture and (fixture_requests is not None or stop_after_point != 36):
        raise ValueError("partial construction must be explicitly a non-deployable fixture")
    if fixture and (
        fixture_task not in TASK_COUNTS
        or type(fixture_requests) is not int
        or not 1 <= fixture_requests <= TASK_COUNTS[fixture_task]
    ):
        raise ValueError("invalid fixture request subset")
    if not 1 <= workers <= 64:
        raise ValueError("invalid workers")
    if custom_all and (fixture or custom_task or checkpoint):
        raise ValueError("all-task construction uses its model root")
    custom = custom_task is not None or custom_all
    rows, pop_id = load_population(population, custom)
    if custom_all and {r["task"] for r in rows} != set(TASK_COUNTS):
        raise ValueError("eight model population required")
    if custom_task and (fixture or {r["task"] for r in rows} != {custom_task}):
        raise ValueError("custom population/task mismatch")
    tasks = (
        [fixture_task]
        if fixture
        else (
            [custom_task]
            if custom_task
            else ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
        )
    )
    if fixture:
        rows = [r for r in rows if r["task"] == fixture_task][:fixture_requests]
    folders = {t: (Path(checkpoint) if custom_task else model_root / t) for t in tasks}
    files = {
        t: verify_model(
            "bert-" + t,
            folders[t],
            custom_weights=custom,
            preprocessor=preprocessor if custom else None,
        )
        for t in tasks
    }
    identity = dict(
        kind="fixture" if fixture else "constructed",
        population_semantic_sha256=pop_id,
        population_artifact_sha256=sha(population),
        graph_id=causal.GRAPH_ID,
        graph_code_hash=causal.GRAPH_CODE_HASH,
        layout="pad128",
        mode="joint",
        fixture_task=fixture_task,
        fixture_requests=fixture_requests,
        stop_after_point=stop_after_point,
        profile="custom" if custom else "paper",
        model_files=files,
        source_hashes=construction_sources("bert"),
    )
    if root.exists() and not resume:
        raise FileExistsError("explicit resume required")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = root / "RUN_IDENTITY.json"
        if binding.exists() and json.loads(binding.read_text()) != identity:
            raise ValueError("construction identity changed; use a new output namespace")
        write_json(binding, identity)
        args = SimpleNamespace(
            stop_after_point=stop_after_point,
            population_id=(
                pop_id
                if custom
                else "4e48dd15c654afa51d8cdbc017cf4b6c2b50aaf3d25536e8dc65adf03dbf4529"
            ),
        )
        completed = []
        documents = []
        kernels = runtime.BatchedKernels()
        fitter = LocalFitter(args.population_id, workers)

        def status(mode, task, point, phase, **extra):
            write_json(
                root / "STATUS.json", dict(mode=mode, task=task, point=point, phase=phase, **extra)
            )

        for task in tasks:
            params, model_hash = load_bert_weights(
                folders[task],
                task,
                bound=files[task],
                preprocessor=preprocessor if custom else None,
            )
            kernels.inputs.clear()
            result, document = causal.run_task(
                args,
                root,
                "joint",
                task,
                [r for r in rows if r["task"] == task],
                params,
                model_hash,
                kernels,
                fitter,
                status,
            )
            completed.append(result)
            documents.append(document)
            del params
            gc.collect()
        table = root / ("fixture-tables.json" if fixture else "private-sites.json")
        merged = causal.merge_profiles(documents, table)
        if not fixture:
            import secrets

            schedule_path = root / "public-schedule.json"
            existing = json.loads(schedule_path.read_bytes()) if schedule_path.exists() else None
            version = existing["version"] if existing else secrets.token_hex(16)
            schedule = dict(
                schema=merged["schema"],
                version=version,
                layout="pad128",
                external_format="FM64/f18",
                cls_recip="SPU at query 0",
                padding="physical SPU; excluded from fit",
                sites=[
                    {k: r[k] for k in ("task", "point", "site", "operator", "enabled", "K")}
                    for r in merged["sites"]
                ],
            )
            if existing is not None and schedule != existing:
                raise ValueError("resumed deployment schedule differs")
            if existing is None:
                write_json(schedule_path, schedule)
        result = dict(
            status="fixture_complete" if fixture else "causal_construction_complete",
            profile=identity["profile"],
            model_files=files,
            task_counts=dict(Counter(r["task"] for r in rows)),
            deployment_allowed=not fixture,
            task_results=completed,
            table_sha256=sha(table),
            population_semantic_sha256=pop_id,
            graph_id=causal.GRAPH_ID,
            graph_code_hash=causal.GRAPH_CODE_HASH,
            numerical_guarantee="finite_design_population_checks_only",
            current_environment_cost_measurements=0,
        )
        write_json(root / "CONSTRUCTION_RESULT.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture-task", choices=list(TASK_COUNTS))
    parser.add_argument("--fixture-requests", type=int)
    parser.add_argument("--stop-after-point", type=int, default=36)
    parser.add_argument("--custom-all", action="store_true")
    parser.add_argument("--custom-task", choices=list(TASK_COUNTS))
    parser.add_argument("--checkpoint")
    parser.add_argument("--preprocessor")
    parser.add_argument("--workers", type=int, default=16)
    a = parser.parse_args()
    print(
        json.dumps(
            run_local(
                a.population,
                a.models,
                a.output,
                resume=a.resume,
                fixture_task=a.fixture_task,
                fixture_requests=a.fixture_requests,
                stop_after_point=a.stop_after_point,
                custom_task=a.custom_task,
                custom_all=a.custom_all,
                checkpoint=a.checkpoint,
                workers=a.workers,
                preprocessor=a.preprocessor,
            )
        )
    )
