import hashlib, json, os, time
from pathlib import Path
import numpy as np
import jax
import model as m, cache
from io_utils import (
    GRAPH_ID,
    atomic_json,
    digest,
    manifest,
    model_and_processor,
    pixels_host,
    residual,
)
from vit_cdi import numeric
from selection import SelectionPlan
from pipeline import prefetch
import abi
import point_timing

SHARD = 16


def retire(path, root):
    path = Path(path).resolve()
    base = Path(root).resolve()
    path.relative_to(base)
    if path.suffix != ".npz" or path.parent.name not in ("states", "prepared"):
        raise ValueError("not a consumable cache")
    if path.exists():
        path.unlink()


def selected_arrays(doc, row, point):
    return SelectionPlan(doc, point)(row)


def stack_rows(rows):
    return jax.tree_util.tree_map(lambda *a: np.stack([np.asarray(x) for x in a]), *rows)


def split_rows(tree, count):
    return [jax.tree_util.tree_map(lambda a: a[i], tree) for i in range(count)]


def save_array(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def run(args):
    service = args.point_service
    root = Path(args.root)
    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    policy = json.loads((root / "CONSTRUCTION_POLICY.json").read_text())
    if policy.get("mode") != abi.CONSTRUCTION_POLICY:
        raise ValueError("local numerical construction policy mismatch")
    if args.audit_after:
        raise ValueError("construction cannot deploy or evaluate a model")
    if (root / "RUN_COMPLETE.json").exists():
        raise FileExistsError("already complete; do not rerun")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise ValueError("CPU x64 construction required")
    rows, population = manifest(args.data, args.identity["profile"] == "custom")
    n = len(rows)
    shards = (n + SHARD - 1) // SHARD

    def binding(point, phase, prefix):
        return dict(graph=GRAPH_ID, population=population, point=point, phase=phase, prefix=prefix)

    params, processor = model_and_processor(args.model, args.preprocessor)
    kernels = m.Kernels()
    identity = dict(
        graph=GRAPH_ID,
        population=population,
        model_graph_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        numeric_sha256=hashlib.sha256(Path(m.p.__file__).read_bytes()).hexdigest(),
        model_id=args.identity["model_files"]["flax_model.msgpack"],
    )
    identity.update(
        selected_numeric_sha256=hashlib.sha256(
            Path(m.__file__).with_name("selected_kernel.py").read_bytes()
        ).hexdigest(),
        construction_policy=digest(policy),
    )
    if (root / "RUN_IDENTITY.json").exists() and json.loads(
        (root / "RUN_IDENTITY.json").read_text()
    ) != identity:
        raise ValueError("resume numerical graph/model identity changed")
    atomic_json(root / "RUN_IDENTITY.json", identity)
    prefix = []
    alive = np.ones(n, bool)
    er = np.zeros(n)
    es = er.copy()
    first = np.full(n, -1, np.int16)
    p12 = np.ones(n, bool)
    for point in range(37):
        folder = output / f"{point:02d}"
        folder.mkdir(exist_ok=True)
        summary = folder / "SUMMARY.json"
        before = digest(prefix)
        kind = m.kind(point)
        op = "reciprocal" if kind == 2 else "rsqrt"
        heads = 12 if kind == 2 else 1
        if summary.exists():
            result = json.loads(summary.read_text())
            doc = json.loads((folder / "selection.json").read_text())
            if result["prefix_before"] != before or result["selection_digest"] != digest(doc):
                raise ValueError("frozen point changed")
            with np.load(folder / "request_state.npz", allow_pickle=False) as f:
                alive = f["alive"]
                er = f["recip_error"]
                es = f["rsqrt_error"]
                first = f["first_failure"]
                p12 = f["p12_pass"]
            service.publish(point, folder / "EVIDENCE.json")
            prefix.append(result["selection_digest"])
            continue
        w = m.weights(params, point)
        start = time.time()
        previous = output / f"{point-1:02d}"
        timings = dict(
            operand_compute=0.0,
            prepare_io=0.0,
            propagation_compute=0.0,
            apply_io=0.0,
            fit_wait=0.0,
            projection=0.0,
        )
        timing_file = folder / "TIMING_CHECKPOINT.json"
        timing_state = point_timing.resume(
            json.loads(timing_file.read_text()) if timing_file.exists() else None, start
        )
        timings.update(timing_state["phase_seconds"])
        for name in [
            "prepared",
            "prepared_ledger",
            "inputs",
            "states",
            "applied",
            "applied_ledger",
        ]:
            (folder / name).mkdir(exist_ok=True)
        attempted = int(alive.sum())

        def load_previous(shard):
            started = time.perf_counter()
            start_id = shard * SHARD
            tag = f"{shard:04d}"
            ids = [i for i in range(start_id, min(n, start_id + SHARD)) if alive[i]]
            if point:
                states, saved = cache.load(
                    previous / "states" / f"{tag}.npz", binding(point - 1, "state", before)
                )
                if ids != saved:
                    raise ValueError("state arrival identities changed")
                images = np.zeros((len(ids), 1), np.int64)
            else:
                states = [None] * len(ids)
                images = (
                    np.stack([pixels_host(args.data, rows[i], processor) for i in ids])
                    if ids
                    else None
                )
            return shard, ids, states, images, time.perf_counter() - started

        pending_shards = [
            i for i in range(shards) if not (folder / "prepared_ledger" / f"{i:04d}.json").exists()
        ]
        for shard, ids, states, images, load_seconds in prefetch(pending_shards, load_previous):
            tag = f"{shard:04d}"
            mark = folder / "prepared_ledger" / f"{tag}.json"
            timings["prepare_io"] += load_seconds
            t = time.perf_counter()
            pending = []
            offset = 0
            if ids[offset:]:
                values = kernels.prepare_batch(
                    kind,
                    stack_rows(states[offset:]),
                    w,
                    images[offset:],
                    np.asarray([m.point_seed(rows[i], point) for i in ids[offset:]], np.uint64),
                )
                pending.extend(split_rows(jax.device_get(values), len(ids) - offset))
            timings["operand_compute"] += time.perf_counter() - t
            t = time.perf_counter()
            prepared = []
            good = []
            bad = []
            xs = []
            for rid, p in zip(ids, pending):
                x = np.asarray(p["operand"].value)
                if int(p["flags"]) or np.any(x <= 0):
                    bad.append(
                        dict(
                            request_id=rid,
                            stage="prepare",
                            flags=int(p["flags"]),
                            minimum_operand=int(x.min()),
                        )
                    )
                else:
                    prepared.append(p)
                    good.append(rid)
                    xs.append(x.reshape(heads, 197))
            dead = ("exps", "value") if kind == 2 else ("centered",)
            cache.save(
                folder / "prepared" / f"{tag}.npz",
                prepared,
                binding(point, "prepared", before),
                good,
                dead_state_fields=dead,
            )
            save_array(
                folder / "inputs" / f"{tag}.npz",
                request_ids=np.asarray(good, np.int64),
                operands=np.asarray(xs, np.int64).reshape(len(good), heads, 197),
            )
            atomic_json(mark, dict(input_ids=ids, prepared_ids=good, invalid=bad))
            if point:
                retire(previous / "states" / f"{tag}.npz", output)
            timings["prepare_io"] += time.perf_counter() - t
            if shard % 8 == 0:
                atomic_json(
                    root / "progress.json",
                    dict(
                        point=point,
                        phase="prepare",
                        shard=shard + 1,
                        total_shards=shards,
                        seconds=time.time() - start,
                    ),
                )
        if not (folder / "operands.npz").exists():
            ids = []
            xs = []
            for shard in range(shards):
                with np.load(folder / "inputs" / f"{shard:04d}.npz", allow_pickle=False) as f:
                    ids.append(f["request_ids"])
                    xs.append(f["operands"])
            ids = np.concatenate(ids)
            xs = np.concatenate(xs)
            save_array(
                folder / "operands.npz",
                request_ids=ids,
                folds=np.asarray([rows[int(i)]["fold_id"] for i in ids], np.int64),
                operands=xs,
            )
            del xs, ids
        if not (folder / "meta.json").exists():
            atomic_json(
                folder / "meta.json",
                dict(
                    point=point,
                    operator=op,
                    graph=GRAPH_ID,
                    population=population,
                    prefix=before,
                    operand_sha256=hashlib.sha256(
                        (folder / "operands.npz").read_bytes()
                    ).hexdigest(),
                ),
            )
        if not (folder / "selection.json").exists():
            fit_started = time.perf_counter()
            current = service.status(point)
            if "input_meta" not in current:
                for name in ["operands.npz", "meta.json"]:
                    service.submit(point, folder / name)
            status = current if current.get("status") == "fit_complete" else service.start(point)
            if status.get("construction_policy") != abi.CONSTRUCTION_POLICY:
                raise ValueError("point belongs to another construction policy")
            doc = service.selection(point)
            if digest(doc) != status["selection_digest"]:
                raise ValueError("CPU admitted selection changed during transfer")
            atomic_json(folder / "selection.json", doc)
            timings["fit_wait"] += time.perf_counter() - fit_started
        else:
            doc = json.loads((folder / "selection.json").read_text())
        if doc["input_meta"] != json.loads((folder / "meta.json").read_text()):
            raise ValueError("selected table input mismatch")
        abi.validate_admitted(doc)
        selected_digest = digest(doc)
        after = digest(prefix + [selected_digest])
        projection = SelectionPlan(doc, point)

        def load_prepared(shard):
            t = time.perf_counter()
            prep, ids = cache.load(
                folder / "prepared" / f"{shard:04d}.npz", binding(point, "prepared", before)
            )
            return shard, prep, ids, time.perf_counter() - t

        pending_shards = [
            i for i in range(shards) if not (folder / "applied_ledger" / f"{i:04d}.json").exists()
        ]
        for shard, prep, ids, load_seconds in prefetch(pending_shards, load_prepared):
            tag = f"{shard:04d}"
            mark = folder / "applied_ledger" / f"{tag}.json"
            timings["apply_io"] += load_seconds
            t = time.perf_counter()
            selections = [projection(rows[rid]) for rid in ids]
            timings["projection"] += time.perf_counter() - t
            t = time.perf_counter()
            outputs = []
            if ids:
                counts = (int(prep[0]["counter"]), int(prep[0]["truncations"]))
                if any((int(p["counter"]), int(p["truncations"])) != counts for p in prep):
                    raise ValueError("B1 public graph counters differ")
                values = kernels.finish_batch(
                    kind, stack_rows(prep), w, stack_rows(selections), counts
                )
                outputs = split_rows(jax.device_get(values), len(ids))
            timings["propagation_compute"] += time.perf_counter() - t
            t = time.perf_counter()
            good = []
            states = []
            bad = []
            ys = []
            rs = []
            normal_count = fallback_count = 0
            normal_fail = 0
            warnings = 0
            pred = []
            p12s = []
            warning_bits = []
            for rid, (state, r) in zip(ids, outputs):
                x = np.asarray(r["operand"]).reshape(-1)
                y = np.asarray(r["inverse"]).reshape(-1)
                normal = np.asarray(r["normal"]).reshape(-1)
                checks = numeric.precision(op, x, y, 12, False)
                normal_fail += int(np.count_nonzero(normal & ~checks))
                normal_count += int(normal.sum())
                fallback_count += int((~normal).sum())
                warnings += int(int(r["warnings"]) != 0)
                warning_bits.append(int(r["warnings"]))
                err = float(residual(op, x, y).max())
                invalid = int(r["flags"]) != 0 or np.any(y <= 0)
                p12s.append(bool(not invalid and np.all(numeric.precision(op, x, y, 12, True))))
                rs.append(None if invalid else err)
                ys.append(y.reshape(heads, 197))
                if invalid:
                    bad.append(dict(request_id=rid, stage="finish", flags=int(r["flags"])))
                else:
                    good.append(rid)
                    states.append(state)
                if point == 36:
                    pred.append(None if invalid else int(np.asarray(state["logits"]).argmax()))
            if normal_fail:
                raise RuntimeError(
                    f"selected-kernel normal p12 inconsistency at point{point} shard{shard}: {normal_fail}"
                )
            cache.save(
                folder / "states" / f"{tag}.npz", states, binding(point, "state", after), good
            )
            save_array(
                folder / "applied" / f"{tag}.npz",
                request_ids=np.asarray(ids, np.int64),
                inverse=np.asarray(ys, np.int64).reshape(len(ids), heads, 197),
            )
            atomic_json(
                mark,
                dict(
                    request_ids=ids,
                    output_ids=good,
                    invalid=bad,
                    residuals=rs,
                    normal_events=normal_count,
                    fallback_events=fallback_count,
                    normal_p12_failures=normal_fail,
                    warning_requests=warnings,
                    warning_bits=warning_bits,
                    predictions=pred,
                    p12_passes=p12s,
                ),
            )
            retire(folder / "prepared" / f"{tag}.npz", output)
            timings["apply_io"] += time.perf_counter() - t
            if shard % 8 == 0:
                atomic_json(
                    root / "progress.json",
                    dict(point=point, phase="apply", shard=shard + 1, total_shards=shards),
                )
        prepared_count = normal_events = fallback_events = prep_fail = post_fail = 0
        warnings = 0
        for shard in range(shards):
            pre = json.loads((folder / "prepared_ledger" / f"{shard:04d}.json").read_text())
            app = json.loads((folder / "applied_ledger" / f"{shard:04d}.json").read_text())
            prepared_count += len(pre["prepared_ids"])
            prep_fail += len(pre["invalid"])
            post_fail += len(app["invalid"])
            for failure in pre["invalid"] + app["invalid"]:
                i = failure["request_id"]
                alive[i] = False
                first[i] = point
                er[i] = es[i] = np.inf
                p12[i] = False
            target = er if op == "reciprocal" else es
            for i, value in zip(app["request_ids"], app["residuals"]):
                target[i] = max(target[i], np.inf if value is None else value)
            for i, passed in zip(app["request_ids"], app["p12_passes"]):
                p12[i] &= passed
            normal_events += app["normal_events"]
            fallback_events += app["fallback_events"]
            warnings += app["warning_requests"]
        save_array(
            folder / "request_state.npz",
            alive=alive,
            recip_error=er,
            rsqrt_error=es,
            first_failure=first,
            p12_pass=p12,
        )
        result = dict(
            point=point,
            operator=op,
            prefix_before=before,
            prefix_after=after,
            selection_digest=selected_digest,
            population=n,
            attempted_requests=attempted,
            prepared_finite_requests=prepared_count,
            inherited_invalid_before=n - attempted,
            prepare_invalid=prep_fail,
            finish_invalid=post_fail,
            alive_after=int(alive.sum()),
            events_per_request=heads * 197,
            theoretical_events=n * heads * 197,
            executed_inverse_events=prepared_count * heads * 197,
            unreached_inherited_events=(n - attempted) * heads * 197,
            invalid_input_unexecuted_events=prep_fail * heads * 197,
            normal_events=normal_events,
            fallback_events=fallback_events,
            normal_p12_failures=0,
            warning_requests=warnings,
            **point_timing.finish(timing_state, time.time()),
            phase_seconds=timings,
            timing_note="prefetch I/O can overlap CPU calculation; active seconds exclude stopped intervals; fit_wait is local worker time",
        )
        prefix.append(selected_digest)
        evidence = dict(
            summary=result,
            prepared=[
                json.loads((folder / "prepared_ledger" / f"{i:04d}.json").read_text())
                for i in range(shards)
            ],
            applied=[
                json.loads((folder / "applied_ledger" / f"{i:04d}.json").read_text())
                for i in range(shards)
            ],
            request_state=dict(
                alive=alive.tolist(),
                recip_error=[float(x) if np.isfinite(x) else None for x in er],
                rsqrt_error=[float(x) if np.isfinite(x) else None for x in es],
                first_failure=first.tolist(),
                p12_pass=p12.tolist(),
            ),
        )
        atomic_json(folder / "EVIDENCE.json", evidence)
        service.publish(point, folder / "EVIDENCE.json")
        atomic_json(summary, result)
    atomic_json(
        root / "CONSTRUCTION_DONE.json",
        dict(
            points=37,
            prefix=digest(prefix),
            alive=int(alive.sum()),
            numeric_invalid=int((~alive).sum()),
        ),
    )
    if not args.audit_after:
        atomic_json(
            root / "progress.json",
            dict(
                phase="numerical_causal_construction_complete",
                points=37,
                formal_deployment_allowed=False,
            ),
        )
        return
