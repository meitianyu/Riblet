from pathlib import Path
import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
from cdi_bert.fitting import ArrivedEvents, fit_candidate, SupportGate
from cdi_bert.batch import _library
from scoring import PlaintextBackend
from experiment_contract import load_config, validate_input_binding, numeric_scope, LINEAGES
from public_sites import site, pool_mask, SCHEMA
from baseline_io import sha, write_json


def fit_head(h):
    x, ids, folds, lengths, pos, mask, point, task = shared
    rec = point % 3 == 1
    ev = (
        (ids.astype(np.uint64) << np.uint64(17)) | np.uint64(point << 11) | np.uint64(h << 7) | pos
    )[mask]
    pooled = x[h, mask]
    events = ArrivedEvents(ids[mask], folds[mask], lengths[mask], pooled, ev)
    s = site(point, h)
    f = fit_candidate(
        events,
        operator="reciprocal" if rec else "rsqrt",
        lineage_id=LINEAGES["joint"],
        model_or_task_id=task,
        public_site_id=s,
        support_gate=SupportGate(),
        scoring_backend=PlaintextBackend("CDI_G9_V2_DESIGN_0"),
        fit_purpose="plaintext_fit",
    )
    feasible = list(f.feasible_Ks)
    full = f.full_score
    row = dict(
        task=task,
        point=point,
        site=s,
        operator=f.operator,
        enabled=bool(feasible),
        K=min(feasible) if feasible else None,
        feasible_K=feasible,
        reason=f.rejection_reason
        or ("strict_oof_and_full_pass" if feasible else "empty_feasible_intersection"),
        events=f.support_event_count,
        requests=f.support_request_count,
        seed_scope=numeric_scope(f.candidate_id) if feasible else None,
        raw_L=int(pooled.min()) if pooled.size else None,
        raw_H=int(pooled.max()) if pooled.size else None,
        L=full.L_encoded if full else None,
        H=full.H_encoded if full else None,
        a_encoded=full.a_encoded if full else None,
        b_encoded=full.b_encoded if full else None,
        H_over_L=full.H_encoded / full.L_encoded if full else None,
        format="FM64/f18",
    )
    a = f.audit_record()
    return row, dict(
        site=s, fold_scores=a["fold_scores"], full_score=a["full_score"], reason=row["reason"]
    )


def main():
    global shared
    parser = argparse.ArgumentParser()
    for key in ("operands", "output", "task", "lineage"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--point", required=True, type=int)
    parser.add_argument("--population-id", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    load_config()
    folder = Path(args.operands)
    meta = json.loads(folder.with_name("input.json").read_bytes())
    validate_input_binding(meta, args.lineage, args.population_id)
    point, task = args.point, args.task
    if (meta["task"], meta["point"], meta["lineage"], meta["sha256"]) != (
        task,
        point,
        args.lineage,
        sha(folder),
    ):
        raise ValueError("point/prefix input identity mismatch")
    with np.load(folder, allow_pickle=False) as f:
        x, ids, folds, lens = (f[k] for k in ("operands", "request_ids", "folds", "lengths"))
    heads = 12 if point % 3 == 1 else 1
    if (
        x.dtype != np.int64
        or x.shape != (heads, int(lens.sum()))
        or not (len(ids) == len(lens) == len(folds))
        or np.any(np.diff(ids) <= 0)
        or np.any((lens < 1) | (lens > 128))
        or np.any(folds > 4)
    ):
        raise ValueError("operand/fold registry mismatch")
    pos = (
        np.concatenate([np.arange(n, dtype=np.uint64) for n in lens])
        if len(lens)
        else np.empty(0, np.uint64)
    )
    shared = (
        x,
        np.repeat(ids, lens),
        np.repeat(folds, lens),
        np.repeat(lens, lens),
        pos,
        pool_mask(lens, heads == 12),
        point,
        task,
    )
    workers = min(args.workers, heads, os.cpu_count() or 1)
    if workers == 1:
        fits = [fit_head(h) for h in range(heads)]
    else:
        _library()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            fits = list(pool.map(fit_head, range(heads)))
    out = Path(args.output)
    out.mkdir()
    write_json(
        out / "tables.json",
        dict(schema=SCHEMA, design=args.population_id, layout="pad128", sites=[r for r, _ in fits]),
    )
    write_json(out / "audit.json", [v for _, v in fits])
    write_json(
        out / "result.json",
        dict(
            status="point_fit_complete",
            point=point,
            task=task,
            lineage=args.lineage,
            model_hash=meta["model_hash"],
            upstream_profiles=meta["upstream_profiles"],
            operands_sha256=meta["sha256"],
            table_sha256=sha(out / "tables.json"),
        ),
    )


if __name__ == "__main__":
    main()
