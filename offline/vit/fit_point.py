import argparse, json, time, sqlite3, multiprocessing as mp, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
from vit_cdi import candidates, layout
from io_utils import GRAPH_ID, atomic_json, digest
from pools import Pool, source_raws
from initial_certificate import initial_certificate
import trajectory, abi, point_rules

POOLS = {}


@lru_cache(maxsize=65536)
def coefficient_family(op, raws, family):
    domains = tuple(candidates.interval(op, *r) for r in raws)
    return domains, tuple(candidates.coefficients(op, L, H, family) for L, H in domains)


def score(spec):
    row, coefs = spec
    op = row["operator"]
    cap = abi.CAP[op]
    pool = POOLS[row["pool_key"]]
    scores = []
    mask = (1 << (cap + 1)) - 1
    opts = dict(bits=22, output="nearest") if op == "reciprocal" else {}
    for fold, ((x, ids, bounds), (a, b)) in enumerate(zip(pool.folds, coefs)):
        certified = initial_certificate(op, *bounds, a, b, opts.get("bits", 18))
        result = trajectory.score(
            x, ids, a, b, op, cap, abi.scope(op, row["site"], fold), certified=certified, **opts
        )
        if result["mask"] and result["visited"] != len(x):
            raise ValueError("incomplete feasible candidate")
        mask &= result["mask"]
        scores.append(
            dict(
                fold=fold,
                scored_events=len(x),
                visited=result["visited"],
                mask=result["mask"],
                initial_valid=result["initial_valid"],
                initial_certificate=certified,
                failure_ids=result["failure_ids"],
                failure_kind=result["failure_kind"],
                a_encoded=a,
                b_encoded=b,
            )
        )
    result = {k: v for k, v in row.items() if k != "pool_key"}
    result.update(
        scores=scores,
        fold_coefficients=coefs,
        fold_masks=[s["mask"] for s in scores],
        qualified=bool(mask),
        feasible_Ks=[k for k in range(cap + 1) if mask & (1 << k)],
        initial_valid=all(s["initial_valid"] for s in scores),
        scanned_K_max=cap,
        cost_assessment="not_measured",
    )
    return result


def fit(root, point, workers):
    global POOLS
    root = Path(root)
    policy = json.loads((root / "CONSTRUCTION_POLICY.json").read_text())
    if policy.get("mode") != abi.CONSTRUCTION_POLICY:
        raise ValueError("numerical construction policy mismatch")
    out = root / "points" / f"{point:02d}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SCORING_DONE.json").exists():
        return finalize_numeric(root, point)
    inp = root / "inputs" / f"{point:02d}"
    meta = json.loads((inp / "meta.json").read_text())
    binding_doc = json.loads((root.parent / "CONSTRUCTION_BINDING.json").read_text())
    population = binding_doc["population"]
    if meta["graph"] != GRAPH_ID or meta["population"] != population or meta["point"] != point:
        raise ValueError("wrong graph/population")
    if hashlib.sha256((inp / "operands.npz").read_bytes()).hexdigest() != meta["operand_sha256"]:
        raise ValueError("input transfer mismatch")
    with np.load(inp / "operands.npz") as d:
        xs = d["operands"]
        rids = d["request_ids"]
        folds = d["folds"]
    master = json.loads((root / "population.json").read_text())["rows"]
    if np.any(rids < 0) or np.any(rids >= len(master)):
        raise ValueError("unknown dense request index")
    if len(rids) != len(np.unique(rids)) or not np.array_equal(
        folds, [master[int(r)]["fold_id"] for r in rids]
    ):
        raise ValueError("identity/fold mismatch")
    op = "reciprocal" if point % 3 == 1 else "rsqrt"
    heads = 12 if op == "reciprocal" else 1
    if xs.dtype != np.int64 or xs.shape != (len(rids), heads, 197) or np.any(xs <= 0):
        raise ValueError("invalid input shape/word")
    POOLS = {}
    jobs = []
    site_counts = {}
    started = time.time()
    families = candidates.families(op)
    db = sqlite3.connect(out / "candidates.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("create table if not exists binding(value text)")
    binding = digest(dict(meta=meta, policy=policy, numeric=abi.NUMERIC[op]))
    old = db.execute("select value from binding").fetchone()
    if old and old[0] != binding:
        raise ValueError("candidate cache binding mismatch")
    if not old:
        db.execute("insert into binding values(?)", (binding,))
    db.execute(
        "create table if not exists candidates(id text primary key,site integer,payload text)"
    )
    db.commit()
    done = {r[0] for r in db.execute("select id from candidates")}
    expected = set()
    for h in range(heads):
        site = (
            point // 3 * 12 + h
            if op == "reciprocal"
            else (24 if point == 36 else 2 * (point // 3) + int(point % 3 == 2))
        )
        roles = np.array([layout.condition(op, p) for p in range(197)])
        site_counts[site] = {c: int((roles == c).sum()) * len(rids) for c in layout.conditions(op)}
        if len(rids) < 919 or any(
            np.count_nonzero(folds != f) < 100 or not np.any(folds == f) for f in range(5)
        ):
            continue
        parent_x = xs[:, h, 1:].reshape(-1)
        parent_folds = np.repeat(folds, 196)
        parent_raws = tuple(source_raws(parent_x, parent_folds))
        for target in layout.conditions(op):
            if op == "reciprocal" and target == "CLS":
                continue
            pp = np.flatnonzero(roles == target)
            xx = xs[:, h, pp].reshape(-1)
            ff = np.repeat(folds, len(pp))
            rr = np.repeat(rids, len(pp))
            ids = (
                ((rr.astype(np.uint64) * 37 + point) * 12 + h) * 197 + np.tile(pp, len(rids))
            ).astype(np.uint64)
            key = f"{site}:{target}"
            pool = Pool(xx, ids, ff)
            POOLS[key] = pool
            sources = [("CHILD", tuple(pool.raws))]
            if target != "CLS":
                sources.append(("PATCH_PARENT", parent_raws))
            for source, raws in sources:
                for family in families:
                    tag = (
                        "analytic"
                        if family[0] == "analytic"
                        else f"robust_j{family[1]:02d}_m{family[2]+16:02d}"
                    )
                    ident = f"VIT_R3:{op}:site{site}:{source}:target{target}:{tag}"
                    expected.add(ident)
                    if ident in done:
                        continue
                    row = dict(
                        candidate_id=ident,
                        operator=op,
                        site=site,
                        source=source,
                        condition=target,
                        family=list(family),
                        pool_key=key,
                        support_request_count=len(rids),
                        support_event_count=len(parent_x) if source == "PATCH_PARENT" else len(xx),
                        source_raws=raws,
                        target_event_count=len(xx),
                        numeric_abi=abi.NUMERIC[op],
                        rng_scope=abi.scope(op, site, 5),
                        design_bits=12,
                        design_inclusive=False,
                    )
                    try:
                        domains, coefs = coefficient_family(op, raws, family)
                        row.update(
                            L_encoded=domains[-1][0],
                            H_encoded=domains[-1][1],
                            a_encoded=coefs[-1][0],
                            b_encoded=coefs[-1][1],
                            source_domains=domains,
                        )
                        jobs.append((row, coefs))
                    except (ValueError, OverflowError):
                        result = {k: v for k, v in row.items() if k != "pool_key"}
                        result.update(
                            qualified=False,
                            feasible_Ks=[],
                            initial_valid=False,
                            rejection_reason="coefficient_encoding_or_domain_failure",
                        )
                        db.execute(
                            "insert into candidates values(?,?,?)",
                            (ident, site, json.dumps(result)),
                        )
    db.commit()
    trajectory.library()
    completed = db.execute("select count(*) from candidates").fetchone()[0]
    atomic_json(
        out / "progress.json",
        dict(
            phase="fit", point=point, completed_candidates=completed, total_candidates=len(expected)
        ),
    )
    with ProcessPoolExecutor(
        max_workers=min(workers, max(1, len(jobs))), mp_context=mp.get_context("fork")
    ) as pool:
        for start in range(0, len(jobs), workers * 4):
            for row in pool.map(score, jobs[start : start + workers * 4], chunksize=1):
                db.execute(
                    "insert into candidates values(?,?,?)",
                    (row["candidate_id"], row["site"], json.dumps(row, separators=(",", ":"))),
                )
                completed += 1
            db.commit()
            atomic_json(
                out / "progress.json",
                dict(
                    phase="fit",
                    point=point,
                    completed_candidates=completed,
                    total_candidates=len(expected),
                    seconds=time.time() - started,
                ),
            )
    actual = {r[0] for r in db.execute("select id from candidates")}
    if actual != expected:
        raise ValueError("candidate enumeration not closed")
    sites = []
    for site, counts in site_counts.items():
        records = [
            json.loads(v[0])
            for v in db.execute("select payload from candidates where site=? order by id", (site,))
        ]
        routes = {}
        normal = fall = iteration = 0
        for target, count in counts.items():
            chosen = point_rules.choose(
                [r for r in records if r["condition"] == target], abi.CAP[op]
            )
            if chosen and count:
                k = chosen["K"]
                row = dict(
                    route="CDI_NORMAL",
                    candidate_id=chosen["candidate_id"],
                    source=chosen["source"],
                    public_Ks=k,
                    **{
                        key: chosen[key]
                        for key in ("a_encoded", "b_encoded", "L_encoded", "H_encoded")
                    },
                    selected_evidence=chosen,
                )
                normal += count
                iteration += count * k
            else:
                row = dict(
                    route="SPU_FALLBACK",
                    candidate_id=None,
                    source=None,
                    public_Ks=0,
                    a_encoded=None,
                    b_encoded=None,
                    L_encoded=None,
                    H_encoded=None,
                )
                fall += count
            row.update(numeric_abi=abi.NUMERIC[op], cost_assessment="not_measured")
            routes[target] = row
        sites.append(
            dict(
                operator=op,
                site=site,
                K_scope="public_site_condition",
                public_Ks=None,
                routes=routes,
                normal_events=normal,
                fallback_events=fall,
                iteration_events=iteration,
                condition_event_counts=counts,
            )
        )
    result = dict(
        schema="vit-joint-r3-provisional-numerical-v1",
        point=point,
        operator=op,
        graph=GRAPH_ID,
        population=population,
        input_meta=meta,
        sites=sites,
        arrived_requests=len(rids),
        candidate_count=completed,
        expected_candidates=len(expected),
        immutable_after_selection=True,
        post_selection_refit=False,
        construction_policy=abi.CONSTRUCTION_POLICY,
        cost_assessment="not_measured",
    )
    immutable_json(out / "numerical_selection.json", result)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    atomic_json(
        out / "SCORING_DONE.json",
        dict(
            status="numerical_complete",
            numerical_digest=digest(result),
            candidates=completed,
            scoring_seconds=time.time() - started,
            workers=workers,
        ),
    )
    return finalize_numeric(root, point)


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("immutable numerical output changed")
    else:
        atomic_json(path, value)


def finalize_numeric(root, point):
    folder = Path(root) / "points" / f"{point:02d}"
    doc = json.loads((folder / "numerical_selection.json").read_text())
    scoring = json.loads((folder / "SCORING_DONE.json").read_text())
    if digest(doc) != scoring["numerical_digest"]:
        raise ValueError("scored selection changed")
    selection = abi.admit_numeric(doc)
    immutable_json(folder / "selection.json", selection)
    result = dict(
        status="fit_complete",
        construction_policy=abi.CONSTRUCTION_POLICY,
        point=point,
        selection_digest=digest(selection),
        input_meta=doc["input_meta"],
        candidates=scoring["candidates"],
        scoring_seconds=scoring["scoring_seconds"],
        workers=scoring["workers"],
        numerical_admission="complete",
        cost_assessment="not_measured",
    )
    immutable_json(folder / "DONE.json", result)
    atomic_json(folder / "progress.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--point", type=int, required=True)
    p.add_argument("--workers", type=int, default=28)
    a = p.parse_args()
    try:
        fit(a.root, a.point, a.workers)
    except BaseException as e:
        atomic_json(Path(a.root) / "points" / f"{a.point:02d}" / "FAILED.json", dict(error=repr(e)))
        raise
