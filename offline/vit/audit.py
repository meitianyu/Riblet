import argparse, hashlib, json
from pathlib import Path
import numpy as np
import abi
from io_utils import GRAPH_ID, digest
from packing import pack_coefficients
from vit_cdi import layout

NUMERIC_ABI = "recip-f22-nearest-f18-K10__rsqrt-f18-original-K5__FM64-v1"


def audit(root):
    root = Path(root)
    fit = root / "fitter"
    required = [root / "CONSTRUCTION_DONE.json", root / "CONSTRUCTION_BINDING.json"]
    required += [root / "outputs" / f"{p:02d}" / "EVIDENCE.json" for p in range(37)]
    required += [fit / "points" / f"{p:02d}" / "DONE.json" for p in range(37)]
    if not all(p.is_file() for p in required):
        raise ValueError("all 37 causal points must complete before packaging")
    binding = json.loads((root / "CONSTRUCTION_BINDING.json").read_text())
    if binding.get("construction_policy") != abi.CONSTRUCTION_POLICY:
        raise ValueError("unknown numerical construction policy")
    prefix = []
    records = []
    summaries = []
    public_points = []
    before_alive = n = binding["requests"]
    totals = {
        op: dict(normal=0, fallback=0, executed=0, theoretical=0, iterations=0) for op in abi.CAP
    }
    master = json.loads((fit / "population.json").read_text())
    raw = (fit / "population.json").read_bytes()
    from riblet.development import image_identity

    population = image_identity(master, custom=binding["profile"] == "custom")
    if (
        population != binding["population"]
        or len(master["rows"]) != n
        or hashlib.sha256(raw).hexdigest() != binding["population_artifact_sha256"]
    ):
        raise ValueError("original population changed")
    for point in range(37):
        evidence = json.loads((root / "outputs" / f"{point:02d}" / "EVIDENCE.json").read_text())
        s = evidence["summary"]
        table = json.loads((fit / "points" / f"{point:02d}" / "selection.json").read_text())
        abi.validate_admitted(table)
        if (
            table["graph"] != GRAPH_ID
            or table["population"] != population
            or table["point"] != point
        ):
            raise ValueError("graph/population")
        if (
            s["prefix_before"] != digest(prefix)
            or s["selection_digest"] != digest(table)
            or table["input_meta"]["prefix"] != digest(prefix)
        ):
            raise ValueError("causal chain")
        if (
            table["candidate_count"] != table["expected_candidates"]
            or s["attempted_requests"] != before_alive
        ):
            raise ValueError("population/candidate closure")
        if (
            s["prepared_finite_requests"] + s["prepare_invalid"] != before_alive
            or s["alive_after"] + s["finish_invalid"] != s["prepared_finite_requests"]
        ):
            raise ValueError("invalid request closure")
        if (
            s["normal_events"] + s["fallback_events"] != s["executed_inverse_events"]
            or s["executed_inverse_events"]
            + s["unreached_inherited_events"]
            + s["invalid_input_unexecuted_events"]
            != s["theoretical_events"]
        ):
            raise ValueError("event closure")
        pre = [i for row in evidence["prepared"] for i in row["prepared_ids"]]
        app = [i for row in evidence["applied"] for i in row["request_ids"]]
        with np.load(fit / "inputs" / f"{point:02d}" / "operands.npz") as f:
            if (
                pre != app
                or pre != f["request_ids"].tolist()
                or f["folds"].tolist() != [master["rows"][i]["fold_id"] for i in pre]
            ):
                raise ValueError("identity/fold projection")
        if s["normal_p12_failures"] or sum(r["normal_p12_failures"] for r in evidence["applied"]):
            raise ValueError("normal precision mismatch")
        for key in ("normal_events", "fallback_events"):
            if (
                sum(r[key] for r in evidence["applied"]) != s[key]
                or sum(site[key] for site in table["sites"]) != s[key]
            ):
                raise ValueError("selected/applied route mismatch")
        for site in table["sites"]:
            for cond, row in site["routes"].items():
                abi.check_row(table["operator"], row)
                if (
                    cond == "CLS"
                    and row["route"] == "CDI_NORMAL"
                    and (table["operator"] == "reciprocal" or row["source"] == "PATCH_PARENT")
                ):
                    raise ValueError("CLS route")
                records.append(
                    dict(
                        index=layout.public_index(table["operator"], site["site"], cond),
                        operator=table["operator"],
                        site=site["site"],
                        condition=cond,
                        **row,
                    )
                )
            totals[table["operator"]]["iterations"] += site["iteration_events"]
        tr = abi.execution_trace(table)
        public_points.append(
            dict(
                point=point,
                operator=table["operator"],
                route=tr["route"],
                K=tr["K"],
                numerical_receipt=abi.schedule_identity(
                    point, table["operator"], tr["route"], tr["K"]
                ),
            )
        )
        t = totals[table["operator"]]
        for field, src in [
            ("normal", "normal_events"),
            ("fallback", "fallback_events"),
            ("executed", "executed_inverse_events"),
            ("theoretical", "theoretical_events"),
        ]:
            t[field] += s[src]
        prefix.append(digest(table))
        before_alive = s["alive_after"]
        summaries.append(s)
    records.sort(key=lambda r: r["index"])
    encoded = pack_coefficients(records)
    for t in totals.values():
        t["coverage_executed"] = t["normal"] / t["executed"] if t["executed"] else None
        t["coverage_original_population"] = t["normal"] / t["theoretical"]
        t["weighted_K_normal"] = t["iterations"] / t["normal"] if t["normal"] else None
    state = evidence["request_state"]
    alive = np.asarray(state["alive"], bool)
    if len(alive) != n or int(alive.sum()) != before_alive:
        raise ValueError("terminal population")
    predictions = sum(sum(v is not None for v in row["predictions"]) for row in evidence["applied"])
    correct = sum(
        sum(
            v is not None and v == master["rows"][i]["class_id"]
            for i, v in zip(row["request_ids"], row["predictions"])
        )
        for row in evidence["applied"]
    )
    report = dict(
        profile=binding["profile"],
        model_files=binding["model_files"],
        population_binding=population,
        construction_origin="local-cpu-construction",
        construction_policy=abi.CONSTRUCTION_POLICY,
        cost_assessment="not_measured",
        status="joint_construction_audited",
        graph_id=GRAPH_ID,
        points=37,
        population=n,
        finite_requests=before_alive,
        numeric_invalid=n - before_alive,
        totals=totals,
        predictions=predictions,
        development_correct=correct,
        normal_p12_failures=0,
        logical_records=len(records),
        encoded_coefficient_slots=len(encoded),
        prefix=digest(prefix),
    )
    out = root / "constructed-joint"
    out.mkdir(exist_ok=False)
    private = dict(graph_id=GRAPH_ID, population=population, records=records, prefix=digest(prefix))
    meta = dict(
        graph_id=GRAPH_ID,
        numeric_abi=NUMERIC_ABI,
        model_revision=(
            binding["model_files"]["flax_model.msgpack"]
            if binding["profile"] == "custom"
            else "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
        ),
        lineage="Joint",
        status="audited_numerical",
        construction_policy=abi.CONSTRUCTION_POLICY,
        cost_assessment="not_measured",
        population_binding=population,
        fold_binding=population,
        points=public_points,
        share_generation="not_performed",
        logical_records=5578,
        coefficient_slots=5434,
    )
    for name, obj in [
        ("joint-private.json", private),
        ("public-deployment-metadata.json", meta),
        ("audit.json", report),
    ]:
        (out / name).write_text(json.dumps(obj, sort_keys=True, allow_nan=False))
    np.save(out / "coefficients-encoded-private.npy", encoded, allow_pickle=False)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    (out / "READY.json").write_text(
        json.dumps(dict(hashes=hashes, status=report["status"], formal_deployment=False))
    )
    for p in out.iterdir():
        p.chmod(0o400 if "private" in p.name else 0o444)
    (root / "AUDIT_COMPLETE.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    audit(Path(p.parse_args().root))
