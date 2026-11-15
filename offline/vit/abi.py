GRAPH_ID = "ours-vit-native-gelu32f12-exp64f18n6-preln197-v1"
CAP = {"reciprocal": 10, "rsqrt": 5}
NUMERIC = {
    "reciprocal": "fm64_f18_coeff_lift22_newton_nearest18",
    "rsqrt": "fm64_f18_original_x_yy_half",
}


def scope(operator, site, fold):
    if (
        operator not in CAP
        or not 0 <= site < (144 if operator == "reciprocal" else 25)
        or not 0 <= fold <= 5
    ):
        raise ValueError("public tape scope")
    suffix = "full" if fold == 5 else str(fold)
    return f"VIT_RNG_AUDIT_V1:common:trial=0:{operator}:site={site}:fold={suffix}"


def check_row(operator, row):
    if operator not in CAP or row.get("numeric_abi") != NUMERIC[operator]:
        raise ValueError("wrong inverse numerical ABI")
    k = row["public_Ks"]
    if type(k) is not int or not 0 <= k <= CAP[operator]:
        raise ValueError("wrong iteration cap")
    if row["route"] == "SPU_FALLBACK":
        if k != 0 or any(
            row.get(key) is not None for key in ("a_encoded", "b_encoded", "L_encoded", "H_encoded")
        ):
            raise ValueError("noncanonical fallback")
    elif row["route"] == "CDI_NORMAL":
        if any(
            type(row.get(key)) is not int
            for key in ("a_encoded", "b_encoded", "L_encoded", "H_encoded")
        ):
            raise ValueError("missing encoded coefficient/domain")
    else:
        raise ValueError("unknown route")


from riblet.vit_parameters import (
    MODEL_LOCAL_POLICY as CONSTRUCTION_POLICY,
    schedule_identity as schedule_identity,
)

import copy
from vit_cdi import layout
from riblet.assets import digest


def execution_trace(doc):
    op = doc["operator"]
    point = doc["point"]
    if doc["graph"] != GRAPH_ID or (op == "reciprocal") != (point % 3 == 1):
        raise ValueError("cost graph/point mismatch")
    sites = sorted(doc["sites"], key=lambda s: s["site"])
    expected = (
        list(range(point // 3 * 12, point // 3 * 12 + 12))
        if op == "reciprocal"
        else [24 if point == 36 else 2 * (point // 3) + int(point % 3 == 2)]
    )
    if [s["site"] for s in sites] != expected:
        raise ValueError("incomplete public call")
    routes = []
    ks = []
    for site in sites:
        if set(site["routes"]) != set(layout.conditions(op)):
            raise ValueError("missing condition")
        for row in site["routes"].values():
            check_row(op, row)
        for pos in range(197):
            row = site["routes"][layout.condition(op, pos)]
            if op == "reciprocal" and pos == 0 and row["route"] != "SPU_FALLBACK":
                raise ValueError("CLS must fallback")
            routes.append(int(row["route"] != "CDI_NORMAL"))
            ks.append(row["public_Ks"])
    return dict(
        operator=op, point=point, numeric_abi=NUMERIC[op], n=len(routes), route=routes, K=ks
    )


def validate_selection(doc):
    execution_trace(doc)
    for site in doc["sites"]:
        for row in site["routes"].values():
            if row["route"] != "CDI_NORMAL":
                continue
            e = row["selected_evidence"]
            k = row["public_Ks"]
            if not e["qualified"] or k not in e["feasible_Ks"] or len(e["scores"]) != 6:
                raise ValueError("unqualified numeric candidate")
            if not all(
                s["mask"] & (1 << k) and s["visited"] == s["scored_events"] for s in e["scores"]
            ):
                raise ValueError("incomplete sixway replay")
            if any(
                row[key] != e[key] for key in ("a_encoded", "b_encoded", "L_encoded", "H_encoded")
            ):
                raise ValueError("coefficients changed after replay")


def admit_numeric(doc):
    if (
        any(
            k in doc
            for k in ("cost_admission", "cost_closed", "native_cost_confirmed", "cost_policy")
        )
        or doc.get("cost_assessment", "not_measured") != "not_measured"
    ):
        raise ValueError("local numerical admission cannot certify measured performance")
    if doc.get("construction_policy") != CONSTRUCTION_POLICY:
        raise ValueError("unknown numerical construction policy")
    if doc["candidate_count"] != doc["expected_candidates"]:
        raise ValueError("candidate enumeration incomplete")
    meta = doc["input_meta"]
    if (meta["point"], meta["population"], meta["graph"]) != (
        doc["point"],
        doc["population"],
        doc["graph"],
    ):
        raise ValueError("numerical selection input binding differs")
    validate_selection(doc)
    result = copy.deepcopy(doc)
    result["schema"] = "vit-model-local-numeric-point-v1"
    result["cost_assessment"] = "not_measured"
    result["numeric_admission"] = digest(result)
    return result


def validate_admitted(doc):
    body = {k: v for k, v in doc.items() if k != "numeric_admission"}
    if doc.get("numeric_admission") != digest(body) or doc.get("cost_assessment") != "not_measured":
        raise ValueError("numerical selection changed after admission")
    if admit_numeric({k: v for k, v in body.items() if k not in ("cost_assessment",)}) != doc:
        raise ValueError("invalid numerical admission")
