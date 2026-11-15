import numpy as np

GRAPH_ID = "ours-vit-native-gelu32f12-exp64f18n6-preln197-v1"
NUMERIC_ABI = "recip-f22-nearest-f18-K10__rsqrt-f18-original-K5__FM64-v1"
MODEL_REVISION = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
SLOTS = 12 * 12 * 36 + 25 * 10


def operator(point):
    if type(point) is not int or not 0 <= point < 37:
        raise ValueError("point")
    return "reciprocal" if point < 36 and point % 3 == 1 else "rsqrt"


def point_shape(point):
    return (1, 12, 197, 1) if operator(point) == "reciprocal" else (1, 197, 1)


def condition_index(op, token):
    if type(token) is not int or not 0 <= token < 197:
        raise ValueError("token")
    if op not in ("reciprocal", "rsqrt"):
        raise ValueError("operator")
    if token == 0:
        return -1 if op == "reciprocal" else 0
    i, j = divmod(token - 1, 14)
    return (
        6 * (6 * i // 14) + 6 * j // 14
        if op == "reciprocal"
        else 1 + 3 * (3 * i // 14) + 3 * j // 14
    )


def slot_indices(point):
    if operator(point) == "reciprocal":
        layer = point // 3
        return np.array(
            [
                ((layer * 12 + h) * 36 + max(0, condition_index("reciprocal", t)))
                for h in range(12)
                for t in range(197)
            ],
            np.int64,
        ).reshape(point_shape(point))
    site = 24 if point == 36 else 2 * (point // 3) + (point % 3 == 2)
    return np.array(
        [5184 + site * 10 + condition_index("rsqrt", t) for t in range(197)], np.int64
    ).reshape(point_shape(point))


def validate_schedule(point, route, k):
    r = np.asarray(route)
    ks = np.asarray(k)
    n = int(np.prod(point_shape(point)))
    if r.size != n or ks.size != n or r.dtype.kind not in "iu" or ks.dtype.kind not in "iu":
        raise ValueError("schedule shape/type")
    r = r.reshape(-1)
    ks = ks.reshape(-1)
    if np.any((r < 0) | (r > 1)) or np.any(
        (ks < 0) | (ks > (10 if operator(point) == "reciprocal" else 5))
    ):
        raise ValueError("route/K")
    if np.any(ks[r == 1] != 0):
        raise ValueError("fallback has no CDI K")
    if operator(point) == "reciprocal" and np.any(r[::197] != 1):
        raise ValueError("CLS reciprocal must fallback")

    ids = slot_indices(point).reshape(-1)
    for slot in np.unique(ids):
        positions = np.flatnonzero(ids == slot)
        if operator(point) == "reciprocal":
            positions = positions[positions % 197 != 0]
        if len(positions) and (
            len(np.unique(r[positions])) != 1 or len(np.unique(ks[positions])) != 1
        ):
            raise ValueError("condition schedule inconsistent")
    return r.reshape(point_shape(point)), ks.reshape(point_shape(point))


MODEL_LOCAL_POLICY = "model-local-numeric-v1"


def schedule_identity(point, op, route, K):
    from .assets import digest

    if op != operator(point):
        raise ValueError("schedule operator mismatch")
    validate_schedule(point, route, K)
    return digest(
        dict(
            policy=MODEL_LOCAL_POLICY,
            point=point,
            operator=op,
            numeric_abi=NUMERIC_ABI,
            route=route,
            K=K,
        )
    )


def validate_package(meta, *, rank, revision=MODEL_REVISION):
    if meta.get("graph_id") != GRAPH_ID:
        raise ValueError("graph mismatch: new causal Joint required")
    if meta.get("numeric_abi") != NUMERIC_ABI or meta.get("model_revision") != revision:
        raise ValueError("numeric/model binding")
    if rank not in (0, 1) or meta.get("share_rank") != rank:
        raise ValueError("share ownership")
    local = meta.get("construction_policy") == MODEL_LOCAL_POLICY
    if meta.get("lineage") != "Joint" or meta.get("OPS", False if local else None) is not False:
        raise ValueError("unsupported ViT parameter schema")
    if "construction_policy" in meta and not local:
        raise ValueError("unknown construction policy")
    if local and meta.get("cost_assessment") != "not_measured":
        raise ValueError("unmeasured local policy")
    if (
        meta.get("status") != ("audited_numerical" if local else "audited_cost_closed")
        or not meta.get("population_binding")
        or not meta.get("fold_binding")
    ):
        raise ValueError("unaudited package")
    points = meta.get("points", [])
    if len(points) != 37:
        raise ValueError("37 points required")
    for point, row in enumerate(points):
        if row.get("point") != point or row.get("operator") != operator(point):
            raise ValueError("topology")
        validate_schedule(point, row["route"], row["K"])
        if local:
            if "cost_receipt" in row or row.get("numerical_receipt") != schedule_identity(
                point, row["operator"], row["route"], row["K"]
            ):
                raise ValueError("numerical schedule identity changed")
        elif not row.get("cost_receipt"):
            raise ValueError("missing frozen parameter receipt")
    return meta


def expand_local_shares(words, meta, *, rank):
    validate_package(meta, rank=rank)
    words = np.asarray(words)
    if words.dtype != np.uint64 or words.shape != (SLOTS, 2):
        raise ValueError("encoded local coefficient share shape")
    return tuple(np.ascontiguousarray(words[slot_indices(p)]) for p in range(37))
