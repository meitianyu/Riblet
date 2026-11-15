import hashlib
import json
import math


def linear_fingerprints(manifest):
    families = manifest["physical_families"]
    if len(families) != 75:
        raise ValueError("BERT physical family count changed")
    dense = [f for f in families if f["kind"] in ("qkv_direct_sum", "other_dense")]
    embedding = [f for f in families if f["kind"] == "embedding_batch"]
    attention = [f for f in families if f["kind"] == "attention_matmul"]
    if (len(dense), len(embedding), len(attention)) != (50, 1, 24):
        raise ValueError("incomplete BERT families")
    ids = [x for f in families for x in f["logical_consumer_ids"]]
    if len(ids) != len(set(ids)) or len(ids) != manifest["logical_consumer_count"]:
        raise ValueError("logical consumers mismatch")
    by_consumer = {x: f for f in dense for x in f["logical_consumer_ids"]}
    ordered = [embedding[0]]
    for layer in range(12):
        root = f"bert/encoder/layer/{layer}"
        ordered.extend(
            [
                by_consumer[root + "/attention/self/query"],
                attention[2 * layer],
                attention[2 * layer + 1],
                by_consumer[root + "/attention/output/dense"],
                by_consumer[root + "/intermediate/dense"],
                by_consumer[root + "/output/dense"],
            ]
        )
    ordered.extend([by_consumer["bert/pooler/dense"], by_consumer["classifier"]])
    if len({f["physical_family_id"] for f in ordered}) != 75:
        raise ValueError("physical family mapping duplicates")
    result = []
    for family in ordered:
        lhs = list(family["lhs_shape"])
        rhs = list(family["rhs_shape"])
        kind = (
            3
            if family["kind"] == "attention_matmul"
            else 2 if family["kind"] == "embedding_batch" else 1
        )
        if kind != 3:
            lhs = [math.prod(lhs[:-1]), lhs[-1]]
        if lhs[-1] != rhs[-2] or (kind == 3 and lhs[0] != rhs[0]):
            raise ValueError("matrix shape mismatch")
        result.append([kind, lhs, rhs, lhs[:-1] + [rhs[-1]]])
    return result


def geometry_sha(manifest):
    return hashlib.sha256(
        json.dumps(linear_fingerprints(manifest), separators=(",", ":")).encode()
    ).hexdigest()


def fingerprint_text(manifest):
    return ";".join(
        ",".join(map(str, [kind, len(lhs), *lhs, *rhs, *out]))
        for kind, lhs, rhs, out in linear_fingerprints(manifest)
    )
