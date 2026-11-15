import numpy as np

S = 1 << 18
SCHEMA = "bert-public-site-f18-p12-v1"


def slots():
    return [
        ("reciprocal", f"layer={l}|head={h}", 1 + 3 * l) for l in range(12) for h in range(12)
    ] + [
        ("rsqrt", f"layernorm_site={i}", 0 if i == 0 else 2 + (i - 1) // 2 * 3 + (i - 1) % 2)
        for i in range(25)
    ]


def limbs(words):
    # Two exact F32 inputs avoid rounding a large f18 integer before SPU infeed.
    w = np.asarray(words, np.int64)
    hi = w // 65536
    lo = w % 65536
    x = hi.astype(np.float32) * np.float32(0.25)
    y = lo.astype(np.float32) / np.float32(S)
    if not np.array_equal(
        (x.astype(np.float64) * S).astype(np.int64) + (y.astype(np.float64) * S).astype(np.int64), w
    ):
        raise ValueError("inexact coefficient limbs")
    return x, y


def validate(rows):
    expected = slots()
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("expected 169 ordered BERT sites")
    for row, (operator, site, point) in zip(rows, expected):
        if (
            set(row) != {"operator", "site", "point", "enabled", "K"}
            or (row["operator"], row["site"], row["point"]) != (operator, site, point)
            or type(row["point"]) is not int
            or type(row["enabled"]) is not bool
        ):
            raise ValueError("invalid BERT site schedule")
        k = row["K"]
        if (row["enabled"] and (type(k) is not int or not 0 <= k <= 8)) or (
            not row["enabled"] and k is not None
        ):
            raise ValueError("invalid iteration count for BERT site")
    return rows


def select(document, task):
    if (document["schema"], document["layout"]) != (SCHEMA, "pad128"):
        raise ValueError("expected a public-site pad128 table")
    by = {r["site"]: r for r in document["sites"] if r["task"] == task}
    if len(by) != 169 or sum(r["task"] == task for r in document["sites"]) != 169:
        raise ValueError("incomplete or duplicate BERT sites")
    rows = [by[site] for _, site, _ in slots()]
    public = validate(
        [{k: r[k] for k in ("operator", "site", "point", "enabled", "K")} for r in rows]
    )
    return public, rows


def coefficient_limbs(records):
    words = np.asarray(
        [(r["a_encoded"], r["b_encoded"]) if r["enabled"] else (0, S) for r in records], np.int64
    )
    return limbs(words)


def runtime_arguments(jnp, rows, hi, lo, n):
    enabled = np.asarray([r["enabled"] for r in rows], bool)
    active = np.broadcast_to(enabled, (n, 169)).copy()
    active[0, :144] = False
    routes = jnp.asarray((~active).astype(np.int64))
    ab = jnp.broadcast_to((hi + lo)[None, :, :], (n, 169, 2))
    return routes[:, :144], ab[:, :144], routes[:, 144:], ab[:, 144:]
