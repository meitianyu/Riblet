import hashlib
import numpy as np

SCHEMA = "bert-public-site-f18-p12-v1"


def site(point, head):
    if point % 3 == 1:
        return f"layer={(point-1)//3}|head={head}"
    return f"layernorm_site={0 if point==0 else 1+2*((point-1)//3)+int((point-1)%3==2)}"


def pool_mask(lengths, recip):
    lengths = np.asarray(lengths, dtype=np.int64)
    out = np.ones(int(lengths.sum()), bool)
    if recip and lengths.size:
        out[np.r_[0, np.cumsum(lengths)[:-1]]] = False
    return out


def project(records, point, ordinal, n, size):
    heads = 12 if point % 3 == 1 else 1
    a = np.zeros((heads, size), np.int64)
    b = np.full_like(a, 1 << 18)
    k = np.zeros_like(a)
    enabled = np.zeros_like(a, dtype=bool)
    seeds = np.zeros_like(a, dtype=np.uint64)
    assert len(records) == heads
    for h, r in enumerate(records):
        assert r["site"] == site(point, h)
        if not r["enabled"]:
            continue
        first = 1 if heads == 12 else 0
        sl = slice(first, n)
        a[h, sl] = r["a_encoded"]
        b[h, sl] = r["b_encoded"]
        k[h, sl] = r["K"]
        enabled[h, sl] = True
        prefix = ("CDI_G9_V2_DESIGN_0\0" + r["seed_scope"] + ":fold=None\0").encode()
        base = (ordinal << 17) | (point << 11) | (h << 7)
        for pos in range(first, n):
            seeds[h, pos] = int.from_bytes(
                hashlib.sha256(prefix + str(base | pos).encode()).digest()[:8], "little"
            )
    return {key: v.ravel() for key, v in dict(a=a, b=b, k=k, enabled=enabled, seeds=seeds).items()}


def write_sites(path, document):
    import os, json

    temp = path.with_suffix(".pending")
    with temp.open("wb") as stream:
        stream.write(
            json.dumps(document, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
