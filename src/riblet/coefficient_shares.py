import secrets
import numpy as np


def split(encoded, *, owner_rank):
    table = np.asarray(encoded)
    if owner_rank != 0:
        raise ValueError("both models use model party rank0")
    if (
        table.dtype != np.dtype(np.uint64)
        or table.ndim != 2
        or table.shape[1] != 2
        or not table.size
    ):
        raise ValueError("encoded uint64 affine coefficient records required")
    model_share = (
        np.frombuffer(secrets.token_bytes(table.nbytes), dtype=np.uint64)
        .copy()
        .reshape(table.shape)
    )
    user_share = table - model_share
    return model_share, user_share
