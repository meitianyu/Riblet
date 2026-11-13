from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np


_BUILD_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None


def _build_library() -> ctypes.CDLL:
    source = Path(__file__).with_name("kernels.cc")
    if not source.is_file():
        raise RuntimeError(f"CDI f18 kernel source is missing: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / "cdi_site_f18" / digest
    target = root / ("site_f18_kernels.dll" if os.name == "nt" else "site_f18_kernels.so")
    if not target.is_file():
        root.mkdir(parents=True, exist_ok=True)
        compiler = os.environ.get("CXX", "g++")
        base = [
            compiler,
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-shared",
            str(source),
            "-o",
            str(target),
        ]
        attempts = (base + ["-static-libgcc", "-static-libstdc++"],) if os.name == "nt" else (base + ["-fopenmp"], base)
        failures: list[str] = []
        for command in attempts:
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode == 0:
                break
            failures.append(completed.stderr.strip())
        else:
            raise RuntimeError(
                "C++17 CDI f18 kernel build failed: " + " | ".join(failures)
            )
    return ctypes.CDLL(str(target))


def _library() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        with _BUILD_LOCK:
            if _LIBRARY is None:
                library = _build_library()
                library.cdi_site_f18_precision_batch.argtypes = (
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.c_int64,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_uint8),
                )
                library.cdi_site_f18_precision_batch.restype = ctypes.c_int
                _LIBRARY = library
    return _LIBRARY


def precision_pass_batch(
    operands,
    outputs,
    *,
    operator: str,
    bits: int,
    inclusive: bool,
) -> np.ndarray:


    if operator not in {"reciprocal", "rsqrt"}:
        raise ValueError("operator must be reciprocal or rsqrt")
    if bits not in {12, 13}:
        raise ValueError("precision bits must be 12 or 13")
    if not isinstance(inclusive, bool):
        raise TypeError("inclusive must be boolean")
    x = np.asarray(operands)
    y = np.asarray(outputs)
    if x.dtype != np.int64 or y.dtype != np.int64 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("precision operands and outputs must be int64 vectors")
    if len(x) != len(y):
        raise ValueError("precision operands and outputs must be aligned")
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    passed = np.empty(len(x), dtype=np.uint8)
    status = _library().cdi_site_f18_precision_batch(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(x),
        int(operator == "rsqrt"),
        int(bits),
        int(inclusive),
        passed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if status != 0:
        raise RuntimeError("CDI f18 precision kernel rejected its array contract")
    return passed.astype(np.bool_)
