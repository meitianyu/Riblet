"""Assemble the inference-only Python runtime from a completed source build."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def install(source, output_base, destination):
    source, output_base, destination = map(Path, (source, output_base, destination))
    generated = output_base / "execroot/spulib/bazel-out/k8-opt/bin"
    required = (
        "spu/libspu.so",
        "libspu/spu_pb2.py",
        "spu/utils/distributed_pb2.py",
        "spu/utils/distributed_pb2_grpc.py",
    )
    if (
        not all((generated / p).is_file() for p in required)
        or not (source / "spu/__init__.py").is_file()
    ):
        raise ValueError("completed library and generated Python protocol targets required")
    if destination.exists():
        raise FileExistsError(destination)
    init = (source / "spu/__init__.py").read_text()
    if init.count("from . import experimental, psi") != 1 or init.count('    "psi",') != 1:
        raise ValueError("unexpected source initializer; review before packaging")

    init = init.replace("from . import experimental, psi", "from . import experimental").replace(
        '    "psi",\n', ""
    )
    selected = []
    for path in (source / "spu").rglob("*.py"):
        rel = path.relative_to(source)
        if "tests" not in rel.parts and path.name != "psi.py":
            selected.append((path, rel))
    selected.extend((generated / rel, Path(rel)) for rel in required)
    destination.mkdir(parents=True)
    for path, rel in selected:
        out = destination / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
    (destination / "spu/__init__.py").write_text(init, encoding="utf-8")
    shutil.copy2(source / "LICENSE", destination / "SPU_LICENSE")
    hashes = {
        p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    (destination / "RUNTIME_FILES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps({"files": len(install(args.source, args.output_base, args.destination))}))
