"""Build both inference targets with the selected Python environment."""

import argparse, json, subprocess, sys, time
from pathlib import Path
from install_runtime import install


def command(bazel, output, cache, jobs):
    if jobs < 1:
        raise ValueError("positive build job count required")
    argv = [
        str(bazel),
        "--batch",
        "--output_base=" + str(Path(output).absolute()),
        "build",
        "//spu:libspu.so",
        "//libspu:spu_py_proto",
        "//spu/utils:distributed_py_proto_grpc",
        "-c",
        "opt",
        "--jobs=" + str(jobs),
        "--spawn_strategy=local",
        "--repo_env=PYTHON_BIN_PATH=" + sys.executable,
    ]
    if cache:
        argv.append("--repository_cache=" + str(Path(cache).absolute()))
    return argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--bazel", default="bazel")
    p.add_argument("--cache")
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    output = Path(a.output).absolute()
    source = Path(a.source).resolve()
    if output.exists() or Path(a.destination).exists():
        raise FileExistsError("new build and runtime destinations required")
    if not (source / "WORKSPACE").is_file():
        raise ValueError("complete source snapshot required")
    if subprocess.check_output([a.bazel, "--version"], text=True).strip() != "bazel 6.5.0":
        raise ValueError("Bazel 6.5.0 required")
    output.mkdir(parents=True)
    argv = command(a.bazel, output, a.cache, a.jobs)
    start = time.monotonic()
    with (output / "riblet-build.log").open("x") as log:
        r = subprocess.run(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(
        argv=argv,
        status=r.returncode,
        seconds=time.monotonic() - start,
        source=str(source),
        python=sys.executable,
    )
    (output / "riblet-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if r.returncode:
        raise SystemExit(r.returncode)
    install(source, output, a.destination)


if __name__ == "__main__":
    main()
