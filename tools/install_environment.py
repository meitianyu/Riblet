"""Install the two backends in one pinned CPU environment."""

import argparse, hashlib, json, shutil, subprocess, sys, venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_roots(root):
    sources = {family: Path(root).resolve() / family for family in ("bert", "vit")}
    if not all((path / "WORKSPACE").is_file() for path in sources.values()):
        raise ValueError("both complete backend source trees required")
    return sources


def install(prefix, bert=None, vit=None, *, source=None, bazel="bazel", cache=None):
    prefix = Path(prefix).absolute()
    if prefix.exists():
        raise FileExistsError(prefix)
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Python 3.10 required")
    if source and (bert or vit):
        raise ValueError("choose source build or previously built artifacts, not both")
    if not source and not (bert and vit):
        raise ValueError("both source-built artifacts required")
    sources = (
        source_roots(source)
        if source
        else {"bert": Path(bert).resolve(), "vit": Path(vit).resolve()}
    )
    for family, root in sources.items() if not source else []:
        files = json.loads((root / "RUNTIME_FILES.json").read_text())
        if "spu/libspu.so" not in files:
            raise ValueError("source-built backend required")
        for name, expected in files.items():
            path = (root / name).resolve()
            if (
                not path.is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("runtime source integrity: " + family + "/" + name)
    venv.EnvBuilder(with_pip=True).create(prefix)
    python = prefix / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(ROOT / "requirements.lock"),
        ],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    for family, root in sources.items():
        if source:
            argv = [
                str(python),
                str(ROOT / "tools/build_backend.py"),
                "--source",
                str(root),
                "--output",
                str(prefix / "build" / family),
                "--destination",
                str(prefix / "riblet-backends" / family),
                "--bazel",
                bazel,
            ]
            if cache:
                argv += ["--cache", str(cache)]
            subprocess.run(argv, check=True)
        else:
            shutil.copytree(
                root,
                prefix / "riblet-backends" / family,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    site = subprocess.check_output(
        [str(python), "-c", 'import sysconfig;print(sysconfig.get_path("purelib"))'], text=True
    ).strip()
    (Path(site) / "riblet.pth").write_text(str(ROOT / "src") + "\n")
    manifest = dict(
        python_prefix=str(prefix),
        backends={k: "riblet-backends/" + k for k in sources},
        requirements_sha256=hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest(),
        source_root=str(ROOT),
        status="installed",
    )
    (prefix / "installation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--bert-runtime")
    p.add_argument("--vit-runtime")
    p.add_argument("--source-root")
    p.add_argument("--bazel", default="bazel")
    p.add_argument("--cache")
    a = p.parse_args()
    print(
        json.dumps(
            install(
                a.prefix,
                a.bert_runtime,
                a.vit_runtime,
                source=a.source_root,
                bazel=a.bazel,
                cache=a.cache,
            )
        )
    )
