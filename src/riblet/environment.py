import json, sys
from pathlib import Path
from .assets import file_hash


def resolve(config, family):
    if family not in ("bert", "vit"):
        raise ValueError("unknown model family")
    if "installation" not in config:
        raise ValueError("shared installation manifest is required")
    manifest = Path(config["installation"]).resolve()
    installed = json.loads(manifest.read_text())
    if Path(installed["python_prefix"]).resolve() != Path(sys.prefix).resolve():
        raise ValueError("run with the installed shared Python environment")
    root = (manifest.parent / installed["backends"][family]).resolve()
    files = json.loads((root / "RUNTIME_FILES.json").read_text())
    if not files:
        raise ValueError("empty backend manifest")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("invalid runtime member")
        if file_hash(path) != expected:
            raise ValueError("runtime content changed: " + name)
    return root


def activate(config, family):
    root = resolve(config, family)
    if "spu" in sys.modules and not Path(sys.modules["spu"].__file__).resolve().is_relative_to(
        root
    ):
        raise ValueError("select another model in a fresh process of the same environment")
    sys.path.insert(0, str(root))
    return root
