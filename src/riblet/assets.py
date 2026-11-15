import hashlib, json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets"


def digest(value):

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def architecture_id(config):

    return digest(
        {k: v for k, v in config.items() if k not in ("_name_or_path", "transformers_version")}
    )


def verify_model(model, folder, *, bound=None, custom_weights=False, preprocessor=None):


    paper = json.loads((ASSETS / "model_identities.json").read_bytes())["models"]
    if model not in paper:
        raise ValueError("unsupported model family")
    folder = Path(folder)
    processor = folder if preprocessor is None else Path(preprocessor)
    paths = {
        name: (folder if name in ("config.json", "flax_model.msgpack") else processor) / name
        for name in paper[model]
    }
    if not all(p.is_file() for p in paths.values()):
        raise ValueError("incomplete checkpoint")
    files = {name: file_hash(p) for name, p in paths.items()}
    if custom_weights or (bound is not None and bound != paper[model]):
        signatures = json.loads((ASSETS / "architecture_signatures.json").read_bytes())
        if architecture_id(json.loads((folder / "config.json").read_bytes())) != signatures[model]:
            raise ValueError("model configuration differs from the supported forward")
    if not custom_weights and files != (paper[model] if bound is None else bound):
        raise ValueError("model artifacts differ from the bound checkpoint")
    return files


def verify_preprocessor(model, folder, *, bound=None):
    paper = json.loads((ASSETS / "model_identities.json").read_bytes())["models"]
    if model not in paper:
        raise ValueError("unsupported model preprocessor")
    names = (
        ("tokenizer.json", "tokenizer_config.json")
        if model.startswith("bert-")
        else ("preprocessor_config.json",)
    )
    files = {name: file_hash(Path(folder) / name) for name in names}
    expected = paper[model] if bound is None else bound
    if any(files[name] != expected[name] for name in names):
        raise ValueError("preprocessor content differs from the bound checkpoint")
    if model.startswith("bert-") and files["tokenizer.json"] != paper[model]["tokenizer.json"]:
        from .preprocessing import load_tokenizer

        tok = load_tokenizer(Path(folder) / "tokenizer.json")
        if tok.token_to_id("[PAD]") != 0 or tok.get_vocab_size() > 30522:
            raise ValueError("custom tokenizer exceeds the supported vocabulary/pad ABI")
    return files
