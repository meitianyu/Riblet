from pathlib import Path
import json
import numpy as np
from .transport import require_owner
from .config import model_spec
from .assets import verify_preprocessor
from .preprocessing import load_tokenizer, tokenize_bert


def bert_runtime_inputs(values):

    return {
        k: np.asarray(values[k], np.int64)
        for k in ("input_ids", "attention_mask", "token_type_ids")
    }


def load_input(rank, model, path, preprocessor, *, bound=None):
    require_owner(rank, "input")
    spec = model_spec(model)
    path = Path(path)
    preprocessor = Path(preprocessor)
    verify_preprocessor(model, preprocessor, bound=bound)
    if spec["family"] == "bert":
        if path.suffix == ".json":
            tokenizer = load_tokenizer(preprocessor / "tokenizer.json")
            values = tokenize_bert(
                spec["task"], json.loads(path.read_bytes()), tokenizer, pad_id=0
            )
        else:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as f:
                    values = {
                        k: np.asarray(f[k])
                        for k in ("input_ids", "attention_mask", "token_type_ids")
                    }
            elif path.suffix == ".npy":
                raw = np.load(path, allow_pickle=False)
                if raw.shape != (3, 1, 128):
                    raise ValueError("BERT stack requires IDs/mask/segments at [3,1,128]")
                values = dict(zip(("input_ids", "attention_mask", "token_type_ids"), raw))
            else:
                raise ValueError("BERT accepts a task-text JSON or prepared NPY/NPZ")
            for key, value in values.items():
                if value.shape != (1, 128) or value.dtype.kind not in "iu":
                    raise ValueError("invalid BERT tensor shape/type")
            ids = values["input_ids"]
            mask = values["attention_mask"]
            segments = values["token_type_ids"]
            n = int(mask.sum())
            if not 1 <= n <= 128 or not np.array_equal(
                mask[0], (np.arange(128) < n).astype(mask.dtype)
            ):
                raise ValueError("right-padded nonempty mask required")
            if (
                np.any(ids < 0)
                or np.any(ids >= 30522)
                or np.any(ids[0, n:] != 0)
                or np.any((segments < 0) | (segments > 1))
            ):
                raise ValueError("token/segment values outside supported layout")
            values = {k: np.asarray(v, np.int32) for k, v in values.items()}
            values["effective_length"] = n
        return bert_runtime_inputs(values), dict(
            effective_length=values["effective_length"], shape=[1, 128], mpc_input_dtype="int64"
        )
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        from PIL import Image
        from transformers import ViTImageProcessor

        processor = ViTImageProcessor.from_pretrained(str(preprocessor), local_files_only=True)
        with Image.open(path) as im:
            value = processor(images=im.convert("RGB"), return_tensors="np")["pixel_values"]
    value = np.asarray(value)
    if value.shape != (1, 3, 224, 224) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("ViT input must be finite float32 [1,3,224,224]")
    return {"image": value}, dict(shape=list(value.shape), tokens=197)


def save_output(rank, model, output, directory):
    require_owner(rank, "output")
    spec = model_spec(model)
    value = np.asarray(output)
    if list(value.shape) != spec["output_shape"]:
        raise ValueError("model output shape differs from the declared task")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "output.npy", value, allow_pickle=False)
    finite = bool(np.isfinite(value).all())
    record = dict(
        output_kind=spec["output_kind"],
        shape=list(value.shape),
        finite=finite,
        source="actual_secure_output",
        plaintext_reexecution=False,
    )
    if finite:
        if spec["output_kind"] == "regression":
            record["value"] = float(value[0, 0])
        else:
            record["prediction"] = int(value.argmax(axis=-1)[0])
    (directory / "output.json").write_text(json.dumps(record, indent=2))
    return record
