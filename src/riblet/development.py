from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, tempfile
from .preprocessing import tokenize_bert, load_tokenizer
from .assets import verify_preprocessor

FIELDS = (
    "request_ordinal",
    "request_digest",
    "task",
    "fold_id",
    "effective_length",
    "source_role",
    "input_ids",
    "token_type_ids",
)
MEMBERSHIP_SHA = "3d6d58fca85357225f7cdcf4b7f42c657c934a5a52f5b8bd763a485ff730cc7a"


def content_hash(member, n, ids, segments):
    content = {
        "schema": "cdi-role6-canonical-input-v1",
        "tokenizer_sha256": member["tokenizer_sha256"],
        "effective_length": n,
        "input_ids": ids,
        "token_type_ids": segments,
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rebuild_record(member, source_row, tokenizer):
    if str(source_row["idx"]) != member["example_id"]:
        raise ValueError("raw source ID differs from frozen membership")
    tokens = tokenize_bert(member["task"], source_row, tokenizer, pad_id=0)
    n = tokens["effective_length"]
    ids = tokens["input_ids"][0, :n].tolist()
    segments = tokens["token_type_ids"][0, :n].tolist()
    if content_hash(member, n, ids, segments) != member["content_sha256"]:
        raise ValueError("tokenized source content differs; do not select a replacement")
    return dict(
        member,
        input_ids=ids,
        token_type_ids=segments,
        effective_length=n,
        label_json=json.dumps(source_row.get("label"), separators=(",", ":")),
    )


def prepare(membership, arrow_root, preprocessors, output):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    members = json.loads(Path(membership).read_text())
    if (
        members.get("schema") != "bert-public-site-membership-v2"
        or hashlib.sha256(Path(membership).read_bytes()).hexdigest() != MEMBERSHIP_SHA
    ):
        raise ValueError("frozen Development membership required")
    records = []
    by_task = {}
    for row in members["rows"]:
        by_task.setdefault(row["task"], []).append(row)
    if len(members["rows"]) != 50000 or set(by_task) != {
        "cola",
        "mnli",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "stsb",
    }:
        raise ValueError("incomplete eight-task Development population")
    for task, selected in by_task.items():
        folder = Path(preprocessors) / task
        verify_preprocessor("bert-" + task, folder)
        tok = load_tokenizer(folder / "tokenizer.json")
        tokenizer_hash = hashlib.sha256((folder / "tokenizer.json").read_bytes()).hexdigest()
        if any(r["tokenizer_sha256"] != tokenizer_hash for r in selected):
            raise ValueError("frozen tokenizer identity mismatch")
        with (Path(arrow_root) / task / "glue-train.arrow").open("rb") as stream:
            table = ipc.open_stream(stream).read_all()
        indices = [r["source_row_index"] for r in selected]
        if any(type(i) is not int or not 0 <= i < table.num_rows for i in indices):
            raise ValueError("original row index outside fixed training data")
        raw = table.take(indices).to_pylist()
        records.extend(rebuild_record(r, s, tok) for r, s in zip(selected, raw, strict=True))
    records.sort(key=lambda r: r["request_ordinal"])
    canonical = json.dumps(
        [{k: r[k] for k in FIELDS} for r in records],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    output = Path(output)
    if output.exists():
        raise FileExistsError("population output is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), output)
    return {
        "requests": len(records),
        "population_semantic_sha256": hashlib.sha256(canonical).hexdigest(),
        "resampled": False,
    }


VIT_ID = "bd92c3647dca809a335099645bf72d4985eb97c9512995b2608ff32df13f0514"


def validate_vit_membership(document):
    rows = document.get("rows", [])
    fields = ("request_ordinal", "relative_id", "fold_id", "class_id", "sha256")
    canonical = json.dumps(
        [{k: r[k] for k in fields} for r in rows], sort_keys=True, separators=(",", ":")
    ).encode()
    if len(rows) != 10000 or hashlib.sha256(canonical).hexdigest() != VIT_ID:
        raise ValueError("frozen image membership/folds changed")
    return rows


def copy_image(source_root, destination_root, row):
    name = row["relative_id"]
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or ".." in p.parts
        or "\\" in name
        or ":" in name
        or not p.parts
        or str(p) != name
    ):
        raise ValueError("unsafe image identity")
    root = Path(source_root).resolve()
    source = root / name
    if source.is_symlink() or not source.resolve().is_relative_to(root):
        raise ValueError("image escapes owned data root")
    data = source.read_bytes()
    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("image content changed")
    target = Path(destination_root) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)


def prepare_construction_inputs(
    family, membership, raw_dataset_root, output, *, preprocessors=None
):
    membership = Path(membership)
    output = Path(output)
    registry = json.loads(
        (Path(__file__).resolve().parents[2] / "assets/resources.json").read_text()
    )
    entries = [r for r in registry["resources"] if r["resource_id"] == "construction-" + family]
    if len(entries) != 1:
        raise ValueError("unsupported construction family")
    member = entries[0]["members"][0]
    if hashlib.sha256(membership.read_bytes()).hexdigest() != member["sha256"]:
        raise ValueError("membership is not the bound resource")
    if output.exists():
        raise FileExistsError("construction inputs are immutable")
    if family == "bert":
        if not preprocessors:
            raise ValueError("eight frozen task preprocessors required")
        return prepare(membership, raw_dataset_root, preprocessors, output)
    doc = json.loads(membership.read_text())
    rows = validate_vit_membership(doc)
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(r["bytes"] for r in rows)
    if shutil.disk_usage(output.parent).free < required:
        raise OSError("insufficient image staging space")
    with tempfile.TemporaryDirectory(prefix=".riblet-inputs-", dir=output.parent) as tmp:
        stage = Path(tmp) / "data"
        stage.mkdir()
        for row in rows:
            copy_image(Path(raw_dataset_root) / "images", stage / "images", row)
        shutil.copyfile(membership, stage / "manifest.json")
        receipt = dict(
            status="construction_inputs_verified",
            requests=10000,
            population_semantic_sha256=VIT_ID,
            resampled=False,
            model_forwards=0,
            coefficients_loaded=False,
        )
        (stage / "INPUTS_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.rename(stage, output)
    return receipt


def select_rows(rows, key, count, seed, namespace):

    if type(count) is not int or not 1 <= count <= len(rows) or type(seed) is not int:
        raise ValueError("invalid construction sample size or seed")
    ids = [str(row[key]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate construction source ID")

    def order(row):
        identity = json.dumps([namespace, seed, str(row[key])], separators=(",", ":")).encode()
        return hashlib.sha256(identity).digest(), str(row[key])

    return sorted(rows, key=order)[:count]


def image_identity(doc, *, custom=False):
    if not custom:
        validate_vit_membership(doc)

        return "ed57ac69573ed401f562bd72c70ca95cdf2d29f274192ee9997eba9405fb079d"
    from .assets import digest

    rows = doc["rows"]
    if (
        doc.get("profile") != "custom"
        or not rows
        or any(
            r["request_ordinal"] != i
            or not 0 <= r["class_id"] < 1000
            or r["fold_id"] not in range(5)
            for i, r in enumerate(rows)
        )
        or len({r["relative_id"] for r in rows}) != len(rows)
        or digest({k: v for k, v in doc.items() if k != "population_id"})
        != doc.get("population_id")
    ):
        raise ValueError("custom image manifest or dense ordinal changed")
    return doc["population_id"]


def task_counts(total):

    weights = {
        "cola": 5873,
        "mnli": 9398,
        "mrpc": 1779,
        "qnli": 9398,
        "qqp": 9398,
        "rte": 1164,
        "sst2": 9398,
        "stsb": 3592,
    }
    if type(total) is not int or total <= 0:
        raise ValueError("positive total required")
    n = sum(weights.values())
    result = {t: total * v // n for t, v in weights.items()}
    for t in sorted(weights, key=lambda t: (-(total * weights[t] % n), t))[
        : total - sum(result.values())
    ]:
        result[t] += 1
    if not all(result.values()):
        raise ValueError("count is too small to represent all eight tasks")
    return result


def custom_inputs(cfg, destination):

    from .assets import digest

    source = cfg["construction"]
    root = cfg["directory"] / source["data_root"]
    selection = source["selection"]
    version = source["dataset_version"]
    if not version or source.get("split", "train") != "train":
        raise ValueError("a versioned training population is required")
    family = cfg["family"]
    task = cfg["model"].removeprefix("bert-")
    if selection["mode"] == "manifest":

        ids = json.loads((cfg["directory"] / selection["path"]).read_text())
        if (
            not ids
            or len({str(r["id"]) for r in ids}) != len(ids)
            or any(type(r["fold"]) is not int or r["fold"] not in range(5) for r in ids)
        ):
            raise ValueError("invalid custom membership/folds")
    elif selection["mode"] != "sample":
        raise ValueError("custom selection must be sample or manifest")

    def choose(rows, key, selected_task=task, count=None):
        if selection["mode"] == "manifest":
            lookup = {str(r[key]): r for r in rows}
            if len(lookup) != len(rows):
                raise ValueError("duplicate source ID")
            return [(lookup[str(r["id"])], r["fold"]) for r in ids]
        if count is None:
            count = selection["count"]
        if isinstance(count, dict):
            count = count[selected_task]
        chosen = select_rows(rows, key, count, selection["seed"], version + "/" + selected_task)
        return [(r, i % 5) for i, r in enumerate(chosen)]

    if family == "bert":
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq

        def task_records(task, processor, count, ordinal_start):
            with (root / task / "glue-train.arrow").open("rb") as stream:
                table = ipc.open_stream(stream).read_all()
            refs = [{"idx": idx, "row": i} for i, idx in enumerate(table["idx"].to_pylist())]
            selected = choose(refs, "idx", task, count)
            tok = load_tokenizer(processor / "tokenizer.json")
            records = []
            for ordinal, ((ref, fold), raw) in enumerate(
                zip(selected, table.take([r["row"] for r, _ in selected]).to_pylist(), strict=True)
            ):
                v = tokenize_bert(task, raw, tok, pad_id=0)
                n = v["effective_length"]
                records.append(
                    dict(
                        request_ordinal=ordinal_start + ordinal,
                        task=task,
                        example_id=str(ref["idx"]),
                        request_digest=digest([version, task, ref["idx"]]),
                        source_role="Development",
                        fold_id=fold,
                        effective_length=n,
                        input_ids=v["input_ids"][0, :n].tolist(),
                        token_type_ids=v["token_type_ids"][0, :n].tolist(),
                        label_json=json.dumps(raw.get("label")),
                    )
                )
            return records

        if selection.get("scope") == "all-paper-tasks":
            models = cfg["directory"] / source["models_root"]
            records = []
            for t, count in task_counts(selection["count"]).items():
                verify_preprocessor("bert-" + t, models / t)
                records.extend(task_records(t, models / t, count, len(records)))
        else:
            records = task_records(task, cfg["processor"], selection.get("count"), 0)
        pq.write_table(pa.Table.from_pylist(records), destination)
    else:

        candidates = json.loads((cfg["directory"] / source["candidates"]).read_text())
        rows = []
        for i, (r, fold) in enumerate(choose(candidates, "relative_id")):
            if (
                r["split"] != "train"
                or type(r["class_id"]) is not int
                or not 0 <= r["class_id"] < 1000
            ):
                raise ValueError("expected original training class IDs")
            row = dict(r, request_ordinal=i, fold_id=fold)
            copy_image(root / "images", destination / "images", row)
            rows.append(row)
        doc = dict(profile="custom", dataset_version=version, selection=selection, rows=rows)
        doc["population_id"] = digest(doc)
        image_identity(doc, custom=True)
        (destination / "manifest.json").write_text(json.dumps(doc))
