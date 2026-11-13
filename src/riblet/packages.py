import json, secrets, shutil, tempfile
from pathlib import Path
import numpy as np
from .config import model_spec
from .assets import digest, file_hash
from . import bert_tables, vit_parameters

BERT_LINEAGE = "public-site-pad128"
BERT_GRAPH = "BERT_PUBLIC_SITE_pad128_NATIVE_INVERSE18"
BERT_ABI = "FM64_F18_CHEETAH_PROBTRUNC_Y_G9"


def _asset(name):
    return json.loads((Path(__file__).resolve().parents[2] / "assets" / name).read_text())


def _base(model, kind):
    spec = model_spec(model)
    return dict(
        schema="riblet-parameter-package-v1",
        method="riblet-full",
        model=model,
        kind=kind,
        model_rank=0,
        data_rank=1,
        layout=spec["layout"],
        selected_K_only=True,
        model_files=_asset("model_identities.json")["models"][model],
    )


def _public_descriptor(owner, deployment_id):
    fields = (
        "method",
        "model",
        "kind",
        "profile",
        "model_rank",
        "data_rank",
        "layout",
        "selected_K_only",
        "graph_id",
        "numeric_abi",
        "lineage",
        "coefficient_slots",
    )
    public = {k: owner[k] for k in fields if k in owner}
    public.update(schema="riblet-parameter-package-v3", deployment_id=deployment_id)
    model = owner["model"]
    names = (
        ("tokenizer.json", "tokenizer_config.json")
        if model.startswith("bert-")
        else ("preprocessor_config.json",)
    )
    public["preprocessor_files"] = {k: owner["model_files"][k] for k in names}
    if model.startswith("bert-"):
        public["schedule"] = bert_tables.validate(owner["schedule"])
    else:
        public["schedule"] = [
            {k: row[k] for k in ("point", "operator", "route", "K")} for row in owner["schedule"]
        ]
    public["package_id"] = digest(public)
    return public


def graph_binding(public):
    return {k: public[k] for k in ("package_id", "deployment_id")}


def require_graph_binding(graph, public):
    if graph_binding(graph) != graph_binding(public) or graph["model"] != public["model"]:
        raise ValueError("public graph/parameters differ")


def _write(output, public, files):
    output = Path(output)
    if output.exists():
        raise FileExistsError("parameter package is immutable")
    owner = dict(public, private_hashes={name: file_hash(path) for name, path in files.items()})
    public = _public_descriptor(owner, secrets.token_hex(32))
    owner.update(graph_binding(public))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".riblet-package-", dir=output.parent) as temporary:
        stage = Path(temporary) / "package"
        (stage / "private").mkdir(parents=True, mode=0o700)
        (stage / "public").mkdir()
        for name, path in files.items():
            target = stage / "private" / name
            shutil.copy2(path, target)
            target.chmod(0o600)
            if file_hash(target) != owner["private_hashes"][name]:
                raise ValueError("parameter source changed during copy")
        manifest = stage / "private/manifest.json"
        manifest.write_text(json.dumps(owner, indent=2) + "\n")
        manifest.chmod(0o600)
        (stage / "public/parameters.json").write_text(json.dumps(public, indent=2) + "\n")
        stage.rename(output)
    return public


def _bert_public(source, model, kind):
    source = Path(source)
    private = json.loads((source / "private-sites.json").read_bytes())
    schedule = json.loads((source / "public-schedule.json").read_bytes())
    task = model_spec(model)["task"]
    pub, _ = bert_tables.select(schedule, task)
    other, _ = bert_tables.select(private, task)
    if pub != other:
        raise ValueError("public and owner schedules differ")
    return dict(
        _base(model, kind),
        graph_id=BERT_GRAPH,
        numeric_abi=BERT_ABI,
        lineage=BERT_LINEAGE,
        schedule=pub,
    )


def pack_frozen_bert(source, model, output):
    source = Path(source)
    expected = _asset("parameter_identities.json")["bert"]
    files = {k: source / k for k in expected}
    if any(file_hash(f) != expected[k] for k, f in files.items()):
        raise ValueError("incorrect frozen BERT package")
    return _write(output, _bert_public(source, model, "frozen"), files)


def pack_frozen_vit(source, output):
    source = Path(source)
    expected = _asset("parameter_identities.json")["vit"]
    for name, wanted in expected.items():
        if file_hash(source / name) != wanted:
            raise ValueError("frozen ViT parameter member changed: " + name)
    meta = json.loads((source / "public-deployment-metadata.json").read_text())
    vit_parameters.validate_package(dict(meta, share_rank=0), rank=0)
    values = np.load(source / "coefficients-encoded-private.npy", allow_pickle=False)
    if values.dtype != np.uint64 or values.shape != (5434, 2):
        raise ValueError("ViT coefficient slots/format changed")
    public = dict(
        _base("vit-base-patch16-224", "frozen"),
        graph_id=vit_parameters.GRAPH_ID,
        numeric_abi=vit_parameters.NUMERIC_ABI,
        lineage="Joint",
        table_sha256=file_hash(source / "coefficients-encoded-private.npy"),
        schedule=meta["points"],
        coefficient_slots=5434,
        population_binding=meta["population_binding"],
        fold_binding=meta["fold_binding"],
    )
    return _write(output, public, {name: source / name for name in expected})


def _bert_evidence(evidence, table_hash):
    if (
        evidence.get("status") != "causal_construction_complete"
        or evidence.get("table_sha256") != table_hash
        or evidence.get("graph_id") != BERT_GRAPH
    ):
        raise ValueError("BERT complete construction evidence does not match parameters")
    tasks = evidence.get("task_results", [])
    expected = {
        "cola": 5873,
        "sst2": 9398,
        "mrpc": 1779,
        "stsb": 3592,
        "qqp": 9398,
        "mnli": 9398,
        "qnli": 9398,
        "rte": 1164,
    }
    custom = evidence.get("profile") == "custom"
    if custom:
        counts = evidence.get("task_counts", {})
        if (
            not counts
            or not set(counts) <= set(expected)
            or set(counts) != set(evidence.get("model_files", {}))
            or not all(type(n) is int and n > 0 for n in counts.values())
            or len(evidence.get("population_semantic_sha256", "")) != 64
        ):
            raise ValueError("missing custom BERT model or population identity")
        expected = counts
    if not custom:
        paper_files = _asset("model_identities.json")["models"]
        if evidence.get("model_files") != {t: paper_files["bert-" + t] for t in expected}:
            raise ValueError("paper construction checkpoints differ")
    if len(tasks) != len(expected) or {r.get("task") for r in tasks} != set(expected):
        raise ValueError("all eight constructed tasks are required")
    if not evidence.get("deployment_allowed", False):
        raise ValueError("incomplete construction")
    for row in tasks:
        if (
            row.get("status") != "causal_task_complete"
            or row.get("mode") != "joint"
            or row.get("graph_id") != BERT_GRAPH
            or row.get("lineage") != BERT_LINEAGE
            or row.get("physical_length") != 128
            or row.get("sites") != 169
            or row.get("points") != 37
            or row.get("requests") != expected[row["task"]]
            or row.get("normal_design_mismatches") != 0
            or row.get("design_bits") != 12
            or row.get("design_inclusive") is not False
        ):
            raise ValueError("incomplete or mismatched BERT causal task")
    return dict(
        origin="local-cpu-construction",
        points_per_task=37,
        tasks=sorted(expected),
        requests=sum(expected.values()),
        normal_design_mismatches=0,
        evidence_file="construction.json",
    )


def pack_constructed_bert(table_path, model, output, evidence_path):
    evidence = json.loads(Path(evidence_path).read_text())
    proof = _bert_evidence(evidence, file_hash(table_path))
    public = _bert_public(Path(table_path).parent, model, "constructed-deployment")
    if evidence.get("profile") == "custom":
        task = model_spec(model)["task"]
        public.update(
            profile="custom",
            model_files=evidence["model_files"][task],
            population_binding=evidence["population_semantic_sha256"],
        )
    public["construction"] = proof
    return _write(
        output,
        public,
        {
            "private-sites.json": table_path,
            "public-schedule.json": Path(table_path).with_name("public-schedule.json"),
            "construction.json": evidence_path,
        },
    )


def _vit_evidence(source):
    source = Path(source)
    audit = json.loads((source / "audit.json").read_text())
    private = json.loads((source / "joint-private.json").read_text())
    meta = json.loads((source / "public-deployment-metadata.json").read_text())
    if (
        audit.get("construction_policy") != vit_parameters.MODEL_LOCAL_POLICY
        or meta.get("construction_policy") != audit["construction_policy"]
        or audit.get("cost_assessment") != "not_measured"
    ):
        raise ValueError("numerical audit provenance mismatch")
    custom = audit.get("profile") == "custom"
    n = audit.get("population") if custom else 10000
    if (
        type(n) is not int
        or n < 1
        or (
            custom
            and not (
                audit.get("population_binding")
                == meta.get("population_binding")
                == private.get("population")
            )
        )
    ):
        raise ValueError("invalid construction population")
    if (
        audit.get("status") != "joint_construction_audited"
        or audit.get("points") != 37
        or audit.get("population") != n
        or audit.get("normal_p12_failures") != 0
        or audit.get("logical_records") != 5578
        or audit.get("encoded_coefficient_slots") != 5434
        or audit.get("graph_id") != vit_parameters.GRAPH_ID
        or private.get("prefix") != audit.get("prefix")
    ):
        raise ValueError("complete ViT construction evidence required")
    vit_parameters.validate_package(
        dict(meta, share_rank=0),
        rank=0,
        revision=(
            audit["model_files"]["flax_model.msgpack"] if custom else vit_parameters.MODEL_REVISION
        ),
    )
    records = private.get("records", [])
    if len(records) != 5578 or [r["index"] for r in records] != list(range(5578)):
        raise ValueError("ViT logical records incomplete")
    encoded = np.zeros((5434, 2), np.uint64)
    for row in records:
        if row["operator"] == "reciprocal":
            site = row["site"]
            condition = row["condition"]
            if type(site) is not int or not 0 <= site < 144:
                raise ValueError("invalid reciprocal site")
            if condition == "CLS":
                cell = -1
            elif (
                condition.startswith("GRID")
                and condition[4:].isdigit()
                and 0 <= int(condition[4:]) < 36
            ):
                cell = int(condition[4:])
            else:
                raise ValueError("invalid reciprocal condition")
            if row["index"] != site * 37 + cell + 1:
                raise ValueError("private reciprocal logical index mismatch")
            index = None if cell == -1 else site * 36 + cell
            point = 3 * (site // 12) + 1
            head = site % 12
        elif row["operator"] == "rsqrt":
            site = row["site"]
            condition = row["condition"]
            if type(site) is not int or not 0 <= site < 25:
                raise ValueError("invalid RSqrt site")
            if condition == "CLS":
                cell = 0
            elif (
                condition.startswith("REGION")
                and condition[6:].isdigit()
                and 0 <= int(condition[6:]) < 9
            ):
                cell = 1 + int(condition[6:])
            else:
                raise ValueError("invalid RSqrt condition")
            if row["index"] != 5328 + site * 10 + cell:
                raise ValueError("private RSqrt logical index mismatch")
            index = 5184 + site * 10 + cell
            point = 36 if site == 24 else 3 * (site // 2) + (2 if site % 2 else 0)
            head = 0
        else:
            raise ValueError("unknown ViT inverse family")
        positions = [
            i for i in range(197) if vit_parameters.condition_index(row["operator"], i) == cell
        ]
        public = meta["points"][point]
        offset = head * 197 + positions[0]
        if (
            public["route"][offset] != int(row["route"] == "SPU_FALLBACK")
            or public["K"][offset] != row["public_Ks"]
        ):
            raise ValueError("private condition differs from the public execution schedule")
        if row["route"] == "CDI_NORMAL":
            e = row["selected_evidence"]
            k = row["public_Ks"]
            if (
                index is None
                or not e.get("qualified")
                or k not in e["feasible_Ks"]
                or len(e["scores"]) != 6
            ):
                raise ValueError("normal coefficient lacks its selected candidate evidence")
            if any(
                s["visited"] != s["scored_events"] or not (s["mask"] & (1 << k))
                for s in e["scores"]
            ):
                raise ValueError("incomplete selected sixway score")
            for field in ("a_encoded", "b_encoded", "L_encoded", "H_encoded"):
                if row[field] != e[field]:
                    raise ValueError("selected coefficient/domain differs")
            encoded[index] = [row["a_encoded"] % 2**64, row["b_encoded"] % 2**64]
        elif (
            row["route"] != "SPU_FALLBACK"
            or row["public_Ks"] != 0
            or any(
                row.get(k) is not None for k in ("a_encoded", "b_encoded", "L_encoded", "H_encoded")
            )
        ):
            raise ValueError("noncanonical fallback record")
    actual = np.load(source / "coefficients-encoded-private.npy", allow_pickle=False)
    if actual.dtype != np.uint64 or not np.array_equal(actual, encoded):
        raise ValueError("encoded coefficients differ from audited records")
    return meta, dict(
        origin="local-cpu-construction",
        points=37,
        requests=n,
        normal_design_mismatches=0,
        construction_policy=vit_parameters.MODEL_LOCAL_POLICY,
        numerical_complete=True,
        cost_assessment="not_measured",
        evidence_file="audit.json",
        prefix=audit["prefix"],
    )


def pack_constructed_vit(source, output):
    source = Path(source)
    meta, proof = _vit_evidence(source)
    public = dict(
        _base("vit-base-patch16-224", "constructed-deployment"),
        graph_id=vit_parameters.GRAPH_ID,
        numeric_abi=vit_parameters.NUMERIC_ABI,
        lineage="Joint",
        table_sha256=file_hash(source / "coefficients-encoded-private.npy"),
        schedule=meta["points"],
        coefficient_slots=5434,
        population_binding=meta["population_binding"],
        fold_binding=meta["fold_binding"],
        construction=proof,
    )
    audit = json.loads((source / "audit.json").read_text())
    if audit.get("profile") == "custom":
        public.update(profile="custom", model_files=audit["model_files"])
    names = (
        "audit.json",
        "joint-private.json",
        "public-deployment-metadata.json",
        "coefficients-encoded-private.npy",
    )
    return _write(output, public, {name: source / name for name in names})


def load_public(folder):
    folder = Path(folder)
    paths = [
        p for p in (folder / "parameters.json", folder / "public/parameters.json") if p.is_file()
    ]
    if len(paths) != 1:
        raise ValueError("missing or ambiguous public parameter identity")
    doc = json.loads(paths[0].read_text())
    if doc.get("schema") != "riblet-parameter-package-v3":
        raise ValueError("unsupported public parameter schema")
    ident = doc.get("deployment_id", "")
    if (
        not isinstance(ident, str)
        or len(ident) != 64
        or any(c not in "0123456789abcdef" for c in ident)
    ):
        raise ValueError("invalid deployment identifier")

    surrogate = dict(doc, model_files=doc.get("preprocessor_files", {}))
    if doc != _public_descriptor(surrogate, ident):
        raise ValueError("unexpected public deployment fields or identity")
    spec = model_spec(doc["model"])
    expected = dict(
        method="riblet-full",
        model_rank=0,
        data_rank=1,
        layout=spec["layout"],
        selected_K_only=True,
    )
    if any(doc.get(k) != v for k, v in expected.items()) or doc["kind"] not in (
        "frozen",
        "constructed-deployment",
    ):
        raise ValueError("method/model/layout/ownership binding differs")
    if doc.get("profile", "paper") not in ("paper", "custom") or (
        doc.get("profile") == "custom" and doc["kind"] != "constructed-deployment"
    ):
        raise ValueError("invalid deployment profile")
    if spec["family"] == "bert":
        if (
            doc.get("graph_id"),
            doc.get("numeric_abi"),
            doc.get("lineage"),
        ) != (BERT_GRAPH, BERT_ABI, BERT_LINEAGE):
            raise ValueError("BERT numeric graph mismatch")
        bert_tables.validate(doc["schedule"])
    else:
        if (doc.get("graph_id"), doc.get("numeric_abi"), doc.get("coefficient_slots")) != (
            vit_parameters.GRAPH_ID,
            vit_parameters.NUMERIC_ABI,
            5434,
        ):
            raise ValueError("ViT numeric graph mismatch")
        if len(doc["schedule"]) != 37:
            raise ValueError("ViT requires all 37 points")
        for i, row in enumerate(doc["schedule"]):
            if row["point"] != i or row["operator"] != vit_parameters.operator(i):
                raise ValueError("ViT topology mismatch")
            vit_parameters.validate_schedule(i, row["route"], row["K"])
    return doc


def _validate_owner(doc):
    spec = model_spec(doc.get("model", ""))
    expected = dict(
        schema="riblet-parameter-package-v1",
        method="riblet-full",
        model_rank=0,
        data_rank=1,
        layout=spec["layout"],
        selected_K_only=True,
        model_files=_asset("model_identities.json")["models"][doc["model"]],
    )
    if doc.get("profile") == "custom":
        if doc.get("kind") != "constructed-deployment" or not doc.get("population_binding"):
            raise ValueError("custom weights require bound construction")
        files = doc["model_files"]
        paper = expected["model_files"]
        if set(files) != set(paper) or any(
            not isinstance(v, str) or len(v) != 64 or any(ch not in "0123456789abcdef" for ch in v)
            for v in files.values()
        ):
            raise ValueError("unsupported custom model ABI")
        expected["model_files"] = files
    if any(doc.get(k) != v for k, v in expected.items()):
        raise ValueError("method/model/layout/ownership binding differs")
    if doc.get("kind") not in ("frozen", "constructed-deployment"):
        raise ValueError("fixture or unfinished parameters cannot deploy")
    if spec["family"] == "bert":
        if (doc.get("graph_id"), doc.get("numeric_abi"), doc.get("lineage")) != (
            BERT_GRAPH,
            BERT_ABI,
            BERT_LINEAGE,
        ):
            raise ValueError("BERT numeric graph mismatch")
        bert_tables.validate(doc["schedule"])
    else:
        proof = doc.get("construction", {})
        local = proof.get("construction_policy") == vit_parameters.MODEL_LOCAL_POLICY
        if doc["kind"] == "constructed-deployment" and not local:
            raise ValueError("constructed ViT parameters require numerical evidence")
        if "construction_policy" in proof and not local:
            raise ValueError("unknown construction policy")
        if (doc.get("graph_id"), doc.get("numeric_abi")) != (
            vit_parameters.GRAPH_ID,
            vit_parameters.NUMERIC_ABI,
        ):
            raise ValueError("ViT numeric graph mismatch")
        if len(doc["schedule"]) != 37:
            raise ValueError("ViT requires all 37 points")
        for p, row in enumerate(doc["schedule"]):
            if row["point"] != p or row["operator"] != vit_parameters.operator(p):
                raise ValueError("ViT topology/cost binding")
            vit_parameters.validate_schedule(p, row["route"], row["K"])
            if local:
                if "cost_receipt" in row or row.get(
                    "numerical_receipt"
                ) != vit_parameters.schedule_identity(p, row["operator"], row["route"], row["K"]):
                    raise ValueError("invalid numerical schedule binding")
            elif not row.get("cost_receipt"):
                raise ValueError("missing frozen parameter receipt")
    if doc["kind"] == "frozen":
        family = spec["family"]
        known = _asset("parameter_identities.json")[family]
        if doc["private_hashes"] != known:
            raise ValueError("frozen package file set or hashes changed")
    else:
        proof = doc.get("construction", {})
        expected_file = "construction.json" if spec["family"] == "bert" else "audit.json"
        if (
            proof.get("evidence_file") != expected_file
            or expected_file not in doc["private_hashes"]
            or proof.get("normal_design_mismatches") != 0
            or proof.get("origin") != "local-cpu-construction"
        ):
            raise ValueError("constructed package lacks complete bound evidence")
        if spec["family"] == "vit" and (
            proof.get("points") != 37
            or proof.get("numerical_complete") is not True
            or proof.get("cost_assessment") != "not_measured"
            or "cost_closed" in proof
        ):
            raise ValueError("ViT numerical construction is not complete")
    return doc


def load_model_package(folder, *, rank):
    if rank != 0:
        raise ValueError("only the model party reads canonical parameters")
    folder = Path(folder)
    public = load_public(folder)
    doc = json.loads((folder / "private/manifest.json").read_text())
    if (
        graph_binding(doc) != graph_binding(public)
        or _public_descriptor(doc, doc["deployment_id"]) != public
    ):
        raise ValueError("owner manifest/public deployment mismatch")
    _validate_owner(doc)
    for name, wanted in doc["private_hashes"].items():
        if Path(name).name != name:
            raise ValueError("invalid private member path")
        path = folder / "private" / name
        if not path.is_file() or path.is_symlink() or file_hash(path) != wanted:
            raise ValueError("private parameter file differs: " + name)
    if doc["model"].startswith("bert-"):
        actual = _bert_public(folder / "private", doc["model"], doc["kind"])
        if actual["schedule"] != doc["schedule"]:
            raise ValueError("public/private schedule disagreement")
        if doc["kind"] == "constructed-deployment":
            ev = json.loads((folder / "private/construction.json").read_text())
            proof = _bert_evidence(ev, doc["private_hashes"]["private-sites.json"])
            if doc.get("profile", "paper") != ev.get("profile", "paper") or (
                ev.get("profile") == "custom"
                and (
                    doc["model_files"] != ev["model_files"][doc["model"][5:]]
                    or doc["population_binding"] != ev["population_semantic_sha256"]
                )
            ):
                raise ValueError("BERT model/population evidence differs")
            if proof != doc["construction"]:
                raise ValueError("BERT construction evidence binding differs")
    else:
        meta = json.loads((folder / "private/public-deployment-metadata.json").read_text())
        if meta["points"] != doc["schedule"]:
            raise ValueError("ViT public/private schedule disagreement")
        if doc["kind"] == "constructed-deployment":
            ev = json.loads((folder / "private/audit.json").read_text())
            if doc.get("profile", "paper") != ev.get("profile", "paper") or (
                ev.get("profile") == "custom"
                and (
                    doc["model_files"] != ev["model_files"]
                    or doc["population_binding"] != ev["population_binding"]
                )
            ):
                raise ValueError("ViT model/population evidence differs")
            _, proof = _vit_evidence(folder / "private")
            if proof != doc["construction"]:
                raise ValueError("ViT construction evidence binding differs")
    return doc
