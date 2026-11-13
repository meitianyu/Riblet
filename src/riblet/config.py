_WIDTHS = {"cola": 2, "sst2": 2, "mrpc": 2, "stsb": 1, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2}


def model_spec(name):
    if name.startswith("bert-") and name[5:] in _WIDTHS:
        task = name[5:]
        return dict(
            model=name,
            family="bert",
            task=task,
            layout="pad128",
            model_rank=0,
            data_rank=1,
            output_shape=[1, _WIDTHS[task]],
            output_kind="regression" if task == "stsb" else "classification",
            batch_size=1,
            max_tokens=128,
            max_concurrency=64,
        )
    if name == "vit-base-patch16-224":
        return dict(
            model=name,
            family="vit",
            layout="vit-standard",
            model_rank=0,
            data_rank=1,
            output_shape=[1, 1000],
            output_kind="classification",
            batch_size=1,
            input_shape=[1, 3, 224, 224],
            tokens=197,
            max_concurrency=64,
            material_lanes=48,
        )
    raise ValueError("Unsupported model/checkpoint family")


def rank_for(model, role):
    spec = model_spec(model)
    if role not in ("model", "data"):
        raise ValueError("Roles are model and data, with ranks 0 and 1")
    return spec[role + "_rank"]


def validate_parties(value):
    if not isinstance(value, dict) or set(value) != {"model", "data"}:
        raise ValueError("Exactly model and data parties are required")
    endpoints = []
    for role, rank in [("model", 0), ("data", 1)]:
        row = value[role]
        if not isinstance(row, dict) or type(row.get("rank")) is not int or row["rank"] != rank:
            raise ValueError("Both models require model0/data1")
        endpoint = row.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip() or ":" not in endpoint:
            raise ValueError("Explicit host:port endpoint is required")
        host, port = endpoint.rsplit(":", 1)
        if not host or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError("Invalid endpoint")
        endpoints.append(endpoint)
    if len(set(endpoints)) != 2:
        raise ValueError("Two distinct party endpoints are required")
    return endpoints


def runtime_config():
    from spu import spu_pb2 as pb

    c = pb.RuntimeConfig(
        protocol=pb.CHEETAH,
        field=pb.FM64,
        fxp_fraction_bits=18,
        fxp_exp_iters=6,
        fxp_div_goldschmidt_iters=2,
        max_concurrency=64,
        enable_type_checker=True,
        experimental_enable_colocated_optimization=True,
        experimental_enable_intra_op_par=False,
    )
    c.cheetah_2pc_config.enable_mul_lsb_error = True
    c.cheetah_2pc_config.approx_less_precision = 4
    c.cheetah_2pc_config.ot_kind = pb.YACL_Ferret
    return c


def public_audit(code):

    import re

    secret, public = "tensor<33x!pphlo.secret<i64>>", "tensor<33xi64>"
    lines = code.splitlines(keepends=True)

    def unique(pattern):
        hits = [(i, m) for i, line in enumerate(lines) if (m := re.search(pattern, line))]
        if len(hits) != 1:
            raise ValueError("expected one direct request audit producer and return")
        return hits[0]

    i, op = unique(r'^\s*(%[\w.$-]+)\s*=\s*pphlo\.custom_call\s+@"mjv_audit_snapshot:[^"]+"')
    sig, _ = unique(r"^\s*func\.func\s+(?:public\s+)?@main\(")

    ret, values = unique(r"^\s*(?:func\.)?return\s+([^:]+)\s*:")
    if values[1].strip().split(",")[-1].strip() != op[1]:
        raise ValueError("audit must be the final request return")
    for index, prefix in [(i, " -> "), (sig, " -> "), (ret, " : ")]:
        before, sep, after = lines[index].rpartition(prefix)

        match = re.fullmatch(
            r"(.*?)(tensor<33x(?:!pphlo\.secret<i64>|i64)>)(\s*\)?\s*(?:\{\s*)?(?:loc\([^\n]*\))?\s*)",
            after,
        )
        if not sep or not match or match[2] not in (secret, public):
            raise ValueError("unexpected audit result signature")
        lines[index] = before + sep + match[1] + public + match[3]

    for index, line in enumerate(lines):
        consumer = re.search(r'pphlo\.custom_call\s+@"mjv_epoch_end:[^"]+"\(([^)]*)\)', line)
        if consumer is None:
            continue
        operands = [value.strip() for value in consumer[1].split(",")]
        if op[1] not in operands:
            continue
        if len(operands) != 2 or operands[-1] != op[1]:
            raise ValueError("unexpected epoch-end audit operand")
        before, sep, signature = line.rpartition(" : ")
        inputs, arrow, output = signature.partition(" -> ")
        match = re.fullmatch(r"(\(.*?,\s*)(tensor<33x(?:!pphlo\.secret<i64>|i64)>)(\))", inputs)
        if not sep or not arrow or not match:
            raise ValueError("unexpected epoch-end audit signature")
        lines[index] = before + sep + match[1] + public + match[3] + arrow + output
    return "".join(lines)


def _validate_sections(doc):
    known = {
        "": "host_role profile model parties runtime parameters construction resources datasets request output_root",
        "model": "name checkpoint_dir preprocessor_dir",
        "runtime": "installation resource_profile network_label",
        "parameters": "source package_dir frozen_source",
        "construction": "split candidates data_root dataset_version membership models_root selection workers output_dir",
        "construction.selection": "mode count seed path scope",
        "resources": "catalogue local_dir cache directory",
        "datasets": "directory",
        "request": "id input paper_sample",
        "parties": "model data",
        "parties.model": "endpoint work_dir",
        "parties.data": "endpoint work_dir",
    }
    for section, names in known.items():
        value = doc
        for key in section.split(".") if section else ():
            value = value.get(key, {})
        extra = set(value) - set(names.split())
        if extra:
            raise ValueError(
                "unknown configuration field: "
                + (section + "." if section else "")
                + sorted(extra)[0]
            )


def _validate_construction(doc, family):
    profile, model = doc["profile"], doc["model"]["name"]
    selection=doc["construction"]["selection"]
    if "scope" in selection and not (profile=="custom" and family=="bert" and selection["mode"]=="sample" and selection["scope"]=="all-paper-tasks" and type(selection["count"]) is int):
        raise ValueError("all-paper-tasks requires a custom BERT sample total")
    if profile == "paper" and doc["construction"]["selection"] != {"mode": "paper"}:
        raise ValueError("paper uses the frozen membership; use custom for sampling")
    if doc["construction"].get("split", "train") != "train":
        raise ValueError("the supported construction split is train")
    if doc["construction"]["selection"].get("mode") == "sample":
        count = doc["construction"]["selection"]["count"]
        if isinstance(count, dict) and (family != "bert" or set(count) != {model[5:]}):
            raise ValueError("custom construction builds the selected task; specify its count only")
    workers = doc["construction"]["workers"]
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("choose 1..64 construction workers")


def load_run_config(path, role, action="all"):
    import json
    from pathlib import Path

    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding="utf-8"))
    _validate_sections(doc)
    if action not in ("all", "resources", "inputs", "prepare-model", "describe", "compile", "run"):
        raise ValueError("unknown configuration action")
    if role not in ("model", "data"):
        raise ValueError("choose the model or data host")
    if doc["host_role"] != role:
        raise PermissionError("role does not match this host configuration")
    model = doc["model"]["name"]
    family = model_spec(model)["family"]
    profile, source = doc["profile"], doc["parameters"]["source"]
    if profile not in ("paper", "custom") or source not in ("construct", "frozen"):
        raise ValueError("unsupported population profile or parameter source")
    if action == "inputs":
        if role != "data":
            raise PermissionError("inference dataset preparation belongs to data1")
        return dict(role=role, model=model, family=family, profile=profile, source=source,
                    directory=path.parent,
                    dataset_root=(path.parent / doc["datasets"]["directory"]).resolve(),
                    processor=(path.parent / doc["model"]["preprocessor_dir"]).resolve(),
                    paper_input_dir=path.parent / "inputs/paper")
    if action == "all" or (action == "prepare-model" and source == "construct"):
        _validate_construction(doc, family)
    if action in ("prepare-model", "resources"):
        if action == "prepare-model" and role != "model":
            raise PermissionError("numerical construction belongs to model0")

        def local(value):
            return (path.parent / value).resolve()

        result = dict(
            role=role,
            model=model,
            family=family,
            profile=profile,
            source=source,
            directory=path.parent,
            parameters=local(doc["parameters"]["package_dir"]),
        )
        if action == "resources":
            result["resources"] = doc["resources"]
            return result
        result.update(
            checkpoint=local(doc["model"]["checkpoint_dir"]),
            processor=local(doc["model"]["preprocessor_dir"]),
        )
        if source == "construct":
            result.update(construction=doc["construction"], construction_root=local(doc["construction"]["output_dir"]))
        elif profile == "paper":
            result["frozen"] = local(doc["parameters"]["frozen_source"])
        return result
    if doc["runtime"]["resource_profile"] != "full64":
        raise ValueError("only the admitted full64 resource profile is supported")
    parties = {
        k: {"rank": n, "endpoint": doc["parties"][k]["endpoint"]}
        for n, k in enumerate(("model", "data"))
    }
    validate_parties(parties)

    def local(value):
        return (path.parent / value).resolve()

    work = local(doc["parties"][role]["work_dir"])
    request = doc["request"]["id"]
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", request):
        raise ValueError("use a fresh alphanumeric request identifier")
    root = work / model / profile
    result = dict(
        role=role,
        request_id=request,
        model=model,
        family=family,
        profile=profile,
        source=source,
        installation=str(local(doc["runtime"]["installation"])),
        parties=parties,
        network=doc["runtime"]["network_label"],
        work=work,
        state_dir=str(work / "ledger"),
        parameters=local(doc["parameters"]["package_dir"]),
        description=root / (request + "-description.json"),
        graph=root / "graphs" / request,
        result=local(doc["output_root"]) / role / model / profile / request,
        directory=path.parent,
    )
    if action == "all" or (role == "model" and action in ("compile", "run")):
        result["checkpoint"] = local(doc["model"]["checkpoint_dir"])
    if (
        action == "all"
        or action == "describe"
        or (role == "data" and action == "run")
        or (role == "model" and action in ("compile", "run"))
    ):
        result["processor"] = local(doc["model"]["preprocessor_dir"])
    if action == "all" or action == "describe" or (role == "data" and action == "run"):
        value = doc["request"].get("input")
        result["inputs"] = local(value) if value else None
        result["paper_sample"] = doc["request"].get("paper_sample")
    if action == "all":
        result["resources"] = doc["resources"]
        result["construction"] = doc["construction"]
        result["construction_root"] = local(doc["construction"]["output_dir"])
        result["frozen"] = local(doc["parameters"]["frozen_source"])
    return result


def construction_policy(config):

    import copy

    construction = config["construction"]
    selection = copy.deepcopy(construction["selection"])
    selection.pop("path", None)  
    return dict(
        profile=config["profile"],
        model=config["model"],
        dataset_version=construction.get("dataset_version"),
        split=construction.get("split", "train"),
        selection=selection,
    )
