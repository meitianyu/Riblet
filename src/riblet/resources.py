from pathlib import Path
import json
from .config import model_spec
from .assets import digest


def _capacity(family):
    return json.loads(
        (Path(__file__).resolve().parents[2] / "assets/material_capacities.json").read_text()
    )[family]


def reservation(family, quotas):
    if family == "bert":
        if len(quotas) != 50 or any(
            len(r) != 4 or any(type(v) is not int or not 0 <= v < 2**32 for v in r) for r in quotas
        ):
            raise ValueError("BERT mixed-ring capacity shape")
        if quotas[-1][0] != 2 or quotas[-1][2:] != [0, 0] or quotas[-2][1:] != [0, 0, 0]:
            raise ValueError("BERT material ABI marker")
        return (
            sum(16 * (r[0] + r[1]) + r[2] + r[3] for r in quotas[:-2])
            + 24 * quotas[-2][0]
            + 12 * quotas[-1][1]
        )
    if family == "vit":
        if len(quotas) != 98 or any(type(v) is not int or not 0 <= v < 2**32 for v in quotas):
            raise ValueError("ViT lane capacity shape")
        return (
            2 * 48 * 10485760 * 16
            + 24 * quotas[0]
            + 12 * quotas[1]
            + sum(32 * quotas[2 + 2 * i] + 49 * quotas[3 + 2 * i] for i in range(48))
        )
    raise ValueError("unknown inventory family")


def bind_budget(graph, runtime_sha256):
    spec = model_spec(graph["model"])
    family = spec["family"]
    if (
        graph.get("family") != family
        or graph.get("method") != "riblet-full"
        or (graph.get("model_rank"), graph.get("data_rank")) != (0, 1)
    ):
        raise ValueError("graph method/ownership mismatch")
    expected = {
        "max_concurrency": 64,
        "field": "FM64",
        "fxp_fraction_bits": 18,
        "intra_op_parallel": False,
    }
    if graph.get("runtime") != expected:
        raise ValueError("explicit frozen 64-worker numerical configuration required")
    if len(runtime_sha256) != 64 or any(c not in "0123456789abcdef" for c in runtime_sha256):
        raise ValueError("runtime content hash required")
    source = _capacity(family)
    size = reservation(family, source["quotas"])
    if size != source["reserved_bytes"]:
        raise ValueError("capacity source accounting differs")
    binding = dict(
        model=graph["model"],
        package_id=graph["package_id"],
        graphs=graph["graphs"],
        runtime_sha256=runtime_sha256,
        model_epoch=graph["model_epoch"],
        request_epoch=graph["request_epoch"],
        model_rank=0,
        data_rank=1,
        max_concurrency=64,
    )
    return dict(
        source,
        binding=binding,
        binding_sha256=digest(binding),
        family=family,
        lanes=48,
        online_refill=False,
        from_warmup_consumption=False,
        source_is_conservative_capacity_not_new_measurement=True,
    )


def validate_budget(budget, graph, runtime_sha256):
    if budget != bind_budget(graph, runtime_sha256):
        raise ValueError("budget/configuration/source binding changed")
    return budget


def require_headroom(available, budget):
    if len(available) != 2 or any(type(v) is not int or v < 0 for v in available):
        raise ValueError("two real memory censuses required")
    required = budget["reserved_bytes"] + budget["workspace_headroom_bytes"]
    if min(available) < required:
        raise MemoryError("insufficient RAM for complete fresh inventory and workspace")
    return required


def require_schedule(model, schedule):

    spec = model_spec(model)
    if not isinstance(schedule, (dict, list)):
        raise ValueError("missing public schedule")
    if spec["family"] == "bert":
        from .bert_tables import validate
        validate(schedule)
    else:
        from .vit_parameters import validate_schedule, operator

        if len(schedule) != 37:
            raise ValueError("incomplete ViT schedule")
        for point, row in enumerate(schedule):
            if row["point"] != point or row["operator"] != operator(point):
                raise ValueError("unexpected ViT point")
            validate_schedule(point, row["route"], row["K"])
