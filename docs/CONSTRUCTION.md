# Domain and Coefficient Construction

[Project overview](../README.md) · [Deployment](DEPLOYMENT.md#prepare-the-model)

Run this optional stage on **model0** using the model owner's checkpoint and training data. No data-owner input or connection is needed. To use frozen coefficients instead, follow [Prepare the model](DEPLOYMENT.md#prepare-the-model). Neither route trains model weights.

## Choose the numerical route

| Setting | Meaning |
|---|---|
| `profile="paper"`, `parameters.source="frozen"` | Import the exact published coefficients. This is the route for comparison with the paper. |
| `profile="paper"`, `parameters.source="construct"` | Use the preserved design population and rebuild numerical parameters locally. BERT's paper-population controller covers all eight tasks. |
| `profile="custom"`, `parameters.source="construct"` | Use a newly selected training population for the selected supported model/task. This creates a different parameter asset. |

Numerical admission is evaluated on the chosen design population, not every possible operand. Allow disk space for the model checkpoints and intermediate shards.

## Install and keep ownership local

First complete [installation](DEPLOYMENT.md#install) on model0. Use its `.venv/bin/python`:

~~~sh
export PY="$PWD/.venv/bin/python"
~~~

Edit the existing `riblet.json`. Keep `host_role="model"` and set `parameters.source="construct"`. Choose unused directories for `parameters.package_dir` and `construction.output_dir`. Do not point reconstruction at the published frozen-resource directories.

Required fields for `prepare-model`: `host_role`, `profile`, `model`, `parameters` and `construction`. Add `resources` if you will fetch assets. The constructor does not require `parties`, `runtime`, `request` or `output_root`; those fields can remain in a full configuration but are not used by this numerical stage.

## BERT with the preserved population

All commands below run on **model0**, from the repository root with `riblet.json` there. For a CoLA package export, set these fields first; keep the remaining CoLA model paths from the deployment guide:

| Field | Value |
|---|---|
| `profile` | `"paper"` |
| `parameters.source` | `"construct"` |
| `parameters.package_dir` | `"parameters/cola-reconstructed"` |
| `construction.split` | `"train"` |
| `construction.data_root` | `"datasets/glue"` |
| `construction.dataset_version` | `"bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"` |
| `construction.membership` | `"resources/construction-bert-v2/bert-development-membership.json"` |
| `construction.models_root` | `"construction-models"` |
| `construction.selection` | `{"mode":"paper"}` |
| `construction.workers` | An integer from 1 to 64; the template uses 16 |
| `construction.output_dir` | `"construction/bert-reconstructed"` |

### Fetch the construction resources once

The `construct` route for any supported BERT task resolves **all eight task checkpoints**, their preprocessing metadata and the frozen construction membership. The `frozen` route instead fetches the selected task's checkpoint and its frozen coefficients. Do not repeat the construction download for every task:

~~~sh
"$PY" -m riblet.cli resources --config riblet.json --role model
~~~

Create the eight model-directory mappings using the configured paths:

~~~sh
"$PY" - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("riblet.json").read_text())
models = Path(config["construction"]["models_root"])
resources = Path(config["resources"]["directory"])
models.mkdir(parents=True, exist_ok=True)
for task in ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"):
    target = (resources / f"model-bert-{task}-v1").resolve()
    link = models / task
    if link.is_symlink() and link.resolve() == target:
        continue
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Review the existing model mapping: {link}")
    link.symlink_to(target, target_is_directory=True)
PY
~~~

This does not start construction or fetch completed coefficient tables.

### Prepare the fixed training data

Obtain [GLUE](https://gluebenchmark.com/tasks/) under its dataset terms. The following command downloads only the eight required **training splits** from the same fixed publisher revision used in the paper. It then exports the Arrow IPC streams expected by the existing constructor, preserving row order, original `idx`, fields and types. It does not select the design population or fit parameters; the constructor later applies the frozen membership, token and fold checks.

This is model-owned preparation, independent of data1's complete inference-dataset download. Existing matching Arrow streams are reused; different existing contents are rejected rather than overwritten.

~~~sh
"$PY" - <<'PY'
# Export the fixed GLUE training splits on model0.
import json
import tempfile
from pathlib import Path
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from riblet.assets import file_hash

config = json.loads(Path("riblet.json").read_text())
if config["host_role"] != "model":
    raise PermissionError("Training-data preparation belongs to model0")
if (config["profile"], config["parameters"]["source"], config["construction"]["split"]) != ("paper", "construct", "train"):
    raise ValueError("Use the paper construction configuration with split=train")
source = json.loads(Path("assets/paper-inputs.json").read_text())["datasets"]["glue"]
if config["construction"]["dataset_version"] != source["revision"]:
    raise ValueError("Training dataset revision differs from the frozen source")
tasks = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte")
files = [row for row in source["files"] if row["path"] in
         {f"{task}/train-00000-of-00001.parquet" for task in tasks}]
if len(files) != len(tasks):
    raise ValueError("Incomplete fixed training-file catalogue")
output = Path(config["construction"]["data_root"])
snapshot = Path(snapshot_download(
    repo_id=source["repo_id"], repo_type="dataset", revision=source["revision"],
    allow_patterns=[row["path"] for row in files],
    local_dir=output / "source-parquet",
))
for row in files:
    path = snapshot / row["path"]
    if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
        raise ValueError(f"Training Parquet identity mismatch: {path}")
    table = pq.read_table(path)
    target = output / Path(row["path"]).parts[0] / "glue-train.arrow"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open("rb") as stream:
            previous = ipc.open_stream(stream).read_all()
        if not previous.equals(table, check_metadata=True):
            raise ValueError(f"Existing training data differs: {target}")
    else:
        with tempfile.TemporaryDirectory(prefix=".glue-export-", dir=target.parent) as stage:
            temporary = Path(stage) / "train.arrow"
            with temporary.open("wb") as stream, ipc.new_stream(stream, table.schema) as writer:
                writer.write_table(table)
            temporary.rename(target)
    print(f"Verified {target}: {table.num_rows} training rows")
PY
~~~

With the example paths, this produces `datasets/glue/<task>/glue-train.arrow` for all eight tasks. Keep `construction.data_root` pointed to that parent directory. Downloads, format conversion and model-local construction are all outside secure Online timing.

The frozen membership selects 50,000 distinct requests across the eight training splits, preserving their original request ordinals and folds. It uses CoLA 5,873; SST-2 9,398; MRPC 1,779; STS-B 3,592; QQP 9,398; MNLI 9,398; QNLI 9,398; RTE 1,164. Never replace training data with validation/test inputs, renumber source IDs or choose a new sample to bypass a membership check.

BERT fits each reciprocal site from valid non-first queries of the corresponding layer and head. Each LayerNorm site uses all valid tokens. Five held-out folds and the full pool determine the smallest common feasible K in 0–8; an unadmitted site has no K. At inference the first reciprocal query and padding use the generic kernel. Domain membership is not checked on private online operands. Only the schedule is shared; encoded coefficients stay with the model owner.

## ViT with the preserved population

Use the [ImageNet training archive](https://www.image-net.org/), not validation images. After selecting ViT in the configuration, set `profile="paper"` and `parameters.source="construct"`, then obtain the checkpoint and membership metadata:

~~~sh
"$PY" -m riblet.cli resources --config riblet.json --role model
~~~

Prepare the original ten images per class from the authorized archive:

~~~sh
PYTHONPATH="$PWD/offline/vit" "$PY" -m vit_cdi.prepare_data \
  --archive /path/to/ILSVRC2012_img_train.tar \
  --output "$PWD/datasets/imagenet-paper"
~~~

The tool prepares ten training images per class. Construction verifies all 10,000 image identities and fold assignments against the frozen membership.

Use the [ViT model paths](DEPLOYMENT.md#run-vit), and set:

| Field | Value |
|---|---|
| `parameters.package_dir` | `"parameters/vit-reconstructed"` |
| `construction.split` | `"train"` |
| `construction.data_root` | `"datasets/imagenet-paper"` |
| `construction.dataset_version` | A stable identifier for that source archive, e.g. `"ILSVRC2012-train"`; keep it unchanged on resume |
| `construction.membership` | `"resources/construction-vit-v2/manifest.json"` |
| `construction.selection` | `{"mode":"paper"}` |
| `construction.workers` | An integer from 1 to 64 |
| `construction.output_dir` | `"construction/vit-reconstructed"` |

Image bytes are read from `construction.data_root/images/<relative_id>`. The membership hash and image hashes, not the human-readable version label alone, bind this population.

## Construct, inspect and resume

On **model0 only**:

~~~sh
"$PY" -m riblet.cli prepare-model --config riblet.json --role model
~~~

The controller processes 37 causal points per model/task: collect the current operands, fit locally, propagate the selected output, then proceed to the next point. It does not fit every layer against an unrelated plaintext replay.

The main outputs are:

| Location | Contents |
|---|---|
| `construction.output_dir/INPUTS.json` | Model/population/policy binding for resume |
| BERT: `construction.output_dir/population.parquet` | Verified design population |
| BERT: `construction.output_dir/fit/{public-schedule.json,private-sites.json,CONSTRUCTION_RESULT.json}` | Constructed table and completion evidence |
| ViT: `construction.output_dir/population/` | Verified image population |
| ViT: `construction.output_dir/fit/constructed-joint/READY.json` | Complete audited numerical output |
| `parameters.package_dir/public/parameters.json` | Public deployment identity and schedule |
| `parameters.package_dir/private/` | Canonical coefficients and owner-only `manifest.json` integrity/provenance bindings |

BERT's paper controller builds the full eight-task Joint table; the CLI packages the model selected by `model.name`. Partial fixtures and unfinished runs cannot be packaged for inference.

Resume an interrupted construction with the same model, input contents, source and numerical policy:

~~~sh
"$PY" -m riblet.cli prepare-model --config riblet.json --role model --resume
~~~

Worker count and data paths are execution settings. Use a new construction directory when the model, selected population, source code or numerical policy changes.

After packaging, follow the deployment guide's [public-file transfer](DEPLOYMENT.md#public-file-transfer-and-path-mapping) and inference steps. Keep the matching `profile` and `parameters.source="construct"` on both hosts so the controller expects a constructed package. Do not rerun construction before every request.

Construction population, fold, model and coefficient content hashes remain on model0. Only the public schedule, public preprocessing identity and independent deployment binding are exported.

## Changing the design population size

This is a research option, not required for paper reproduction. Keep the paper checkpoint and preprocessor, select `profile="custom"`, use new output/package directories and set `parameters.source="construct"`. The `resources` command is not available under the custom profile; obtain the matching paper model/preprocessor first.

For sampling, the `construction.selection` object is:

~~~json
{"mode": "sample", "count": 1000, "seed": 1}
~~~

This is a format example, not a recommended sample size or a quality guarantee. The selector orders stable source IDs deterministically without replacement and assigns five folds. By default, custom BERT constructs the selected task; `count` is an integer or a one-task map. `construction.data_root/<task>/glue-train.arrow` must contain the original IDs and text fields. To divide one total among all eight tasks, use `{"mode":"sample","scope":"all-paper-tasks","count":40000,"seed":1}` with eight checkpoint/tokenizer directories in `construction.models_root/<task>`. This is a custom population, not the paper membership.

For ViT, also set `construction.candidates` to a JSON list of original training-image records. Each record needs `split="train"`, `relative_id`, `class_id` (original 0..999 label), `bytes` and `sha256`. Images live under `construction.data_root/images/`. Do not renumber a subset of classes.

An explicit selection instead uses:

~~~json
{"mode": "manifest", "path": "design-selection.json"}
~~~

That file is a list of unique `{"id": "...", "fold": 0}` records, with folds in 0..4. BERT IDs match `idx`; ViT IDs match `relative_id`. The manifest contents are hashed into the construction identity.

Changing count, seed or membership can change support, domains, coefficients, K and fallback choices. It does not reproduce the frozen table by definition. No new model training, online secret-domain checking or measured-cost tuning is performed by these options.

