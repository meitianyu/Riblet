# Deployment and inference

[Project overview](../README.md) · [Domain and Coefficient Construction](CONSTRUCTION.md)

Run the commands from the project root (the directory containing `riblet.example.json`) on two Linux hosts. The model owner is rank 0 and the data/output owner is rank 1 for both models. This guide uses frozen parameters; to rebuild them on the model host, see [Construction](CONSTRUCTION.md).

[Install](#install) · [Configure](#configure) · [BERT](#run-bert) · [ViT](#run-vit) · [Results](#read-results) · [Configuration fields](#configuration-fields)

## Install

Use the same source revision on both Linux x86-64 hosts. The reference stack is Ubuntu 22.04, Python 3.10 and GCC 11. The model host also uses Bazel 6.5.0. Each party needs 64 logical CPUs and a 128 GiB-class machine with sufficient *available* memory for the full inventory and model.

The encoded inventory reservations are about 71.79 GiB + 8 GiB workspace for BERT, and 74.13 GiB + 16 GiB workspace for ViT. These are not whole-process memory estimates. Allow additional disk for Bazel builds and caches. Do not reduce inventory or enable online refills to fit a smaller host.

On **both hosts**:

~~~sh
sudo apt-get update
sudo apt-get install -y gcc-11 g++-11 python3.10-venv python3.10-dev \
  libssl-dev git patch unzip zip cmake ninja-build autoconf make lld-15 \
  nasm pkg-config curl
export CC=gcc-11 CXX=g++-11
python3.10 --version
gcc-11 --version
~~~

On **model0 only**, install [Bazel 6.5.0](https://github.com/bazelbuild/bazel/releases/tag/6.5.0) if needed:

~~~sh
(
  cd "$(mktemp -d)"
  curl -fLO https://releases.bazel.build/6.5.0/release/bazel-6.5.0-linux-x86_64
  printf '%s  %s\n' \
    a40ac69263440761199fcb8da47ad4e3f328cbe79ffbf4ecc14e5ba252857307 \
    bazel-6.5.0-linux-x86_64 | sha256sum -c - &&
    sudo install -m 0755 bazel-6.5.0-linux-x86_64 /usr/local/bin/bazel
)
bazel --version
~~~

The output must be `bazel 6.5.0`. Each host's installer creates a new `.venv` and installs [requirements.lock](../requirements.lock). Do not install a stock SPU wheel over the source-built runtime.

### Model owner: build once

To prefill Bazel's cache for Eigen, use the pinned revision and digest below. For other build dependencies, use the URLs and SHA-256 values declared by the corresponding Bazel repository rules.

~~~sh
export BAZEL_CACHE="$PWD/cache/bazel"
(
  set -e
  eigen_commit=66e8f38891841bf88ee976a316c0c78a52f0cee5
  eigen_sha=01fcd68409c038bbcfd16394274c2bf71e2bb6dda89a2319e23fc59a2da17210
  archive="$(mktemp)"
  curl -fL --retry 3 -o "$archive" \
    "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/$eigen_commit/eigen-$eigen_commit.tar.gz"
  printf '%s  %s\n' "$eigen_sha" "$archive" | sha256sum -c -
  mkdir -p "$BAZEL_CACHE/content_addressable/sha256/$eigen_sha"
  install -m 0644 "$archive" "$BAZEL_CACHE/content_addressable/sha256/$eigen_sha/file"
  rm -- "$archive"
)
~~~

~~~sh
python3.10 tools/install_environment.py --prefix "$PWD/.venv" \
  --source-root "$PWD/backend" --bazel "$(command -v bazel)" \
  --cache "$BAZEL_CACHE"
export PY="$PWD/.venv/bin/python"
"$PY" -m pip check
tar -C .venv/riblet-backends -cf built-runtimes.tar bert vit
~~~

The installer builds both backends in the same pinned environment. Its source paths are recorded, so keep the checkout at this location. Copy `built-runtimes.tar` to the data host's source root using your normal authenticated transfer method. For example, set these destinations in the **model-owner shell**, replacing the example values:

~~~sh
export DATA_SSH='user@data-host'
export DATA_ROOT='/absolute/path/to/Riblet'
scp built-runtimes.tar "$DATA_SSH:$DATA_ROOT/"
~~~

### Data owner: install the same build

~~~sh
mkdir built-runtimes
tar -xf built-runtimes.tar -C built-runtimes
python3.10 tools/install_environment.py --prefix "$PWD/.venv" \
  --bert-runtime "$PWD/built-runtimes/bert" \
  --vit-runtime "$PWD/built-runtimes/vit"
export PY="$PWD/.venv/bin/python"
"$PY" -m pip check
~~~

Copy only the built runtime artifacts, not the model owner's virtual environment. Both hosts should now have `.venv/installation.json`. If a build fails, read `.venv/build/<family>/riblet-build.log` before retrying with a fresh environment location; the installer does not overwrite an existing environment.

If PyPI transfers time out, set `PIP_DEFAULT_TIMEOUT=180` and `PIP_RETRIES=5` before either installer. An optional software mirror is `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`; keep the lockfile unchanged. This setting does not change model/dataset URLs.

## Configure

On both hosts:

~~~sh
cp riblet.example.json riblet.json
export PY="$PWD/.venv/bin/python"
~~~

Edit `riblet.json`, not the Python source. The example selects CoLA, `pad128`, batch size 1 and the frozen paper parameters.

| Field | Model owner | Data owner |
|---|---|---|
| `host_role` | `"model"` | `"data"` |
| `parties.model.endpoint` | Model owner's reachable private `host:19001` | Same value |
| `parties.data.endpoint` | Data owner's reachable private `host:19001` | Same value |
| `profile` / `parameters.source` | `"paper"` / `"frozen"` | Same values |
| `request.id` | A fresh ID, initially `"repro-cola01-lan-001"` | Same ID |
| `request.paper_sample` | Not read by model0 | `"samples-eval-v2-cola-01"` |
| `request.input` | Not read by model0 | Keep JSON `null` |
| `datasets.directory` | Not read by model0 | `"datasets"` or an existing data disk directory |
| `resources.local_dir` | `null` for pinned downloads (default); optionally a directory of verified local files | `null` |
| `runtime.resource_profile` | Keep `"full64"` | Same value |

Keep the example's other paths for the first run. Paths resolve relative to the configuration file. Restrict protocol ports to the peer host; SSH copy destinations are separate from these protocol endpoints. `runtime.network_label` defaults to `unshaped`; it describes your network but does not shape it.

## Prepare the model

On **model0**, set the resource location as above, then install the resources and import the frozen parameters:

~~~sh
"$PY" -m riblet.cli resources --config riblet.json --role model
"$PY" -m riblet.cli prepare-model --config riblet.json --role model
~~~

On **data1**, install the bundled public preprocessing metadata:

~~~sh
"$PY" -m riblet.cli resources --config riblet.json --role data
mkdir -p parameters/cola-paper
~~~

Model0 now has `parameters/cola-paper/public/parameters.json` and a `private/` directory. Copy **only the public file** from model0 to data1:

~~~sh
scp parameters/cola-paper/public/parameters.json \
  "$DATA_SSH:$DATA_ROOT/parameters/cola-paper/parameters.json"
~~~

Data1's package directory must contain only `parameters.json`. Never copy the model package's `private/` directory, weights or a material ledger between roles. Reuse this unchanged parameter package for later requests; do not call `prepare-model` again for an already created package.

This frozen route imports coefficients. It does not reconstruct them, needs no training dataset and does not contact data1. To rebuild coefficients locally from model-owned training data, follow [parameter construction](CONSTRUCTION.md) instead. That optional path does not train or fine-tune model weights.

### Models and resources

The resource command automatically fetches the selected model's weights and coefficients, verifies their hashes and installs matching configuration files.

- The [resource catalogue](../assets/resources.json) pins the eight BERT task checkpoints and both parameter archives. For frozen BERT, model0 downloads the selected weight and `parameters-bert-v2.tar`; the loader verifies the archive and its two pad128 members. Data1 downloads neither.
- The BERT task weights were converted from PyTorch to Flax. The [official BERT base](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594) cannot replace a task checkpoint. Users may convert the same original task weights themselves, but this release provides no converter and still requires the frozen identities.
- ViT uses the [official Flax weight](https://huggingface.co/google/vit-base-patch16-224/tree/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3). Its fixed file hash is recorded in [model identities](../assets/model_identities.json); it is not duplicated in the project resource repository.
- Configurations, tokenizers/processors and optional construction ID/fold lists are bundled in `assets/reproduction/`. Raw [GLUE](https://gluebenchmark.com/tasks/) and [ImageNet](https://www.image-net.org/) data are obtained from their official sources, not redistributed here.

Original model and dataset terms remain separate from the source license. No passwords, tokens, runtime keys or consumed randomness are distributed. Download and verification time is outside Online. For a manual local mirror, see [resource settings](#resource-settings).

## Prepare the inference datasets

This step runs on **data1 only**, after its `resources` command installs the public tokenizer/processor. It needs no weights, canonical coefficients or peer process. It is distinct from model0's `prepare-model` command.

For BERT, run:

~~~sh
"$PY" -m riblet.cli inputs --config riblet.json --role data
~~~

This downloads the **complete GLUE release**, including train/validation/test splits, WNLI and diagnostic data, from the [publisher's fixed snapshot](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c). All 34 Parquet files are checksum-verified under `datasets/glue/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/`. The command then prepares the paper's 16 BERT inputs in `inputs/paper/`. Switching BERT tasks reuses this corpus; downloading WNLI/diagnostic data does not add model support.

For ViT, obtain the **complete ImageNet-1K / ILSVRC2012 classification dataset** through [ImageNet's authorized download page](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php). Log in/request access and accept its terms yourself. Place the original archives under `datasets/imagenet-1k/` (relative to the configured data root):

~~~text
ILSVRC2012_img_train.tar
ILSVRC2012_img_val.tar
ILSVRC2012_img_test.tar
ILSVRC2012_devkit_t12.tar.gz
~~~

Allow roughly 160 GB for the archives, in addition to builds and models. Keep them intact: full extraction and a second archive copy are unnecessary. After selecting ViT and installing its processor below, the same `inputs` command verifies all four archives and prepares the two paper images from validation. Missing or changed files fail explicitly. No ImageNet credentials or restricted download URLs are distributed.

The complete inference datasets remain on data1. The [paper input index](../assets/paper-inputs.json) selects 16 BERT inputs and two ViT images by ID, split, layout and tensor hash. A single run processes one of these inputs.

## Run BERT

The commands below use CoLA-01 and fresh execution ID `repro-cola01-lan-001`.

### Data owner: describe the prepared paper input

After completing the input preparation above, run on **data1**:

~~~sh
"$PY" -m riblet.cli describe --config riblet.json --role data
~~~

Copy the resulting public descriptor to model0. Set the destinations in the **data-owner shell**, replacing the example values:

~~~sh
export MODEL_SSH='user@model-host'
export MODEL_ROOT='/absolute/path/to/Riblet'
ssh "$MODEL_SSH" "mkdir -p '$MODEL_ROOT/work/model/bert-cola/paper'"
scp work/data/bert-cola/paper/repro-cola01-lan-001-description.json \
  "$MODEL_SSH:$MODEL_ROOT/work/model/bert-cola/paper/"
~~~

The descriptor includes effective length and public shape/binding information, not text or token IDs. The release therefore does not hide this descriptor's length metadata.

### Model owner: compile and transfer the public graph

On **model0**:

~~~sh
"$PY" -m riblet.cli compile --config riblet.json --role model
ssh "$DATA_SSH" "mkdir -p '$DATA_ROOT/work/data/bert-cola/paper/graphs'"
scp -r work/model/bert-cola/paper/graphs/repro-cola01-lan-001 \
  "$DATA_SSH:$DATA_ROOT/work/data/bert-cola/paper/graphs/"
~~~

### Both parties: Cryptographic preprocessing and Online inference

Start model0, then promptly start data1 in its own terminal.

**Model owner:**

~~~sh
"$PY" -m riblet.cli run --config riblet.json --role model
~~~

**Data owner:**

~~~sh
"$PY" -m riblet.cli run --config riblet.json --role data
~~~

The first process may wait for its peer. Both then perform Cryptographic preprocessing, enter Online inference, deliver the output to data1 and retire the material. Preprocessing can take substantial time; the full shell duration is not Online latency. There is no separate offline-inventory file to save and reuse.

For CoLA-02, set data1's `request.paper_sample` to `samples-eval-v2-cola-02` and choose a new shared `request.id`. Repeat describe, compile, graph transfer and run. Keep the existing parameter package. Reusing an old graph or deleting the consumption ledger is not a valid retry.

## Run ViT

Use the same hosts and installed environment. Change these fields in both configurations:

| Field | ViT setting |
|---|---|
| `model.name` | `"vit-base-patch16-224"` |
| `model.checkpoint_dir` | `"resources/model-vit-v1"` |
| `model.preprocessor_dir` | `"resources/preprocess-vit-v1"` |
| `parameters.frozen_source` | `"resources/parameters-vit-v1"` |
| `parameters.package_dir` | `"parameters/vit-paper"` |
| `request.id` | A fresh ID, e.g. `"repro-vit01-lan-001"` |
| `request.paper_sample` on data1 | `"samples-eval-v2-imagenet1k-01"` |
| `request.input` on data1 | JSON `null` |

Keep `profile="paper"`, `parameters.source="frozen"` and each host's role/endpoints. The `construction` subsection is unused for frozen inference; it need not be changed to run ViT.

On model0, obtain the ViT resources and import its frozen coefficients:

~~~sh
"$PY" -m riblet.cli resources --config riblet.json --role model
"$PY" -m riblet.cli prepare-model --config riblet.json --role model
~~~

On data1:

~~~sh
"$PY" -m riblet.cli resources --config riblet.json --role data
mkdir -p inputs parameters/vit-paper work/data/vit-base-patch16-224/paper/graphs
~~~

After placing the four complete ImageNet archives, run on data1:

~~~sh
"$PY" -m riblet.cli inputs --config riblet.json --role data
~~~

The command prepares ImageNet-01 (`ILSVRC2012_val_00030431.JPEG`) and ImageNet-02 (`ILSVRC2012_val_00015321.JPEG`). On model0, copy the public parameters, compile and transfer the graph:

~~~sh
scp parameters/vit-paper/public/parameters.json \
  "$DATA_SSH:$DATA_ROOT/parameters/vit-paper/parameters.json"
"$PY" -m riblet.cli compile --config riblet.json --role model
scp -r work/model/vit-base-patch16-224/paper/graphs/repro-vit01-lan-001 \
  "$DATA_SSH:$DATA_ROOT/work/data/vit-base-patch16-224/paper/graphs/"
~~~

ViT uses a fixed 197-token layout, so skip `describe`. Run the same two `run` commands shown for BERT, one on each host. Do not run the two models concurrently on the pair.

## Read results

For the CoLA example, inspect these files on **data1**:

~~~text
results-local/data/bert-cola/paper/repro-cola01-lan-001/
├── result.json
└── client-output/
    ├── output.npy
    └── output.json
~~~

The data owner also gets `paper-input.json`, recording the selected content ID and tensor identity separately from the fresh execution ID. It is written locally after the protocol and is not sent to model0.

Print the actual Online time, logits and prediction:

~~~sh
"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np

p = Path("results-local/data/bert-cola/paper/repro-cola01-lan-001")
result = json.loads((p / "result.json").read_text())
print("status:", result["status"])
print("Online seconds:", result["online_latency_seconds"])
print("logits:", np.load(p / "client-output/output.npy", allow_pickle=False))
print("prediction/score:", json.loads((p / "client-output/output.json").read_text()))
PY
~~~

For ViT, change `p` to `results-local/data/vit-base-patch16-224/paper/repro-vit01-lan-001`. STS-B returns a regression value instead of a class index. Inputs do not require labels or plaintext inference. The package does not compute an arbitrary dataset's accuracy from a single request.

See [Outputs and costs](#outputs-and-costs) for the timing and communication fields.

## Outputs and costs

| Result field | Interpretation |
|---|---|
| `status` | A successful complete run reports `complete`. Failures have their own record; do not treat missing output as zero. |
| `online_latency_seconds` | Data1's continuous Online duration. It is null on model0. |
| `participant_online` | That party's online seconds and sent-byte counter. |
| `phase_totals.setup` / `preprocessing` | Separate per-party preparation boundaries. |
| `phases` and lifecycle records | Detail for model/session reuse, fresh material and auxiliary work. Nested scopes are not extra additive phase totals. |
| `owned_auxiliary` | Local input preparation or model loading outside Online. |
| `actual_setup_requests` / `setup_amortization_claimed` | The release runner records one setup request and does not claim measured amortization across separate invocations. |

Online is data1's continuous client duration, from the online barrier and input placement/sharing through output recovery. It includes the BERT coefficient infeed, secure computation, communication waits and required conversions. Model preparation, local input processing, compilation, Cryptographic preprocessing, retirement and output-file writing are outside that timer.

Add the two `participant_online.sent_bytes` values once; do not also add received bytes. Keep Setup and per-request preprocessing separate. The parties prepare concurrently, so their elapsed times must not be summed into a wall-clock cost. Nested scopes are included in their parent totals.

Data1's auxiliary input/lifecycle records can report a file-write warning without changing a completed protocol result. Preserve the output and address the file error; do not repeat successful inference just to recreate auxiliary records. Protocol and required-output failures remain errors.

One request is not a dataset-accuracy estimate. Compare classification logits/predictions with the matching reference, or use the regression score for STS-B.

## Configuration fields

Paths are relative to `riblet.json`. [Configure](#configure) lists the fields needed for a first request. Keep `runtime.installation` pointed to the installed environment and `runtime.resource_profile="full64"`. Set `output_root` and each party's `work_dir` to writable local directories. For other tasks use [Paper inputs](#paper-inputs); for numerical construction see [Construction](CONSTRUCTION.md).

## Resource settings

[assets/resources.json](../assets/resources.json) pins remote revisions and checks hashes for downloads and installed members. Leave `resources.local_dir=null` for automatic downloads. For manual BERT frozen import, place the selected task weight and `parameters-bert-v2.tar` in the configured local directory. ViT can obtain its checkpoint from the official model even if the optional local copy is absent. Data1 installs only bundled preprocessing metadata.

Keep the download cache to resume an interrupted transfer. Download locks are not request ledgers; never reset a ledger to replay a request.

## Paper inputs

Use `profile="paper"`, `parameters.source="frozen"` and `request.input=null`. The data-owner `inputs` command prepares all 16 BERT or both ViT inputs from complete datasets. Run it once per family after installing the corresponding processor, not once per request.

| Task | `model.name` | Paper content IDs | Output |
|---|---|---|---|
| CoLA | bert-cola | samples-eval-v2-cola-01 / -02 | [1,2] |
| SST-2 | bert-sst2 | samples-eval-v2-sst2-01 / -02 | [1,2] |
| MRPC | bert-mrpc | samples-eval-v2-mrpc-01 / -02 | [1,2] |
| STS-B | bert-stsb | samples-eval-v2-stsb-01 / -02 | [1,1], regression |
| QQP | bert-qqp | samples-eval-v2-qqp-01 / -02 | [1,2] |
| MNLI | bert-mnli | samples-eval-v2-mnli-01 / -02 | [1,3] |
| QNLI | bert-qnli | samples-eval-v2-qnli-01 / -02 | [1,2] |
| RTE | bert-rte | samples-eval-v2-rte-01 / -02 | [1,2] |
| ImageNet-1K | vit-base-patch16-224 | samples-eval-v2-imagenet1k-01 / -02 | [1,1000] |

For BERT, change both hosts' `model.name`, checkpoint path, package directory and fresh `request.id` together; set data1's full `request.paper_sample` ID. Use these paths:

~~~text
model.checkpoint_dir      resources/model-bert-<task>-v1
model.preprocessor_dir   resources/preprocess-bert-v1
parameters.frozen_source resources/parameters-bert-v2/pad128
parameters.package_dir  parameters/<task>-paper
~~~

Model0 imports each task package once and transfers only its public parameter file. Data1 reuses the complete GLUE corpus and prepared inputs. Every execution still needs a fresh graph and fresh protocol material.

BERT preprocessing is longest-first truncation to 128 tokens including special tokens and right padding with ID 0. The [frozen index](../assets/paper-inputs.json) records source ID, original row and effective length. MRPC source IDs are not row offsets. Hashes cover the int32 [3,1,128] element bytes in IDs/mask/segments order, not the NPY container. Runtime loading retains the original int64 boundary and subsequent secure conversion.

ViT hashes cover processed float32 [1,3,224,224] element bytes, not JPEG or NPY container bytes. The processor uses RGB images and the bound settings. The graph remains 197 tokens. Cached tensors are not processed twice.

Each `run` handles one batch1 request. Downloading the complete datasets does not run whole-dataset inference or compute dataset accuracy. Labels and stored FP32 references are not inputs to MPC.

The BERT public descriptor contains effective length, shape and request binding, not raw text or token IDs. The selected content ID and its tensor hash stay in data1's local input/result records and are not added to the protocol transcript.

## Public file transfer and path mapping

Using the example work roots, substitute the selected model, profile and new request ID:

| Artifact | Source | Destination |
|---|---|---|
| Public parameter identity | model0: `<package_dir>/public/parameters.json` | data1: `<package_dir>/parameters.json` |
| BERT request descriptor | data1: `work/data/<model>/<profile>/<id>-description.json` | model0: `work/model/<model>/<profile>/<id>-description.json` |
| Compiled request graph | model0: `work/model/<model>/<profile>/graphs/<id>/` | data1: `work/data/<model>/<profile>/graphs/<id>/` |

Create the destination parent before copying. Copy the graph directory only once, and use authenticated transfer. Only one public descriptor may exist at the configured location. For custom BERT transfer it **before** data1 runs `describe`; ViT needs no request descriptor.

Only the public schedule, preprocessing identity and opaque deployment binding are shared. The model owner checks hashes of private model and coefficient files locally. A change of model or coefficients requires a new package, graphs and fresh materials. Never transfer the private directory, input or request ledger.

## New requests and recovery

For every new request, set a fresh shared `request.id`, select data1's `request.paper_sample`, and repeat describe (BERT), compile, graph transfer and run. The unchanged model parameter package is reusable; masks and request inventory are consumed once. Run one request at a time on the pair.

| Problem | Action |
|---|---|
| Existing environment or output directory | Reuse a complete environment/package where appropriate; use a fresh request ID or new build destination. |
| Resource download failed | Keep the cache and retry the same command. Check the fixed URL; do not disable checksum or TLS checks. |
| Peer is waiting | Start both commands and check endpoints and the peer-only firewall rules. |
| Insufficient RAM | Free competing resources or provide the required capacity; do not reduce the inventory or enable Online refills. |
| Graph/parameter identity differs | Check the selected model/package on both hosts and transfer the newly compiled graph. |
| A run failed after claiming material | Keep its failure record and ledger. Use a fresh ID and graph for another attempt. |

The numerical constructor has a separate `prepare-model --resume` option. A source, model or population change requires a new construction directory. There is no online replay option, and the consumption ledger must not be reset to repeat a request.

## Reproducing timing conditions

The paper's canonical networks are LAN 3 Gbit/s / 1 ms RTT and WAN 400 Mbit/s / 10 ms RTT. RTT is round trip, not the delay to apply independently at both ends. Network shaping is not included in the launcher; measure actual ping and throughput before assigning either label.

Use the same checkpoint, preprocessing, input/layout, numerical assets, CPU/thread budget and material policy for a quantitative comparison. New-host initialization and per-invocation setup may differ from historical sessions. Each request establishes its own protocol context; a preceding warm-up does not imply that setup is shared.

Full stock requires the available-memory headroom stated in [Install](#install). Reserve disk separately for source builds, downloaded models, request graphs and optional construction checkpoints.

## Paper results

[Per-request records](../results/paper/requests.json), [means](../results/paper/means.csv), [references](../results/paper/references.json) and [setup events](../results/paper/setup-events.json) contain the paper measurements. BERT records come from the public-site pad128 protocol with YACL plus measured protocol RPC payloads; the release runner counts its direct two-party link. Current deployments write elsewhere and do not replace those records.

The set contains 36 Formal executions (32 BERT, 4 ViT), 18 content IDs and 18 task/network averages. Each average uses two different requests, not repeated timing of one sample. These are curated reproduction inputs, not an unbiased full-validation-set accuracy benchmark.

Classification comparisons use the same-weight FP32 output and label. STS-B retains its scalar and deviations. A p12 inverse residual does not bound final logit error. The release reports real secure outputs; it does not replace them with plaintext re-execution.

Online, fresh preprocessing and setup events remain separate. Count setup once over its actual consumers, including warm-up where recorded; theoretical reuse is not measured amortization. Do not add parallel party times or call a sum of phases a continuous response time. Communication is the two sent totals, not sent plus received. Historical sender arrays retain their original rank order: the original ViT model party was rank 1; this release consistently uses model0.

