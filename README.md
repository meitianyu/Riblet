# Riblet v0.1.0

Riblet runs two-party BERT and ViT inference on CPUs. It extends [OpenBumbleBee](https://github.com/AntCPLab/OpenBumbleBee/tree/ndss) with minimum joint-view correlated triples (MJV-CT) for linear layers and calibrated-domain inversion (CDI) at public computation sites. Model weights and encoded coefficients belong to rank 0; the input and recovered logits belong to rank 1.

## Run the paper models

Use the same pair of Linux hosts for both models. See [installation](docs/DEPLOYMENT.md#install) for the pinned toolchain, memory requirements and how to install one environment with both backends. Copy `riblet.example.json` to `riblet.json` on each host and [set the roles, paths and endpoints](docs/DEPLOYMENT.md#configure).

1. **Model owner:** [install the model resources and import the frozen coefficients](docs/DEPLOYMENT.md#prepare-the-model). With the default configuration, the resource command downloads the selected BERT task weight and its frozen public-site parameters from pinned revisions. To regenerate numerical parameters instead, follow [Domain and Coefficient Construction](docs/CONSTRUCTION.md); only the model owner participates.
2. **Data owner:** [prepare the inference dataset and selected input](docs/DEPLOYMENT.md#prepare-the-inference-datasets). The data owner does not need the model checkpoint or private coefficient files.
3. **Model owner:** compile the request and send the public graph to the data owner. **Both hosts:** follow [Run BERT](docs/DEPLOYMENT.md#run-bert) or [Run ViT](docs/DEPLOYMENT.md#run-vit). Each session performs **Cryptographic preprocessing** followed by **Online inference**.
4. **Data owner:** read the logits, class or STS-B regression score and `online_latency_seconds` from the [result directory](docs/DEPLOYMENT.md#read-results). Setup and preprocessing are recorded separately.

## Where to look

| Directory | Contents |
|---|---|
| `src/riblet/`, `tools/` | Configuration, resources, installation and two-party execution |
| `offline/`, `frontends/` | Model-local parameter construction and secure model graphs |
| `backend/` | SPU/OpenBumbleBee source and Riblet protocol implementations |
| `assets/` | [Resource catalogue](assets/resources.json), model identities and input manifests |
| `results/paper/` | [Measured requests](results/paper/requests.json), [two-input means](results/paper/means.csv) and [timing conditions](docs/DEPLOYMENT.md#reproducing-timing-conditions) |

Riblet-owned code is [Apache-2.0 licensed](LICENSE). See [NOTICE](NOTICE), [modifications](MODIFICATIONS.md) and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and resource terms.

