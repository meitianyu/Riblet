# Third-party notices

Riblet includes modified SPU/OpenBumbleBee source under Apache-2.0. The source  
trees retain their original LICENSE, LEGAL and copyright notices.

- [OpenBumbleBee](https://github.com/AntCPLab/OpenBumbleBee/blob/ndss/LICENSE)
- [SPU](https://github.com/secretflow/spu/blob/main/LICENSE)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)

Upstream revisions are listed in [backend/sources.lock.json](backend/sources.lock.json).  
YACL, SEAL, XLA and other build dependencies are pinned in the Bazel files and  
retain their own licenses. Python dependencies are listed in  
[requirements.lock](requirements.lock); their package licenses apply.

Riblet-owned code uses the root Apache-2.0 license. This does not relicense  
third-party models or datasets. Weights and raw datasets are not included in  
this source archive. Model configurations and preprocessing metadata retain  
their providers' terms; construction manifests contain IDs, folds and hashes,  
not training text or images. Resource sources are listed in the  
[deployment guide](docs/DEPLOYMENT.md#models-and-resources).

