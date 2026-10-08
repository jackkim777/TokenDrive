# Third-Party Notices

TokenDrive builds on public autonomous-driving research code and benchmark tooling. This repository includes only the selected TokenDrive source needed for model training, inference, and evaluation; large datasets, benchmark assets, simulator assets, checkpoints, and ONNX exports are intentionally excluded.

## SimLingo

Parts of the training, data-loading, and Bench2Drive agent stack are adapted from or interface with SimLingo-style code and data conventions.

- Upstream repository: https://github.com/RenzKa/simlingo
- Upstream code license: Apache License 2.0
- Preserved license text: [`licenses/SimLingo-LICENSE.txt`](licenses/SimLingo-LICENSE.txt)
- Upstream `NOTICE`: no separate `NOTICE` file was present in the repository root when this release package was prepared.

The SimLingo dataset license is separate from the SimLingo code license. TokenDrive does not redistribute the SimLingo dataset; users should obtain data from the original source and follow the applicable dataset terms.

## Bench2Drive, CARLA, Leaderboard, and Scenario Runner

The evaluation agent targets CARLA 0.9.15 and the Bench2Drive evaluation stack. Those projects and assets are external dependencies and are not redistributed in this repository. Users should install them separately and follow their respective licenses and usage terms.

## InternViT

TokenDrive uses `OpenGVLab/InternViT-300M-448px` as the vision encoder through the Hugging Face/Transformers ecosystem. Model weights are downloaded separately according to the upstream model license.

## PyTorch, PyTorch Lightning, DeepSpeed, Hydra, Transformers, and Other Dependencies

Runtime dependencies are listed in [`environment.yaml`](environment.yaml). Users should review upstream package licenses when redistributing modified environments or binaries.
