# Third-Party Notices

TokenDrive builds on public autonomous-driving research code and benchmark tooling. This repository includes only the selected TokenDrive source needed for model training, inference, and evaluation; large datasets, benchmark assets, simulator assets, checkpoints, and ONNX exports are intentionally excluded.

## SimLingo

Parts of the training and data-loading stack are adapted from SimLingo-style code and data conventions. The local reference copy used while preparing this release is licensed under Apache License 2.0.

## Bench2Drive, CARLA, Leaderboard, and Scenario Runner

The evaluation agent targets CARLA 0.9.15 and the Bench2Drive evaluation stack. Those projects and assets are external dependencies and are not redistributed in this repository. Users should install them separately and follow their respective licenses and usage terms.

## InternViT

TokenDrive uses `OpenGVLab/InternViT-300M-448px` as the vision encoder through the Hugging Face/Transformers ecosystem. Model weights are downloaded separately according to the upstream model license.

## PyTorch, PyTorch Lightning, DeepSpeed, Hydra, Transformers, and Other Dependencies

Runtime dependencies are listed in [`environment.yaml`](environment.yaml). Users should review upstream package licenses when redistributing modified environments or binaries.
