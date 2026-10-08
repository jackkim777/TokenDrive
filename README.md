# TokenDrive

Official code release for **TokenDrive: Real-Time Closed-Loop Driving with Structured Decision Tokens**.

TokenDrive is an LM-free end-to-end autonomous driving policy. It predicts structured context and action tokens from a front-view image, high-level command, and ego speed, then conditions waypoint prediction through an Action Conditioning Module (ACM). The online control path does not use autoregressive language generation.

## Highlights

- Frozen **InternViT-300M-448px** vision encoder
- Structured scene tokens for lateral action, longitudinal action, speed context, and route context
- ACM with FiLM modulation for token-conditioned waypoint prediction
- 10 short-horizon speed waypoints and 64 long-horizon route waypoints
- V3 training with second-order waypoint smoothness losses
- Bench2Drive closed-loop evaluation agent

## Repository Structure

```text
.
├── tokendrive/                  # Model, training, data loading, and analysis code
│   ├── train_v3.py              # Main training entry point
│   ├── config/                  # Hydra configs
│   ├── models/                  # TokenDrive model definitions
│   ├── dataloader/              # SimLingo-style dataset and datamodule
│   ├── data_generation/         # Structured token label generation scripts
│   ├── ablation/                # Ablation and intervention analysis scripts
│   └── tools/                   # Lightweight dataset/token summary utilities
├── team_code/                   # Bench2Drive/CARLA evaluation agent
├── pretrained/                  # Checkpoint placeholder layout
├── start_eval_tokendrive_v3.py  # Bench2Drive SLURM evaluation launcher
├── environment.yaml             # Conda environment
├── licenses/                    # Third-party license texts
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Installation

```bash
git clone https://github.com/<USER_OR_ORG>/TokenDrive.git
cd TokenDrive
conda env create -f environment.yaml
conda activate tokendrive
```

The evaluation scripts additionally require CARLA 0.9.15 and the Bench2Drive benchmark assets.

## Data Preparation

Prepare the SimLingo-style dataset so that it contains `data/simlingo` and `commentary/simlingo` under one dataset root:

```text
<SIMLINGO_DATASET_ROOT>/
├── data/simlingo/
└── commentary/simlingo/
```

Generate TokenDrive structured labels:

```bash
export SIMLINGO_DATASET_ROOT=/path/to/simlingo_dataset
export TOKENDRIVE_REPO_ROOT=$PWD

python tokendrive/data_generation/slots_generator.py
python tokendrive/data_generation/template_id_generator.py
```

The training config reads the dataset path from `SIMLINGO_DATASET_ROOT` by default, or from the Hydra override `data_module.data_path=...`.

## Training

Single-node training entry point:

```bash
python tokendrive/train_v3.py experiment=td_v3 \
    data_module.data_path=/path/to/simlingo_dataset
```

Useful overrides:

```bash
python tokendrive/train_v3.py experiment=td_v3 \
    data_module.data_path=/path/to/simlingo_dataset \
    data_module.batch_size=8 \
    data_module.num_workers=4 \
    gpus=8 \
    name=td_v3_smooth_seed42
```

The main config is [`tokendrive/config/experiment/td_v3.yaml`](tokendrive/config/experiment/td_v3.yaml).

## Pretrained Checkpoint

Large model files are not stored in this GitHub repository. Download the released checkpoint from Hugging Face:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="<HF_ORG_OR_USER>/TokenDrive",
    filename="checkpoints/best.ckpt/checkpoint/mp_rank_00_model_states.pt",
    local_dir="pretrained",
)
PY
```

Expected checkpoint layout:

```text
pretrained/
└── checkpoints/
    └── best.ckpt/
        ├── latest
        └── checkpoint/
            └── mp_rank_00_model_states.pt
```

The `pretrained/checkpoints/best.ckpt/latest` pointer file is included for DeepSpeed checkpoint loading.

## Evaluation

Set paths for CARLA, Bench2Drive, and the checkpoint:

```bash
export TOKENDRIVE_REPO_ROOT=$PWD
export CARLA_ROOT=/path/to/carla0915
export TOKENDRIVE_CHECKPOINT=pretrained/checkpoints/best.ckpt
export TOKENDRIVE_EVAL_OUTPUT=eval_output
```

Run Bench2Drive evaluation:

```bash
python start_eval_tokendrive_v3.py
```

The launcher expects Bench2Drive assets under:

```text
$TOKENDRIVE_REPO_ROOT/Bench2Drive/
```

Evaluation outputs are written to `$TOKENDRIVE_EVAL_OUTPUT`.

## Ablations

See [`tokendrive/ablation/README.md`](tokendrive/ablation/README.md) for ablation and token-intervention analysis commands.

## Large Files

Do not commit checkpoints, ONNX exports, datasets, CARLA assets, Bench2Drive assets, logs, or evaluation outputs to GitHub. GitHub blocks normal Git uploads above 100 MiB, and these artifacts are better hosted separately, for example on Hugging Face.

## License

This repository is released under the Apache License 2.0. See [`LICENSE`](LICENSE).

TokenDrive reuses and modifies portions of the SimLingo codebase and follows its SimLingo-style data conventions. SimLingo is released under Apache License 2.0; the preserved license text is provided at [`licenses/SimLingo-LICENSE.txt`](licenses/SimLingo-LICENSE.txt), and attribution details are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The upstream SimLingo repository is https://github.com/RenzKa/simlingo.

The SimLingo dataset license is separate from the SimLingo code license. Dataset files are not redistributed in this repository.
