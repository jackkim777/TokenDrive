# Training and evaluation

[Back to TokenDrive](../README.md)

Commands below use Bash and run from the TokenDrive repository root unless stated otherwise. Training requires a compatible NVIDIA GPU environment; the supplied environment is a Linux Conda export.

## 1. Environment and upstream code

```bash
conda env create -f environment.yaml
conda activate tokendrive
export TOKENDRIVE_REPO_ROOT="$PWD"
```

Training and data utilities import `simlingo_base_training`, which is not bundled here. Obtain the [upstream SimLingo code](https://github.com/RenzKa/simlingo) and add its root to `PYTHONPATH`:

```bash
git clone https://github.com/RenzKa/simlingo.git ../simlingo
export SIMLINGO_CODE_ROOT="$(cd ../simlingo && pwd)"
export PYTHONPATH="$TOKENDRIVE_REPO_ROOT:$SIMLINGO_CODE_ROOT:${PYTHONPATH:-}"
python -c "import simlingo_base_training; print(simlingo_base_training.__file__)"
```

The upstream revision is not pinned in this release. Record the revision used (`git -C "$SIMLINGO_CODE_ROOT" rev-parse HEAD`) and check compatibility with the supplied environment. The module check above does not validate all transitive dependencies.

The encoder loads `OpenGVLab/InternViT-300M-448px` through Transformers. First use requires access to the model files or a populated local model cache.

## 2. Data preparation

Obtain the dataset separately under its upstream terms. Prepare this layout:

```text
simlingo_dataset/
|-- data/simlingo/
`-- commentary/simlingo/
```

Generate structured token labels before training:

```bash
export SIMLINGO_DATASET_ROOT=/path/to/simlingo_dataset
python tokendrive/data_generation/slots_generator.py
python tokendrive/data_generation/template_id_generator.py
```

These scripts generate labels in the dataset tree. The main experiment reads `SIMLINGO_DATASET_ROOT`, falling back to `simlingo_dataset`. You can also set `data_module.data_path` with a Hydra override.

## 3. Training

Inspect the resolved configuration without starting training:

```bash
python tokendrive/train_v3.py experiment=td_v3 --cfg job --resolve
```

This still imports training dependencies. It checks configuration composition, not dataset availability or GPU execution.

Start the main V3 experiment:

```bash
python tokendrive/train_v3.py experiment=td_v3 \
    data_module.data_path="$SIMLINGO_DATASET_ROOT"
```

The checked-in experiment uses 8 GPUs and DeepSpeed stage 2. Adjust resources for your host, for example:

```bash
python tokendrive/train_v3.py experiment=td_v3 \
    data_module.data_path="$SIMLINGO_DATASET_ROOT" \
    data_module.batch_size=8 \
    data_module.num_workers=4 \
    gpus=1 \
    strategy=auto \
    name=td_v3_smooth_seed42
```

| Setting | Main experiment default |
| :--- | :--- |
| Experiment | [`td_v3.yaml`](../tokendrive/config/experiment/td_v3.yaml) |
| Seed / epochs | 42 / 20 |
| Batch size per device | 8 |
| GPUs / strategy | 8 / `deepspeed_stage_2` |
| Speed / route smoothness weights | 0.1 / 0.1 |
| Weights & Biases | Disabled; enable with `enable_wandb=true` |

Hydra determines the run directory. Checkpoints are written to `./checkpoints` relative to that directory. The script saves periodic, last, and best validation-loss checkpoints.

Encoder freezing is controlled by `model.vision_model.freeze`; the encoder default is `false`. Set it explicitly when matching a particular training protocol.

## 4. Closed-loop evaluation

The launcher submits **SLURM jobs**. It expects `sbatch` and `squeue`, a configured Conda environment, CARLA 0.9.15, and Bench2Drive assets.

Place Bench2Drive at `$TOKENDRIVE_REPO_ROOT/Bench2Drive` with XML routes in `leaderboard/data/bench2drive_split/`, its evaluator at `leaderboard/leaderboard/leaderboard_evaluator.py`, and its `scenario_runner/` directory.

Prepare the [checkpoint layout](../pretrained/README.md), then configure paths:

```bash
export TOKENDRIVE_REPO_ROOT="$PWD"
export CARLA_ROOT=/path/to/carla0915
export TOKENDRIVE_CHECKPOINT="$TOKENDRIVE_REPO_ROOT/pretrained/checkpoints/best.ckpt"
export TOKENDRIVE_EVAL_OUTPUT="$TOKENDRIVE_REPO_ROOT/eval_output"
printf '1\n' > max_num_jobs.txt
```

`max_num_jobs.txt` sets the concurrency limit and is read by the scheduler loop. Before launching, review `partition_name` (default `main`), generated job resources, and port-based GPU assignment in [`start_eval_tokendrive_v3.py`](../start_eval_tokendrive_v3.py) for your cluster. Default evaluation seeds are 1, 2, and 3.

```bash
python start_eval_tokendrive_v3.py
```

The launcher reads environment variables; it does **not** implement `--checkpoint` or `--name` flags. Use a different `TOKENDRIVE_EVAL_OUTPUT` for each experiment.

```text
eval_output/tokendrive_v3/bench2drive/<seed>/
|-- res/       # Per-route result JSON
|-- out/       # Standard output logs
|-- err/       # Error logs
|-- run/       # Generated SLURM scripts
`-- viz/       # Agent visualization output
```

## 5. Troubleshooting

| Symptom | Check |
| :--- | :--- |
| `No module named simlingo_base_training` | Add the upstream SimLingo root to `PYTHONPATH` in the active shell. |
| Missing dataset files | Verify both dataset subtrees and `SIMLINGO_DATASET_ROOT`; generate labels first. |
| Model download fails | Check InternViT access and the local Transformers cache. |
| Missing checkpoint payload | The included `latest` file is only a pointer; obtain the actual weights separately. |
| Missing `max_num_jobs.txt` | Create the file in the repository root before evaluation. |
| `sbatch` / `squeue` unavailable | Run on a SLURM host with an appropriate partition. |

See [ablations](../tokendrive/ablation/README.md) for intervention analysis and the [release checklist](../RELEASE_CHECKLIST.md) before publishing a paper-matched release.
