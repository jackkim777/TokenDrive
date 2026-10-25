# TokenDrive ablation study

[Back to TokenDrive](../../README.md) | [Setup and evaluation](../../docs/USAGE.md)


## Structure

```
tokendrive/ablation/
├── README.md                  ← this file
├── compute_delta_wp.py        ← computes Δ_wp token-intervention metric
├── aggregate_results.py       ← aggregates B2D + Δ_wp results → CSV + Markdown
├── delta_wp_results.json      ← (generated after running compute_delta_wp.py)
├── results_ablation.csv       ← (generated after running aggregate_results.py)
└── results_ablation.md        ← (generated after running aggregate_results.py)

tokendrive/config/experiment/
├── ablate_A1_no_acm.yaml
├── ablate_A2_token_agnostic.yaml
├── ablate_B1_single_pass.yaml
├── ablate_B2_lon_shortcut.yaml
├── ablate_C1_no_gt.yaml
└── ablate_C2_no_grad_mask.yaml
```

---

## Ablation Descriptions

| ID | Name | Change | What it verifies |
|----|------|--------|-----------------|
| **A1** | No-ACM | Force γ=1, β=0 (heads/losses retained) | Effect of FiLM conditioning on waypoints |
| **A2** | Token-agnostic ACM | Remove e_lat, e_lon from ACM input (context/HLC/speed retained) | Contribution of action tokens to FiLM |
| **B1** | Single-pass | Remove Pass-1; lon head uses v_pool directly, one ACM call | Necessity of two-pass design |
| **B2** | Lon-head shortcut | lon head input: e_ctx → v_pool | Effect of context-aware input from e_ctx |
| **C1** | No GT-conditioning | Teacher forcing fully disabled from epoch 0 | Training stability contribution of teacher forcing schedule |
| **C2** | No grad mask | Remove ACM detach for samples without labels | Effect of gradient masking on preserving conditioning signal |

---

## Training Commands

Run all ablations from the repository root.

```bash
cd TokenDrive

# A1: No-ACM
python tokendrive/train_v3.py experiment=ablate_A1_no_acm \
    name=ablate_A1_no_acm_seed42 seed=42

# A2: Token-agnostic ACM
python tokendrive/train_v3.py experiment=ablate_A2_token_agnostic \
    name=ablate_A2_token_agnostic_seed42 seed=42

# B1: Single-pass
python tokendrive/train_v3.py experiment=ablate_B1_single_pass \
    name=ablate_B1_single_pass_seed42 seed=42

# B2: Lon-head shortcut
python tokendrive/train_v3.py experiment=ablate_B2_lon_shortcut \
    name=ablate_B2_lon_shortcut_seed42 seed=42

# C1: No GT-conditioning
python tokendrive/train_v3.py experiment=ablate_C1_no_gt \
    name=ablate_C1_no_gt_seed42 seed=42

# C2: No gradient masking
python tokendrive/train_v3.py experiment=ablate_C2_no_grad_mask \
    name=ablate_C2_no_grad_mask_seed42 seed=42
```

For cluster training, wrap the command in a job script configured for your SLURM resources. No training SLURM script is bundled in this repository.

---

## Evaluation Commands (Bench2Drive Closed-Loop)

After training, select the checkpoint and a separate output directory through environment variables. The launcher does not parse `--checkpoint` or `--name` flags.

```bash
export TOKENDRIVE_CHECKPOINT=/path/to/ablate_A1_no_acm_seed42/last.ckpt
export TOKENDRIVE_EVAL_OUTPUT="$PWD/eval_results/ablate_A1_no_acm_seed42"
printf '1\n' > max_num_jobs.txt
python start_eval_tokendrive_v3.py
```

Follow the [evaluation prerequisites](../../docs/USAGE.md#4-closed-loop-evaluation) first. The launcher uses SLURM and writes per-route results under `tokendrive_v3/bench2drive/<seed>/res/` inside the selected output directory. Merge and score these with the tools supplied by your Bench2Drive version before aggregation.

---

## Δ_wp Metric Computation

```bash
# Single checkpoint
python tokendrive/ablation/compute_delta_wp.py \
    --checkpoint results/ablate_A1_no_acm_seed42/last.ckpt \
    --run_name ablate_A1_no_acm_seed42 \
    --data_path <simlingo_dataset> \
    --n_samples 256 \
    --seed 42 \
    --output_json tokendrive/ablation/delta_wp_results.json

# Entire checkpoint directory (ablate_* pattern)
python tokendrive/ablation/compute_delta_wp.py \
    --checkpoint_dir results/ablations/ \
    --data_path <simlingo_dataset> \
    --output_json tokendrive/ablation/delta_wp_results.json
```

---

## Result Aggregation

```bash
python tokendrive/ablation/aggregate_results.py \
    --b2d_dir    eval_results/ablations/ \
    --delta_wp   tokendrive/ablation/delta_wp_results.json \
    --output_csv tokendrive/ablation/results_ablation.csv \
    --output_md  tokendrive/ablation/results_ablation.md
```

If B2D results are entered manually into a CSV:
```bash
python tokendrive/ablation/aggregate_results.py \
    --input_csv  tokendrive/ablation/b2d_manual.csv \
    --delta_wp   tokendrive/ablation/delta_wp_results.json \
    --output_csv tokendrive/ablation/results_ablation.csv \
    --output_md  tokendrive/ablation/results_ablation.md
```

---

## Output Artifacts

1. `results_ablation.csv` — all metrics (DS, SR, Eff, Cmf, Δ_spd, Δ_rt, Δ_sum)
2. `results_ablation.md`  — Markdown table ready to paste into LaTeX paper
3. `delta_wp_results.json` — per-run per-(lat,lon)-pair Δ_wp details

---

## Notes

- Three seeds would be ideal; results are reported with **1 seed (seed=42)** due to compute constraints.
- Dataset, training schedule, and optimizer are identical to the full model.
- B1/B2 differ in lon_action_head input dim (256→512), so checkpoint structure differs from the full model.
- Checkpoint run name prefix: `ablate_<id>_seed<N>` (e.g., `ablate_A1_no_acm_seed42`)
