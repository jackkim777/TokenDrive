# Pretrained weights

[Back to TokenDrive](../README.md) | [Evaluation guide](../docs/USAGE.md#4-closed-loop-evaluation)

Weight binaries are not included. A public TokenDrive checkpoint download URL has not yet been configured here. Obtain the checkpoint separately before evaluation; the pointer file alone is insufficient.

## Expected layout

```text
pretrained/
`-- checkpoints/
    `-- best.ckpt/
        |-- latest
        `-- checkpoint/
            `-- mp_rank_00_model_states.pt
```

The included `latest` file selects the DeepSpeed checkpoint tag. Keep the payload in the matching `checkpoint/` directory and preserve any additional shard files supplied with the release.

From the repository root:

```bash
export TOKENDRIVE_CHECKPOINT="$PWD/pretrained/checkpoints/best.ckpt"
```

See the [evaluation guide](../docs/USAGE.md#4-closed-loop-evaluation) for simulator, benchmark, and SLURM prerequisites. Weight binaries are excluded from Git by `.gitignore`.
