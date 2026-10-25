# Release checklist

[Back to TokenDrive](README.md)

Check items only after verifying them for the paper-matched release.

## Artifacts and provenance

- [ ] Upload checkpoint payloads and add their verified download URL to `README.md` and `pretrained/README.md`.
- [ ] Record checkpoint configuration, training seed, dataset version, and upstream SimLingo commit.
- [ ] Confirm encoder freezing and other defaults match the reported experiment.
- [ ] Confirm `LICENSE`, `licenses/SimLingo-LICENSE.txt`, and `THIRD_PARTY_NOTICES.md` are present.
- [ ] Keep datasets, simulator assets, benchmark assets, and weight binaries outside Git history.

## Reproduction

- [ ] Clone `https://github.com/jackkim777/TokenDrive.git` into a fresh directory.
- [ ] Create the environment and configure `simlingo_base_training` using `docs/USAGE.md`.
- [ ] Compose the V3 Hydra configuration and run label generation on a small data subset.
- [ ] Run a GPU training smoke check and verify checkpoint loading in the evaluation agent.
- [ ] Verify SLURM settings and complete a Bench2Drive route with the released checkpoint.
- [ ] Confirm metrics, evaluation seeds, and ablation artifacts match the paper.

## Publication

- [ ] Review the rendered README, diagrams, relative links, and commands on GitHub.
- [ ] Tag the verified commit, for example `v1.0-jcde`.
- [ ] Add verified code, model, and dataset links to the paper's Data Availability section.
