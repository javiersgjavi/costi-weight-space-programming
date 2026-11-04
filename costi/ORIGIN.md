# About this folder

This is the original CoSTI repository, [javiersgjavi/CoSTI](https://github.com/javiersgjavi/CoSTI), at commit [`3621926`](https://github.com/javiersgjavi/CoSTI/tree/362192650eef40aea3dea45c9e1d5fe1b920db2f). It is kept here, separate from the weight-space work, so it is always clear what belongs to the published model and what was added in this study. Its own documentation is in [`README.md`](README.md), and its licence (MIT) in [`LICENSE`](LICENSE).

**Nothing in the model, configs or training code was changed.** The only differences from upstream are:

1. `src/nn/blocks/bimamba/modules/mamba_simple.py`, lines 16 and 22: the fallback assignments used when the optional CUDA packages (`causal_conv1d`, `mamba_ssm`) are missing had the wrong number of `None`s, so the import crashed on CPU. They now unpack correctly. This affects importing, not computation.
2. Line endings and a final newline in some files. The bytes of `src/` match the SHA-256 hashes recorded in [`../results/evidence/v4_freeze_before_test.json`](../results/evidence/v4_freeze_before_test.json), so the evaluated code can be verified.

The study's tools in [`../weight_space/`](../weight_space/) import this code as `src.*`. `weight_space/__init__.py` adds this folder to `sys.path`.

The training scripts (`scripts/run_*`) were **not** run in this study. No CoSTI model was trained or fine-tuned.
