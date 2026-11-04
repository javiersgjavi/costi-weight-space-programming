# Reproducing

There are three levels, from "no downloads" to "re-run the test set".

## 1. Rebuild the figures (no data, no GPU)

Every figure in `results/figures/` is drawn from the CSV/JSON files in `results/data/`:

```bash
pip install -r requirements.txt
python scripts/make_figures.py
```

## 2. Inspect the edit itself (checkpoint needed, no GPU)

Download the released CoSTI checkpoints and the preprocessed PhysioNet data with the upstream script, which uses `gdown`:

```bash
cd costi && bash download_data.sh && cd ..
# -> costi/weights/ctimba_<dataset>/version_<n>/*.ckpt
# -> costi/data/mimic-iii_challenge/{train,val,test}/x.npy
```

Apply the V4 recipe to the original weights and confirm that it touches 965 values:

```bash
python - <<'EOF'
import json
from pathlib import Path
from weight_space.run import Inference

ckpt = next(Path("costi/weights/ctimba_mimic-iii-challenge_point/version_0").glob("*.ckpt"))
edit = json.loads(Path("weight_space/edits/physionet/clinical_v0_relative_tail.json").read_text())
model = Inference(ckpt, replicate=0, edit=edit)
for change in model.changes:
    print(change["key"], change["kind"], change["changed_values"])
EOF
```

The per-operation counts add up to 966, because the output bias is touched by two operations. There are 965 distinct values.

Recompute the effective output weights shown in figure 2. This needs an exported V4 checkpoint; see level 3 or `python -m weight_space.export --help`.

```bash
python scripts/export_head_coefficients.py \
    --original costi/weights/ctimba_mimic-iii-challenge_point/version_0/*.ckpt \
    --edited   path/to/v4/weights.ckpt
```

## 3. Re-run the evaluation (GPU recommended)

The harness expects the data under `<data>/clinical/`, so link it once:

```bash
mkdir -p artifacts/data
ln -s ../../costi/data/mimic-iii_challenge artifacts/data/clinical
```

Then run the paired test evaluation for one seed. It writes one result for the original model and one for the edited model, with the same masks and noise:

```bash
python -m weight_space.paired_head \
    --dataset mimic-iii-challenge_point --replicate 0 --split test \
    --samples 100 --seed 20261103 \
    --weights costi/weights --data artifacts/data \
    --edit weight_space/edits/physionet/clinical_v0_relative_tail.json \
    --output-original artifacts/results/original_test_seed1 \
    --output-edited   artifacts/results/v4_test_seed1
```

Repeat with `--seed 20261104`. Use `--split calibration` or `--split confirmation` (with `--limit 128` for the screening cohorts) to reproduce the validation stages. `python -m weight_space.run --help` evaluates a single model, and `python -m weight_space.compare --help` computes paired patient-bootstrap intervals between two results.

Numbers to expect, averaged over the two seeds: original MAE 2.480960 and MSE 350.120368; V4 MAE 2.417163 and MSE 347.977979. The original runs used a CUDA GPU with the fused Triton kernel (`--backend triton`, the default for `paired_head`). The CPU reference backend (`--device cpu --backend reference`) agrees within floating-point tolerance, but it is very slow for 4,033 patients × 100 samples.

### Checks

```bash
pytest weight_space/test_reference.py     # 18 tests; needs the data linked as above
```

## What is not included

- **The edited checkpoint and the patient-level predictions**, because of their size and to avoid redistributing derived patient-level data. The recipe plus the original checkpoint regenerate the edited weights exactly.
- **The agent's intermediate artifacts** (activation dumps, plans, logs). Many one-off scripts in `weight_space/` read or write `artifacts/...` paths from the original workspace. They are kept as a faithful record of what the agent ran, but they are not all runnable from a fresh clone.

Tested with Python 3.12, torch 2.12, numpy 2.4, pandas 2.2, einops 0.8 and triton 3.7.
