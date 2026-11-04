# `weight_space/`: the agent's toolkit

Everything the agent wrote to read, edit and evaluate CoSTI's weights. No file here computes a gradient or runs an optimizer: models are loaded with `requires_grad_(False)` under `torch.no_grad()`, and edits are explicit tensor operations stored as JSON.

The CoSTI model code is imported from [`../costi/`](../costi/) (see `__init__.py`).

## Core harness (reusable)

| File | What it does |
|---|---|
| `run.py` | Loads a checkpoint, applies an edit recipe (`apply_edit`) and runs paired, seeded CoSTI inference. CLI: `python -m weight_space.run --help`. |
| `paired_head.py` | Faster paired evaluation for head-only edits: runs the backbone once and the original and edited heads side by side. Used for the test runs. |
| `data.py` | Rebuilds CoSTI's datasets, splits, masks and scaler exactly as the original pipeline does, without importing the training stack. |
| `checkpoints.py` | Loads checkpoints with PyTorch's restricted unpickler, so no training objects are executed. |
| `reference_mamba.py` | Device-independent reference implementation of the BiMamba-v2 inner function. |
| `cuda_scan.py` | Forward-only fused Triton kernel for the selective scan (GPU speed-up). |
| `compare.py` | Paired comparison of two results with a patient-level bootstrap. |
| `summarize_test.py`, `clinical_partition_report.py`, `evaluation_audit.py` | Aggregate seeds, report intervals, audit stored measurements. |
| `export.py`, `export_hdf.py` | Export a validated edited checkpoint, keeping the original untouched. |
| `audit.py` | Inspects all released tensors (finiteness, statistics). |
| `batch.py`, `paired_batch.py` | Run a fixed, reviewable list of evaluation jobs. |
| `test_reference.py` | 18 tests: CPU reference vs upstream equations, Triton vs CPU, edits, graph factors. |
| `protocol.json` | The pre-registered evaluation protocol for the PhysioNet study. |

## Reading the model (diagnostics)

`diagnose_*.py` and `inspect_*.py` record activations of specific neurons on **calibration** patients and relate them to errors, for example `inspect_clinical_head.py` (GELU paths of the head), `diagnose_clinical_head_both_steps.py` (both denoising steps, median-selected sample), `diagnose_clinical_mask_signal.py` (mask direction through the conditioning), `diagnose_memory.py` (Mamba timescales) and `diagnose_attention.py`. None of them read test data.

## Writing the model (edit builders)

`build_clinical_*.py` each turn one diagnosis into one frozen recipe in `edits/physionet/`, together with an audit file describing its construction. For example:

| Builder | Recipe | Version |
|---|---|---|
| `build_clinical_midrange.py` | `clinical_v0_local_positive_repair.json` (bump on 198) | part of V2 |
| `build_clinical_gate.py` | `clinical_v0_head47_sharp101.json` (unit 47) | part of V2 |
| `build_clinical_composed_head.py` | `clinical_v0_midrange_gate47.json` | V2 |
| `build_clinical_gate_translation.py` | `clinical_v0_composed_gate255_translate.json` | V3 |
| `build_clinical_relative_tail.py` | `clinical_v0_relative_tail.json` | **V4, final** |

The remaining builders produced the rejected or pending candidates listed in [`../docs/experiment-log.md`](../docs/experiment-log.md).

## Verification and orchestration

`verify_clinical_source_semantics.py` checks the harness against functions extracted from the original CoSTI code. `verify_clinical_query_prior.py` checks one attention edit. `finalize_clinical*.py`, `export_clinical_after_test.py`, `test_clinical_after_queue.py`, `confirm_clinical_after_scheduler.py`, `clinical_validation_sequence.py` and `after_plan.py` queue frozen jobs and write the final audit. `plot_clinical_*.py` drew the agent's own progress plots, which the figures in `results/figures/` replace.

## Recipes

- `edits/physionet/`: all 21 PhysioNet recipes, kept, rejected and pending.
- `edits/pilots/`: recipes from the earlier METR-LA and AQI-36 pilot (see the experiment log).

A recipe is a list of operations on named tensors, for example:

```json
{"key": "out_proj.1.weight", "kind": "multiply", "value": 1.005}
{"kind": "scale_effective_columns", "key": "out_proj.0.mlp.3.weight",
 "readout_key": "out_proj.1.weight", "columns": [198, 38], "value": 1.01}
```

`add_values` operations carry the full dense delta of a tensor, which is why some recipes are large. Each recipe's `rationale` field contains the agent's own reasoning for the edit.

## Caveat

Most diagnostic, builder and orchestration scripts read and write paths under `artifacts/` in the original workspace (activation dumps, plans, logs), which are not distributed. They are kept as a faithful record of what the agent ran. The core harness is runnable from a fresh clone; see [`../docs/reproduce.md`](../docs/reproduce.md).
