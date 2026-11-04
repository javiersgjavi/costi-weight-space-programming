# Results

## `figures/`

Rebuild them all with `python scripts/make_figures.py`. Figures 1–7 and 9 are 2400×1350 px (16:9), sized for X, LinkedIn and slides. Figure 8 is tall, for the README.

| File | Shows |
|---|---|
| `01_headline.png` | the headline numbers |
| `02_read_the_weights.png` | effective weight of each of the 256 output-head neurons; 176 are idle |
| `03_fix_unit_198.png` | how three idle neurons fix unit 198: its error by input, the three switches, their sum, and the error after the fix |
| `04_two_programs.png` | the bump and the ramp, each with its building blocks, plus the full V4 recipe |
| `05_search.png` | all 17 hypotheses on the 128-patient screening cohorts: kept, rejected, pending |
| `06_progress.png` | MAE and MSE reduction at each of the 17 iterations, with the accepted model as a staircase and the final test numbers |
| `07_where_the_gain_comes_from.png` | gain by variable group and by version (V1 vs V4) |
| `08_per_variable.png` | MAE and MSE change for each of the 40 variables |
| `09_neuron_roles.png` | follow-up: what units 47, 255, 198 and 38 respond to, and why "high" on the model's scale is not extreme for the patient |

## `data/`

| File | Contents |
|---|---|
| `test_summary.csv` | test MAE/MSE for the original model, V1 and V4, with patient-bootstrap intervals |
| `hypotheses_128.csv` | every hypothesis: calibration and confirmation reductions vs the original model (128 + 128 patients) |
| `calibration_512_vs_v3.csv` | later candidates on 512 calibration patients vs V3, with 95% intervals |
| `per_variable_test.csv` | test MAE/MSE per variable, original vs V4 |
| `per_group_test.json` | test MAE/MSE for physiological vs context variables, with each group's share of the reduction |
| `unit198_bump_before_after_calibration.csv` | median and mean error by range of unit 198's input z₁₉₈, for V1 and for V1 plus the bump only, on the 128 calibration patients |
| `clinical_neuron_concepts_summary_20261105.json` | follow-up aggregate: correlations with signal properties, input nudges, last-step ablations, scale percentiles (no patient-level rows) |
| `output_head_coefficients.csv` | for each of the 256 head neurons: effective output weight, bias and input norm, original vs V4, and its role in the edit |

## `evidence/`

Records written by the agent during the study. The only change is that the home directory in absolute paths has been replaced with `<HOME>`; the paths still point into the original workspace.

| File | Contents |
|---|---|
| `v4_freeze_before_test.json` | V4's recipe hash, parent and source-code hashes, recorded **before** its evaluation |
| `v4_test_decision.json` | the decision to keep V4 after the test |
| `v4_vs_original_test_report.json` | full test report, V4 vs original, per seed, with intervals |
| `v4_vs_v1_test_report.json` | same, V4 vs V1 |
| `v4_verification.json` | audit of the exported checkpoint: state, graph factors, reload parity, recomputed metrics |
| `parameter_counts.json` | the 2,406,657 registered parameters, buffers and unregistered graph values |
| `clinical_concepts_forward_plan_20261105.json` | follow-up: the paired V1/V4 forward plan, fixed before running it |
| `clinical_concept_ablation_verification_20261105.json` | follow-up: check of the last-step ablation against a real activation intervention |
| `clinical_concepts_provenance_20261105.json` | follow-up: hashes of the private inputs behind the aggregate |
