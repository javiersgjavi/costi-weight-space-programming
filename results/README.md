# Results

## `figures/`

Rebuild them all with `python scripts/make_figures.py`. Figures 1–6 are 2400×1350 px (16:9), sized for X, LinkedIn and slides. Figure 7 is tall, for the README.

| File | Shows |
|---|---|
| `01_headline.png` | the headline numbers |
| `02_read_the_weights.png` | effective weight of each of the 256 output-head neurons; 176 are idle |
| `03_write_the_weights.png` | how three idle neurons fix unit 198: its error by input, the three switches, and their sum |
| `04_search.png` | all 17 hypotheses on the 128-patient screening cohorts: kept, rejected, pending |
| `05_progress.png` | MAE and MSE reduction at each of the 17 iterations, with the accepted model as a staircase and the final test numbers |
| `06_where_the_gain_comes_from.png` | gain by variable group and by version (V1 vs V4) |
| `07_per_variable.png` | MAE and MSE change for each of the 40 variables |

## `data/`

| File | Contents |
|---|---|
| `test_summary.csv` | test MAE/MSE for the original model, V1 and V4, with patient-bootstrap intervals |
| `hypotheses_128.csv` | every hypothesis: calibration and confirmation reductions vs the original model (128 + 128 patients) |
| `calibration_512_vs_v3.csv` | later candidates on 512 calibration patients vs V3, with 95% intervals |
| `per_variable_test.csv` | test MAE/MSE per variable, original vs V4 |
| `per_group_test.json` | test MAE/MSE for physiological vs context variables, with each group's share of the reduction |
| `unit198_residuals_calibration.csv` | V1's median error by range of unit 198's input, on 128 calibration patients (the diagnosis behind the bump) |
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
