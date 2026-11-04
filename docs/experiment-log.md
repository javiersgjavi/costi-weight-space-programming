# Experiment log

Every weight edit the agent evaluated on the PhysioNet checkpoint, in the order it was tried, with its numbers and the reason it was kept or rejected. Recipes are in [`weight_space/edits/physionet/`](../weight_space/edits/physionet/). The numbers below come from [`results/data/hypotheses_128.csv`](../results/data/hypotheses_128.csv) and [`results/data/calibration_512_vs_v3.csv`](../results/data/calibration_512_vs_v3.csv).

How to read the tables:

- **% reduction** is the improvement over the *original* model on the same patients and noise. Higher is better; a negative value means worse than the original.
- An edit is **kept** only if it lowers *both* MAE and MSE relative to its *parent* (the previous kept version), not just relative to the original. That is why an edit with a large reduction can still be rejected.
- Different cohorts have very different absolute errors (the 128 confirmation patients contain large outliers), so compare percentages only within a column.

## 1. Screening on 128 + 128 patients

Calibration: 128 patients, seed 20261103. Confirmation: 128 different patients, mean of seeds 20261103 and 20261104. 100 imputations each.

| # | Hypothesis | Recipe | Calib. MAE | Calib. MSE | Conf. MAE | Conf. MSE | Decision |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Halve the random-graph branch | `pilots/random_graph_half` | −0.73% | −0.08% | — | — | **Rejected.** Worse than the original on both. Its purpose was to test whether the unregistered random graph matters. It does. |
| 2 | Output gain ×1.005, bias −0.012 | `head_affine` | 2.02% | 0.34% | 1.96% | 0.66% | **Kept** (first step towards V1) |
| 3 | + units 198 and 38 boosted by 1% | `affine_tail_readout` | 2.39% | 0.55% | 2.21% | 0.90% | **Kept → V1**. Tested on the full test set: −2.29% MAE, −0.52% MSE. |
| 4 | Bounded negative tail on unit 20 | `bounded_negative_tail` | 2.40% | 0.56% | 2.14% | 0.90% | **Rejected.** Both metrics worse than V1 on both confirmation seeds. |
| 5 | Mid-range bump on unit 198 (alone) | `local_positive_repair` | 2.53% | 0.98% | 2.13% | 0.96% | **Rejected alone.** MSE better than V1, MAE worse (worse on both seeds). |
| 6 | Longer memory in the backward Mamba branch | `reverse_memory_finite_gain` | 0.73% | −1.23% | — | — | **Rejected on calibration.** MSE worse than the original. |
| 7 | Attention prior towards observed positions | `mask_key_component` | 2.38% | 0.51% | — | — | **Rejected on calibration.** Below V1 on both. Mask and variable identity are entangled. |
| 8 | Sharpen unit 47 (alone) | `head47_sharp101` | 2.57% | 0.56% | 2.40% | 0.88% | **Rejected alone.** MAE better than V1, MSE slightly worse. |
| 9 | Bump on 198 + sharpen unit 47 | `midrange_gate47` | 2.72% | 0.99% | 2.32% | 0.95% | **Kept → V2.** Both better than V1. Passed the full 2,016-patient confirmation. |
| 10 | Shift unit 255 bias by −0.01 | `composed_gate255_translate` | 2.85% | 1.00% | 2.39% | 0.95% | **Kept → V3**, as the base for later edits. Better than V2 on the 128 confirmation patients. On the full 2,016: clearly better than V1 on both, better than V2 on MAE, but not on MSE (+0.002%, interval includes zero). It was therefore never exported or tested. |
| 11 | Constant prior on the mask query (LayerNorm identity) | `constant_mask_query_prior` | 2.43% | 0.59% | 2.24% | 0.78% | **Rejected.** Below its parent on both. |
| 12 | Contrast ramp between units 38 and 198 | `relative_tail` | 2.85% | 1.00% | 2.40% | 1.01% | **Kept → V4.** On the full 2,016 confirmation, both metrics are lower than V1, V2 and V3 on each seed. Against V1 both intervals are negative. The increments over V2 and V3 are within noise. Tested: −2.57% MAE, −0.61% MSE. |
| 13 | Tail correction on unit 167 (V5) | `gate167_tail` | **3.03%** | 1.12% | 1.89% | 0.93% | **Rejected.** Best of all on calibration, then clearly worse on confirmation. A textbook case of overfitting the diagnostic patients. |
| 14 | Shoulder correction on unit 212 (V7) | `gate212_shoulder` | — | — | 2.40% | 0.95% | **Not kept.** Confirmed against V3, but MSE worse than V4. Reused inside V14. |
| 15 | Mid-range correction on unit 50 (V9) | `gate50_midrange` | — | — | 2.41% | 1.02% | **Pending.** Full two-seed confirmation better than V4 on both metrics. Never tested. |
| 16 | Inactive-tail correction on unit 145 (V10) | `gate145_inactive_tail` | — | — | 2.41% | 1.02% | **Pending.** Failed the per-seed rule on the 128 patients, but its mean MSE was lower and both 512-calibration intervals were negative, so one full-confirmation run was allowed. Its first seed beat V4 on both metrics. The second was stopped by the shutdown. |
| 17 | Lobe correction on unit 67 (V12) | `gate67_shoulder` | — | — | 2.41% | 1.00% | **Rejected.** MSE worse than V4 on both confirmation seeds. |

Recipe names omit the `clinical_v0_` prefix and `.json` suffix. A dash means the stage was not run, either because the edit had already been rejected or because the later candidates were screened on 512 calibration patients instead (next table).

## 2. Later candidates on 512 calibration patients

After V3, single-cohort differences became too small to read on 128 patients, so new candidates were screened on 512 calibration patients against V3 (MAE 2.522218, MSE 1115.594 on this cohort). The intervals are paired patient-bootstrap 95% intervals for the difference (candidate − V3). Negative means better.

| Version | Hypothesis | MAE red. | MAE Δ 95% CI | MSE red. | MSE Δ 95% CI | Decision |
|---|---|---:|---|---:|---|---|
| V4 | Contrast ramp 38 / 198 | 0.0012% | [−0.00011, 0.00001] | 0.039% | [−1.34, 0.0004] | **Kept** (confirmed, then tested) |
| V5 | Tail correction on unit 167 | 0.260% | [−0.0126, −0.0021] | 0.011% | [−0.22, −0.045] | Rejected: worse on confirmation |
| V6 | Shift unit 12 | 0.075% | [−0.0022, −0.0016] | −0.003% | [−0.0004, 0.083] | Rejected: MSE worse on calibration |
| V7 | Shoulder on unit 212 | 0.005% | [−0.0002, −0.00003] | 0.0002% | [−0.0057, −0.0004] | Component only (see V14) |
| V8 | Mid-range on unit 21 | 0.004% | [−0.0002, 0.00001] | 0.028% | [−0.93, 0.0006] | Rejected: MSE worse than V4. Two-step feedback analysis (method §5). |
| V9 | Mid-range on unit 50 | 0.019% | [−0.0008, −0.0002] | 0.040% | [−1.36, 0.001] | Pending: confirmation favourable, never tested |
| V10 | Inactive tail on unit 145 | 0.017% | [−0.0007, −0.0002] | 0.042% | [−1.43, −0.001] | Pending: one confirmation seed done |
| V12 | Lobe on unit 67 | 0.016% | [−0.0006, −0.0003] | 0.040% | [−1.37, −0.002] | Rejected: MSE worse on confirmation |
| V13 | Near-zero compensation on unit 36 | −0.0004% | [−0.0001, 0.0001] | 0.041% | [−1.39, 0.001] | Rejected: MAE worse than V4. It pushed up over-predicted O2Sat values. |
| V15 | Redistribute adaptive graph to stored graph | 0.009% | [−0.0004, −0.0001] | −0.043% | [−0.008, 1.39] | Rejected: MSE worse than V4, almost all through AST |
| V14 | Units 50 + 212 combined | 0.024% | [−0.0010, −0.0003] | 0.040% | [−1.36, −0.001] | Pending: calibration favourable, confirmation not run |

Version numbers follow the order in which candidates were frozen. V11 was reserved for a V9 + V10 composition. It was never built, because V10 failed the per-seed rule on the 128 confirmation patients.

## 3. Full test-set evaluations

Only two candidates were ever run on the test set (4,033 patients, 100 imputations, seeds 20261103 and 20261104):

| Candidate | MAE | MSE | vs original |
|---|---:|---:|---|
| Original | 2.480960 | 350.120368 | |
| V1 | 2.424104 | 348.309750 | −2.29% / −0.52% |
| V4 | 2.417163 | 347.977979 | −2.57% / −0.61% |
| V4 vs V1 | | | −0.29% / −0.10% (95% CI MAE [−0.0076, −0.0063], MSE [−0.52, −0.17]) |

Reports: [`v4_vs_original_test_report.json`](../results/evidence/v4_vs_original_test_report.json), [`v4_vs_v1_test_report.json`](../results/evidence/v4_vs_v1_test_report.json).

## Unfinished work

When the machine was shut down (night of 3 November 2026), the queue held:

- **V9** (unit 50): full two-seed confirmation better than V4 on both metrics. Next step: test.
- **V10** (unit 145): first full-confirmation seed better than V4 on both metrics (MAE 2.47976, MSE 565.873). Second seed pending.
- **V14** (units 50 + 212): beats both of its parts on 512 calibration patients. Confirmation and test pending.

None of these is part of the reported result. If any of them is tested later, that will be a third use of the test set.

The user also asked whether continuing with ordinary SGD from V4 would now go further. The agent's answer: it is worth a *controlled* test. Train both the original and V4 with the same optimizer, steps, data order and seeds, and compare. Otherwise any gain could simply come from training longer. That experiment has not been run.

## Earlier pilots (not part of the result)

- **All configurations.** The session started on AQI-36, METR-LA (point and block) and PhysioNet. PEMS-BAY was skipped because its data file was not available locally. A backward-memory edit (`pilots/backward_slow_modes_x2`) made AQI-36 worse. Two AQI-36 output edits (`pilots/aqi_v0_*`) did not hold up on confirmation windows. On PhysioNet the output gain/bias edit (#2 above) was already confirmed at this stage.
- **METR-LA, point missing, version_0.** At the user's request the agent then focused on one traffic checkpoint. The final METR-LA edit (`pilots/metr_point_v0_graphhalf_bounded_with206`, 8,966 values) halves the random-graph branch and programs bounded corrections for three output neurons, using six idle neurons. On the full test set (285 windows, 2 seeds), MAE went from 1.753983 to 1.753537 (−0.025%) and MSE from 8.999162 to 8.995568 (−0.040%), but the MSE interval included zero. The gain was negligible, so the user asked which checkpoint looked most promising, and the agent chose PhysioNet.
