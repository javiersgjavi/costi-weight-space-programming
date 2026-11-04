# Method: how the 965 values were chosen

This is the technical companion to the [README](../README.md). It explains the algebra the agent used to read CoSTI's output head, the exact function behind each edit, the evaluation protocol, and what was checked along the way.

All numbers refer to the PhysioNet Challenge 2019 checkpoint `ctimba_mimic-iii-challenge_point/version_0` (SHA-256 `a012e3bc…a7f1c2`).

---

## 1. Where CoSTI makes its prediction

CoSTI is a consistency model. One denoising step takes a noisy version `x_σ` of the series and returns

```
x̂ = c_skip(σ) · x_σ + c_out(σ) · F(c_in(σ) · x_σ, conditioning, σ)
```

where `F` is the neural network and `c_skip`, `c_out`, `c_in` are fixed scalings of the noise level σ. Inference here uses two steps (σ = 80, then σ = 20.92), 100 independent samples per patient, and the lower-central median of the samples as the imputation. Observed values are always kept as they are. Only the hidden ones come from the model.

The last block of `F` is the **output head**, applied independently to every (time step, variable) position:

```
h ∈ ℝ⁶⁴ ─► W_in (256×64), b_in ─► GELU ─► W_mid (64×256), b_mid ─► r (1×64), b_out ─► y ∈ ℝ
```

(`model.out_proj.0.mlp.0`, `.mlp.3` and `model.out_proj.1` in the checkpoint; the dropout in between is inactive at inference.)

## 2. Reading the head: one number per neuron

Because `W_mid` and `r` are consecutive linear maps, they compose exactly:

```
y = r·(W_mid · GELU(W_in h + b_in) + b_mid) + b_out
  = b_eff + Σⱼ aⱼ · GELU(zⱼ)

zⱼ    = w_in,j · h + b_in,j        (pre-activation of hidden neuron j)
aⱼ    = r · W_mid[:, j]            (effective output weight of neuron j)
b_eff = r · b_mid + b_out
```

This identity is exact. It turns 256×64 + 64×256 + 64 parameters into 256 interpretable scalars `aⱼ`, each saying how strongly neuron j pushes the head output when it fires. [`results/data/output_head_coefficients.csv`](../results/data/output_head_coefficients.csv) lists them for the original and V4 models. [`scripts/export_head_coefficients.py`](../scripts/export_head_coefficients.py) recomputes them from the checkpoints.

What the weights alone show:

- **176 of 256 neurons are idle**: |aⱼ| < 0.01, median |aⱼ| ≈ 5·10⁻⁵. Most of them also have small input rows (median norm 0.34, against 1.2–6 for active neurons). A handful, such as units 7 and 152, have large inputs and a near-zero output weight. They have almost no influence on the output and can be reused.
- **About 80 neurons do the work.** The largest are unit 115 (a = +5.26), units 113 and 89 (≈ +2.57) and unit 221 (−2.56). The neurons the agent edited are moderately strong: 198 (+1.48), 38 (+1.37), 47 (−1.67), 255 (−1.40).

### Reading with data

The weights tell you what a neuron *can* do. The agent then ran the model on calibration patients and recorded, for every scored entry, `zⱼ` for the relevant neurons in **both** denoising steps, the sample that ended up as the median, and the normalized error. Errors were then grouped by neuron activation range. Every edit below starts from one of these observations.

## 3. The edits

Each edit is a JSON recipe of explicit tensor operations (`add`, `multiply`, `scale_effective_columns`, `add_values`), applied by [`weight_space/run.py:apply_edit`](../weight_space/run.py). The coefficients were set from the weights and calibration diagnostics and frozen before the edit's first evaluation. No coefficient was searched, swept or fitted to errors.

### 3.1 Output gain and bias (part of V1)

```
r     ← 1.005 · r
b_out ← 1.005 · b_out − 0.012
```

so the head output becomes `y' = 1.005 · y − 0.012`. On calibration, predictions in the low and middle normalized range were slightly too high and those in the high range too low: a small compression plus an offset. A gain above 1 expands, and a negative bias shifts down.

This edits the *network output F*, not the final imputation. Because `x̂ = c_skip · x_σ + c_out · F`, the change reaches the prediction through the consistency scaling and then through the second denoising step. It is **not** the same as recalibrating the final predictions afterwards.

### 3.2 Boost units 198 and 38 by 1% (rest of V1)

On calibration, entries where units 198 or 38 were active (z > 0) were still underestimated after 3.1, across 119 and 90 patients respectively, and almost all of them were measured physiological variables. The agent increased both effective output weights by 1%.

To change only `aⱼ`, the edit moves column j of `W_mid` along `r`, the smallest change that does it (operation `scale_effective_columns`):

```
W_mid[:, j] ← W_mid[:, j] + rᵀ · (s − 1) · aⱼ / ‖r‖²,    s = 1.01
```

V1 (3.1 + 3.2) changes **193 values** and is the version that was tested first.

### 3.3 A bump on unit 198 (part of V2)

After V1, the median-sample activations still showed unit 198 underestimating while its input was moderately positive (median normalized error −0.013, −0.033 and −0.049 in bins z ∈ [0, 0.5), [0.5, 1), [1, 2), across 93, 81 and 96 patients) and overestimating above z = 2 (+0.022). The fix is a local bump that rises over (0, 1.5), falls back to zero at 2 and stays off beyond.

**How it is built.** With a steep input, GELU is almost a ReLU: `GELU(16u)/16 ≈ max(0, u)`. Three idle neurons (246, 226, 179) get:

```
input row   w = 16 · w_in,198
bias        b = 16 · (b_in,198 − t),          t = 0, 1.5, 2
output      a = (k/16) · (+1, −4, +3),        k = 0.04 · a₁₉₈
```

so together they add

```
k · [ ReLU(z₁₉₈) − 4·ReLU(z₁₉₈ − 1.5) + 3·ReLU(z₁₉₈ − 2) ]
```

a triangle that peaks at z₁₉₈ = 1.5 with height 1.5k ≈ 0.09 (in normalized head-output units) and is exactly zero for z₁₉₈ ≤ 0 and z₁₉₈ ≥ 2. The slope k is a fixed 4% of unit 198's own effective weight.

**Alone it failed:** on confirmation it improved MSE but worsened MAE.

### 3.4 Sharpen unit 47 (part of V2)

Unit 47 is a negative-only path: on the traces examined, its input is mostly negative. GELU has a small negative dip (minimum ≈ −0.17 at z ≈ −0.75), and multiplying it by a₄₇ = −1.67 turns the dip into a **positive** bump in the output. The edit:

```
input row and bias of 47  × 1.01
effective output weight   ÷ 1.01
```

The term becomes `(a/1.01) · GELU(1.01 z)`. For large positive z it has the same slope as before, since GELU(u) ≈ u there. In the negative region the dip gets slightly narrower and shallower, which reduces the positive contribution there.

**Alone it failed:** on confirmation it improved MAE but slightly worsened MSE. **Combined with 3.3 it improved both**, and the combination became V2. The two edits use disjoint neurons, so they do not interact inside the head.

### 3.5 Shift unit 255 (V3)

All calibration pre-activations of unit 255 were negative, again inside the GELU dip that a negative a₂₅₅ turns into a positive bump. The residual changed sign within that range: median +0.0067 for z ∈ [−1.5, −1) and −0.0020 for z ∈ [−1, −0.5), each across 127 patients. Moving the bump instead of shrinking it fits that pattern:

```
b_in,255 ← b_in,255 − 0.01
```

This was one fixed value. The association was stronger on context variables and mixed on physiological ones, so it does not show a single function for unit 255.

### 3.6 A capped ramp on z₃₈ − z₁₉₈ (V4)

Units 38 and 198 are the two positive paths edited in 3.2. Their input weights are partly aligned (cosine 0.69), and normally z₃₈ < z₁₉₈. On 512 calibration patients, the reversal **z₃₈ > z₁₉₈** happened in only 15 scored entries from 11 patients. Their errors were negative on average, but 6 of the 15 were already over-predicted. That is weak, mixed evidence, so the edit was kept small and capped, and accepted only after full evaluations.

Two idle neurons (181, 150) compute a capped ramp of the contrast d = z₃₈ − z₁₉₈:

```
input rows   w = 64 · (w_in,38 − w_in,198)
biases       b = 64 · (b_in,38 − b_in,198),   64 · (b_in,38 − b_in,198 − 1)
outputs      a = +A/64, −A/64,                A = 0.05 · a₃₈ = 0.0696
```

```
A · [GELU(64 d) − GELU(64 (d − 1))] / 64  ≈  0.0696 · clip(d, 0, 1)
```

The factor 64 keeps GELU's small negative dip close to the threshold. Capping the correction *inside the head* does not cap its effect on the final imputation, because of the two-step feedback described in §5.

### 3.7 What changed, in total

| Tensor | Values changed | Edits |
|---|---:|---|
| `out_proj.1.weight` (r) | 64 | 3.1 |
| `out_proj.1.bias` | 1 | 3.1 |
| `out_proj.0.mlp.3.weight` (W_mid) | 509 | 3.2, 3.3, 3.4, 3.6 |
| `out_proj.0.mlp.0.weight` (W_in) | 384 | 3.3, 3.4, 3.6 (6 rows) |
| `out_proj.0.mlp.0.bias` | 7 | 3.3, 3.4, 3.5, 3.6 |
| **Total** | **965** of 2,406,657 registered parameters (0.0401%) | |

Three of the 512 affected `W_mid` entries happen to keep their value, which is why that row shows 509. The model also has 40 buffer values and 800 unregistered graph values, none of which count in the denominator ([`parameter_counts.json`](../results/evidence/parameter_counts.json)).

The algebra tells us exactly *what function* each edit adds to the head. It does not tell us *how much of the final improvement* each one is responsible for: no per-edit ablation was run on test, and functions that add up inside the head do not add up after two denoising steps and a median over 100 samples.

## 4. Evaluation protocol

| Set | Patients | Used for |
|---|---:|---|
| Calibration | 2,017 (even slots of the original validation split) | looking at activations, designing edits; first screens on 128, later on 512 patients |
| Confirmation | 2,016 (odd slots) | accepting or rejecting frozen edits; 128-patient screens, then the full 2,016 |
| Test | 4,033 | final evaluation only |

- **One 48-hour window = one patient.** Bootstrap intervals resample patients (paired, 95%). The two seeds are averaged per patient first, so the intervals do not include seed-to-seed variation.
- **Paired randomness.** Original and edited models use identical missing masks, per-patient seeds and noise. The original CoSTI mask generator, scaler (including a quirk of its implementation) and interpolation are reproduced exactly.
- **Freeze before evaluation.** Each candidate's recipe hash, parent and decision rule were written down before its first accuracy measurement ([example](../results/evidence/v4_freeze_before_test.json)).
- **Acceptance rule.** Both MAE and MSE must go down relative to the parent version. A MAE-only gain never counts. This was the user's explicit choice when the first METR-LA test showed MAE down and MSE up.
- **Test history.** V1 was evaluated on the full test set first. V4 is the second, and last, candidate evaluated there. No test example or test result was used to build any edit. The cost of the second evaluation is selection, not leakage: the reported V4 is the better of two tested versions, so its margin over V1 is less certain than its margin over the original.

## 5. What made this hard

- **Two-step feedback.** A head edit changes the output of the first step, which is the input of the second step, which uses the same head again. It can also change which of the 100 samples is the median. In the two AST cases studied for a unit-21 edit, intervening in the first step alone reproduced the whole harmful effect, while intervening in the last step alone barely changed the prediction.
- **Entangled representations.** An edit on unit 36 looked right when activations were grouped, but it pushed up O2Sat values that were already over-predicted. MSE improved slightly and MAE got worse, so it was rejected.
- **Mask and content share directions.** In attention, the signal "this position was observed" is mixed with variable identity. An exact linear extraction of a pure mask signal was impossible for this construction and domain. That does not rule out non-linear or coordinated edits.
- **Memory asymmetry is not a bug.** The backward Mamba branch keeps information for far less time than the forward one. Lengthening it analytically worsened MAE by 1.70% and MSE by 1.79% relative to V1 on calibration.
- **MSE lives in the extremes.** Redistributing adaptive-graph weight towards the stored graph helped MAE but hurt MSE, almost entirely through AST.
- **Unregistered graph factors.** CoSTI's adaptive graph factors are not in the `state_dict`, so reloading a checkpoint does not restore the trained graph exactly. The harness regenerates them by replaying the original experiment's model-initialisation seed and random draws, freezes them, and the exported checkpoint stores them explicitly (`weight_space_graph_factors`). This is a reproducibility fix, not an accuracy gain.

## 6. How the harness was verified

The agent rewrote CoSTI's inference path so that it could run thousands of paired evaluations quickly. Before any result was trusted:

- **CPU reference vs upstream.** The BiMamba selective scan and inner function were checked against the reference equations extracted from the CoSTI source ([`test_reference.py`](../weight_space/test_reference.py), 18 tests).
- **GPU kernel vs CPU reference.** The forward-only Triton kernel ([`cuda_scan.py`](../weight_space/cuda_scan.py)) and the CUDA-graph replay match the CPU reference. No backward pass is implemented anywhere.
- **Harness vs original CoSTI.** CoSTI's own `forward_cm`, `_impute_batch`, scheduler scalings, interpolation and scoring-mask functions were extracted and run on two calibration patients with the same noise. Normalized predictions agreed within 2.5·10⁻⁵, and interpolation and masks matched exactly ([`verify_clinical_source_semantics.py`](../weight_space/verify_clinical_source_semantics.py)).
- **Shared-backbone head evaluator.** For head-only edits, [`paired_head.py`](../weight_space/paired_head.py) runs the shared backbone once and the original and edited heads side by side. It produced exactly the same predictions as two separate native runs.
- **Programmed functions.** Each compiled function (bump, ramp, tails) was checked numerically against its formula on arbitrary inputs. For example, the maximum discrepancy for one tail program after float32 storage was 1.7·10⁻⁸.
- **Export.** The final checkpoint's state and graph factors equal the frozen candidate's exactly, all test metrics were recomputed from the raw prediction arrays, and loading the exported file through the CLI reproduces the candidate's predictions exactly ([`v4_verification.json`](../results/evidence/v4_verification.json)).

## 7. Variables and what the metric rewards

The 40 PhysioNet columns are groups of *data columns*, not groups of neurons:

- **34 physiological:** vital signs (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2) and lab values (pH, lactate, creatinine, platelets…).
- **6 context:** Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.

No variable was prioritised: the goal was global MAE and MSE. Every scored entry weighs the same, so frequent columns weigh more. The six context columns make up about **46% of all scored entries**, and they are nearly constant within a patient, which makes them easy to correct. MSE also mixes variables with very different units, so a lab value with large magnitudes (AST, fibrinogen, platelets) dominates it. These are properties of the benchmark, not of the model.

Test breakdown for V4 versus the original:

| Group | MAE | MSE | Share of total MAE reduction | Share of total MSE reduction |
|---|---|---|---:|---:|
| 34 physiological | 4.1848 → 4.1728 (−0.29%) | 636.03 → 633.01 (−0.47%) | 10.3% | 76.5% |
| 6 context | 0.4529 → 0.3275 (−27.69%) | 9.813 → 8.713 (−11.21%) | 89.7% | 23.5% |

17 of the 40 variables improve on both metrics, 24 on MAE and 25 on MSE. O2Sat improves by 3.6% (MAE) and 4.8% (MSE). ICULOS improves MAE by 16.4% but worsens MSE by 5.8%. Sparse labs swing the most, for example pH (+76% MAE). Full table: [`per_variable_test.csv`](../results/data/per_variable_test.csv). These breakdowns are descriptive: they have no intervals of their own.

The practical reading is that **V4 is a better readout of representations CoSTI already had**. It has not been shown to improve the model's grasp of clinical dynamics. To target physiological variables, a next stage would have to make them an explicit objective.
