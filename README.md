# Weight-space programming on CoSTI

**An AI agent improved a trained spatio-temporal imputation model by reading its weights and rewriting 965 of them by hand. It ran no training, used no gradients and added no data.**

![965 weights edited by hand, MAE −2.57%, MSE −0.61%](results/figures/01_headline.png)

| Test set (4,033 ICU patients) | Values changed | MAE | MSE |
|---|---:|---:|---:|
| CoSTI, original checkpoint | 0 | 2.4810 | 350.12 |
| **CoSTI, weight-space edited (V4)** | **965** (0.04%) | **2.4172** (−2.57%) | **347.98** (−0.61%) |

The 95% interval for the per-patient change is [−0.067, −0.061] for MAE and [−4.67, −0.60] for MSE. Both metrics improve on each noise seed separately. Please read the [caveats](#how-to-read-this-result) before quoting these numbers: most of the MAE gain comes from easy context columns, and the test set was used twice.

---

## Contents

1. [What is weight-space programming?](#what-is-weight-space-programming)
2. [The model and the task](#the-model-and-the-task)
3. [What the agent did](#what-the-agent-did)
   1. [Read the weights](#step-1--read-the-weights)
   2. [Write the weights](#step-2--write-the-weights)
   3. [Test every idea with forward passes](#step-3--test-every-idea-with-forward-passes)
4. [Results](#results)
5. [How to read this result](#how-to-read-this-result)
6. [What did not work](#what-did-not-work)
7. [Who did what](#who-did-what)
8. [Repository layout](#repository-layout)
9. [Reproducing](#reproducing)
10. [Credits and citation](#credits-and-citation)

---

## What is weight-space programming?

A trained neural network is usually treated as a black box: to change its behaviour you retrain it, and an optimizer such as Adam updates every weight a little, guided by gradients.

**Weight-space programming** is a different idea, introduced by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL) in [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/) (November 2026). A language model:

1. **reads** a network's weights and works out what each part computes,
2. **diagnoses** a specific failure,
3. **writes** new weight values that fix it, as an explicit, human-readable edit,
4. **checks** the edit with forward passes only.

There is no optimizer, no backpropagation and no gradient anywhere in the loop. The authors show this working on RL policies, small diffusion models, a Go network and Qwen3.5-0.8B, where removing one direction from two MLP layers fixed a reasoning-loop failure.

**This repository applies the idea to a real, published research model.** [CoSTI](https://github.com/javiersgjavi/CoSTI) is a consistency model for multivariate time-series imputation, published in *Knowledge-Based Systems* (2025). The question was simple: can an agent following the weight-space-programming recipe beat the released checkpoint on its own benchmark, under a strict protocol, without any training? On one checkpoint and one benchmark, the answer was yes, by a modest but statistically clear margin. The full story, including the failures, is below.

## The model and the task

**Imputation.** Hospital records are full of holes: heart rate every hour, a lab value once a day, some measurements never taken. Imputation means estimating the missing values from the observed ones, using how each variable evolves over time and how variables relate to each other.

**Benchmark.** [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/): 48-hour windows of 40 ICU variables per patient (34 vitals and labs, plus 6 context columns: age, gender, two ICU-unit flags, hospital admission time and ICU length of stay). In the *point missing* setting, 25% of the observed values are hidden at random, the model imputes them, and MAE and MSE are measured on those hidden values.

**CoSTI.** A consistency model: it learns to map a noisy version of the data straight back to clean data in one or two steps, instead of the hundreds of steps a diffusion model needs. Here each patient is imputed with **2 denoising steps** (noise levels σ = 80 and 20.92), **100 random samples**, and the median of the samples as the answer. The checkpoint is `ctimba_mimic-iii-challenge_point/version_0`, one of the five released by the CoSTI authors.

The CoSTI source lives, unmodified, in [`costi/`](costi/). See [`costi/ORIGIN.md`](costi/ORIGIN.md).

## What the agent did

### Step 1 · Read the weights

The agent started by auditing all released checkpoints, then concentrated on the last block of the network, the **output head**, which turns a 64-number internal representation into the predicted value of one variable at one time step:

```
h (64) ─► Linear 64→256 ─► GELU ─► Linear 256→64 ─► Linear 64→1 ─► y
```

The last two layers are linear with nothing in between, so they collapse into a single vector. The whole head can then be written exactly as

```
y = b + Σⱼ aⱼ · GELU(zⱼ),     zⱼ = wⱼ · h + bⱼ
```

Each of the 256 hidden neurons j has an input `zⱼ` (a weighted reading of the representation) and **one effective weight `aⱼ`**, which says how hard that neuron pushes the prediction up or down. This turns the head into something you can read like code.

![176 of the 256 output neurons contribute almost nothing](results/figures/02_read_the_weights.png)

Two facts came straight out of the weights:

- **176 of the 256 neurons are idle** (|aⱼ| < 0.01, with a median of 5·10⁻⁵). Training left them effectively disconnected from the output. They are free space.
- **A few neurons carry the output.** Units 198 and 38 are strong positive paths (a ≈ +1.48 and +1.37), and units 47 and 255 are strong negative ones (a ≈ −1.67 and −1.40).

The agent then ran the model on **calibration patients** (validation data, never test), recorded each neuron's activations in both denoising steps, and compared them with the errors. Three patterns showed up:

- The output was slightly compressed and offset: low and middle values came out too high, high values too low.
- When units 198 and 38 were active, the prediction was still too low.
- Several neurons left systematic errors in specific ranges of their input, for example unit 198 underestimating between z = 0 and 2 and overestimating above 2.

### Step 2 · Write the weights

Each diagnosis became one explicit weight edit. The coefficients were chosen from the weights and the calibration diagnostics and **fixed before the edit was ever evaluated**. Nothing was fitted or tuned by search.

The most interesting trick is how the agent **built new functions out of idle neurons**. A GELU with a very steep input acts like a ReLU, a clean on/off switch. To make an idle neuron switch on exactly when unit 198's input crosses some threshold t, the agent:

1. copies unit 198's input weights into the idle neuron, multiplied by 16 (or 64),
2. sets its bias so that it switches on at t,
3. gives it a tiny output weight.

Adding a few of these switches draws a shape. The two programmed shapes in the final model:

![Building small functions out of idle neurons](results/figures/03_write_the_weights.png)

- **A bump on unit 198** (units 246, 226, 179): it adds a little to the output while z₁₉₈ is between 0 and 2, peaks at 1.5 and switches off after 2. That is exactly the range where unit 198 left predictions too low.
- **A capped ramp on a contrast** (units 181, 150): normally z₃₈ < z₁₉₈. In the rare cases where unit 38 overtakes 198, the ramp adds 0.070 · clip(z₃₈ − z₁₉₈, 0, 1).

The complete final recipe (V4) has six edits, applied in four validated versions:

| Version | Edit | Why |
|---|---|---|
| V1 | Output layer: gain ×1.005, bias −0.012 | undo the compression and offset of the output |
| V1 | Units 198 and 38: effective output weight +1% | these active paths were still underestimating |
| V2 | Bump on unit 198, built in idle units 246, 226 and 179 | local underestimation for z₁₉₈ ∈ (0, 2) |
| V2 | Unit 47: input row and bias ×1.01, output weight ÷1.01 | same slope for large inputs, shallower negative GELU dip |
| V3 | Unit 255: bias −0.01 | slides its negative GELU dip, where the error changed sign |
| V4 | Ramp on z₃₈ − z₁₉₈, built in idle units 181 and 150 | rare reversal of the two positive paths |

In total **965 values in 5 tensors** of the output head change. The exact recipe is [`weight_space/edits/physionet/clinical_v0_relative_tail.json`](weight_space/edits/physionet/clinical_v0_relative_tail.json), and the algebra behind every edit is in [`docs/method.md`](docs/method.md).

### Step 3 · Test every idea with forward passes

Every edit was a hypothesis, and most hypotheses failed. The protocol was designed so that a lucky or overfitted edit could not slip through:

- **Three disjoint patient sets.** The original validation split (4,033 patients) was divided into **calibration** (even slots), used to look at activations and design edits, and **confirmation** (odd slots), used only to accept or reject a frozen edit. The **test** split (4,033 patients) was used only at the end.
- **Frozen before evaluation.** Each recipe was written, hashed and frozen before its first forward pass.
- **Paired comparisons.** Original and edited models always see the same patients, missing-value masks and noise, so differences come from the weights alone.
- **Both metrics or nothing.** At the user's request, an edit was kept only if it lowered **both** MAE and MSE on the confirmation patients. Before the test, V4 had to beat V1, V2 and V3 on both metrics, on each seed of the full 2,016-patient confirmation. It did, although only the margin over V1 is clearly outside the noise.

![17 hypotheses: 5 kept, 10 rejected, 2 pending](results/figures/04_search.png)

Two cases show why this matters:

- **"Mid-range bump on 198" and "sharpen unit 47" each failed on their own.** The first improved MSE but hurt MAE, and the second did the opposite. **Combined, they improved both** and became V2.
- **The tail correction on unit 167 looked like the best idea of all on calibration** (+3.03% MAE), then lost on confirmation patients. It was rejected. Without a separate confirmation set it would have been kept.

Every hypothesis, with its numbers and the reason it was kept or rejected, is listed in [`docs/experiment-log.md`](docs/experiment-log.md).

Seen in order, the search is a staircase: the accepted model only moves when an edit passes, and most iterations leave it where it was.

![The error went down one kept edit at a time](results/figures/05_progress.png)

## Results

Full test set: 4,033 patients, 100 imputations per patient, 2 noise seeds (20261103, 20261104), mean of the two seeds. The baseline is our own run of the released checkpoint under exactly the same protocol.

| | MAE | MSE |
|---|---:|---:|
| Original | 2.480960 | 350.120368 |
| V1 (193 values) | 2.424104 (−2.29%) | 348.309750 (−0.52%) |
| **V4 (965 values)** | **2.417163 (−2.57%)** | **347.977979 (−0.61%)** |
| 95% CI of the per-patient change, V4 vs original | [−0.0668, −0.0608] | [−4.672, −0.596] |
| Patients improved (of 4,013 with scored values) | 3,229 | 2,332 |

The exported V4 checkpoint was audited: its state and graph factors match the frozen candidate exactly, every test metric was recomputed from the saved raw predictions, and reloading the exported file through the CLI reproduces the candidate's predictions exactly in a separate check. See [`results/evidence/v4_verification.json`](results/evidence/v4_verification.json).

## How to read this result

This is a real improvement, measured carefully, but it is easy to oversell. In particular:

![Most of the gain is recalibration, not better physiology](results/figures/06_where_the_gain_comes_from.png)

- **Most of the MAE gain is in the 6 context columns.** Age, gender, ICU unit, admission time and length of stay improve by 27.7% (MAE) and 11.2% (MSE). The 34 vitals and lab values improve by only **0.29% (MAE) and 0.47% (MSE)**. 90% of the total MAE reduction comes from context columns, while 77% of the MSE reduction comes from physiological ones.
- **The simplest edit did most of the work.** V1, a recalibration of the output plus two boosted neurons (193 values), already gives −2.29% MAE on test. The programmed functions of V2–V4 improve on V1 by a further 0.29% (MAE) and 0.10% (MSE).
- **Not every variable improves.** 17 of 40 improve on both metrics. Some sparse lab values get clearly worse: pH (+76% MAE, 706 scored values) and chloride (+40%, 188 values). See the [per-variable figure](results/figures/07_per_variable.png).
- **The test set was used twice.** V1 was evaluated on test before V4 was built, so V4's test number is not a pristine hold-out. No test data was used to design any edit, but a fresh external evaluation would be the proper confirmation.
- **One checkpoint, one benchmark.** Two noise seeds of the same checkpoint are not five independently trained models. The paper reports the average over five checkpoints, so this baseline is not the paper's number.
- **Explainability is functional, not clinical.** We know exactly which function each edit adds to the head. We do not know which clinical concept a neuron stands for, and probably none maps cleanly to one.
- **This is an imputation benchmark.** Nothing here shows clinical usefulness.

## What did not work

The agent also tried editing deeper parts of CoSTI, where the representation is built, not just read out. None of these passed the both-metrics rule:

- **Temporal memory (Mamba).** The backward branch of the bidirectional Mamba keeps information for a much shorter time than the forward branch (nominally about 0.1–0.2 vs 6–12 steps). Lengthening it analytically made things worse: +1.7% MAE and +1.8% MSE on calibration. The asymmetry is real, but it is not a bug.
- **Attention.** An edit meant to make attention favour observed positions also changed preferences *between variables*, because mask and content are entangled in the same directions. An exact linear extraction of a "pure mask" signal turned out to be impossible for this construction. MSE got worse and the edit was rejected.
- **Graph between variables.** Shifting weight from the learned adaptive graph to the stored graph helped MAE a little but hurt MSE, almost entirely through one lab variable (AST).
- **Two-step feedback.** Changing the first denoising step changes the input of the second, and can change which of the 100 samples ends up as the median. In the two cases studied for a unit-21 edit, the first step alone reproduced the whole harmful effect. A correction that looks local in the head can act globally.

One side finding concerns reproducibility. CoSTI's adaptive graph factors are not stored in the standard `state_dict`, so a reloaded checkpoint does not reproduce the trained graph exactly. The agent reconstructed the factors and saved them explicitly in the exported checkpoint. That fixes reproducibility. It does not count towards the accuracy gain.

## Who did what

- **Javier Solís-García** (first author of CoSTI) set the goal and the rules: follow the weight-space-programming blog post, no gradient-based training, both MAE and MSE must improve. He also chose which checkpoint to focus on, authorised the GPU and decided when to stop.
- **The agent** (Astra 6 Max, working in OpenAI Codex in a terminal) did everything else: it read the code and the weights, wrote the inference harness, designed the validation protocol, diagnosed the model, wrote every edit, ran every evaluation and documented the failures.

The session ran on 3 November 2026, from 10:21 to about 23:40 UTC. The agent worked alone for several hours while the user was away. It started on all benchmark configurations, and an output-layer edit on PhysioNet already looked promising at that point. At the user's request it then spent a few hours on a single METR-LA traffic checkpoint, where the best edit gave only −0.025% MAE and an MSE change whose interval included zero. The user then asked which checkpoint looked most promising, and the agent picked the PhysioNet one shown here. Three promising candidates (V9, V10 and V14) were still in the queue when the machine was shut down. Their status is in the [experiment log](docs/experiment-log.md#unfinished-work).

The session transcript, user and agent messages only and in the original Spanish, is in [`docs/transcript/`](docs/transcript/).

## Repository layout

```
README.md                 this page
costi/                    the original CoSTI repository (upstream code, configs, figures)
weight_space/             the agent's toolkit: inference harness, diagnostics, edit builders
  edits/physionet/        every PhysioNet recipe, including the final V4
  edits/pilots/           recipes from the earlier METR-LA / AQI-36 pilot
results/
  figures/                the six figures on this page (16:9, ready for social media)
  data/                   every number behind the figures, as CSV/JSON
  evidence/               freeze record, test reports and the verification of the export
docs/
  method.md               the full technical write-up: algebra, each edit, protocol
  experiment-log.md       every hypothesis, with numbers and the reason it was kept or rejected
  reproduce.md            how to run the evaluation and rebuild the figures
  social-thread.md        a ready-to-post thread
  transcript/             session transcript (Spanish) and its provenance
scripts/
  make_figures.py         rebuilds results/figures from results/data
  export_head_coefficients.py   reads the two checkpoints, writes the head coefficients CSV
```

## Reproducing

The figures rebuild from the data in this repository alone:

```bash
pip install -r requirements.txt
python scripts/make_figures.py
```

Re-running the evaluations requires the released CoSTI checkpoints and the PhysioNet data (`costi/download_data.sh`) and, for reasonable speed, a CUDA GPU. Patient-level predictions and the edited checkpoint are not distributed here. [`docs/reproduce.md`](docs/reproduce.md) has the exact commands.

## Credits and citation

- Method: R. Takehi, J. Teoh, A. Zhang, O. Khattab, [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL, 2026.
- Model: J. Solís-García, B. Vega-Márquez, J. A. Nepomuceno, I. A. Nepomuceno-Chamorro, *CoSTI: Consistency models for (a faster) spatio-temporal imputation*, Knowledge-Based Systems 327, 114117 (2025), [doi:10.1016/j.knosys.2025.114117](https://doi.org/10.1016/j.knosys.2025.114117).
- Data: PhysioNet Computing in Cardiology Challenge 2019.

If you use this case study, please cite it via [`CITATION.cff`](CITATION.cff) and cite CoSTI as above. Code is MIT-licensed ([`LICENSE`](LICENSE)).
