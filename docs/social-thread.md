# Thread draft (X / Bluesky / LinkedIn)

Not yet posted. Each post is under 280 characters. Attach the figure listed under it.

---

**1/8**
Can an AI agent improve a trained neural network without training it?

We tried weight-space programming on CoSTI, our imputation model: the agent read the weights and edited 965 of 2.4M by hand. No gradients.

Test MAE −2.57%, MSE −0.61%. 🧵

📎 `results/figures/01_headline.png`

---

**2/8**
The idea is from Takehi, Teoh, Zhang & Khattab (MIT CSAIL): an LLM reads a network's weights, works out what they compute and writes the fix straight into them. Only forward passes to check.

We tried it on a real research model.
https://rikiyat.github.io/blog/weight-space-programming/

---

**3/8**
Step 1: read.

CoSTI's output head collapses exactly into y = b + Σ aⱼ·GELU(zⱼ), one effective weight per neuron.

Sorting the 256 values exposes something you'd never see from the loss curve: 176 of those neurons do almost nothing.

📎 `results/figures/02_read_the_weights.png`

---

**4/8**
Step 2: write.

A steep GELU is a ReLU switch. The agent copied a neuron's input weights ×16 into idle neurons and set new biases, and the switches added up to a bump exactly where unit 198 under-predicted.

Small functions, compiled into weights by hand.

📎 `results/figures/03_write_the_weights.png`

---

**5/8**
Step 3: test everything.

17 hypotheses: 5 kept, 10 rejected, 2 pending. Each was frozen before its first evaluation and judged on held-out patients, and it had to lower both MAE and MSE.

The best idea on calibration failed on confirmation and was dropped.

📎 `results/figures/04_search.png`
📎 `results/figures/05_progress.png`

---

**6/8**
The honest part.

Most of the MAE gain is in 6 context columns (age, ICU unit, length of stay…). On the 34 vitals and labs: −0.29% MAE, −0.47% MSE.

And the simplest edit, recalibrating the output plus 2 neurons, already gives −2.29%.

📎 `results/figures/06_where_the_gain_comes_from.png`

---

**7/8**
What failed: edits to the deeper blocks (Mamba memory, attention, the variable graph) never improved both metrics. A local fix in denoising step 1 can spread to step 2.

Test set scored twice (V1, then V4), never used to design edits. One checkpoint: needs external validation.

---

**8/8**
Everything is open: the recipe (965 values as JSON), the agent's toolkit, every rejected hypothesis with its numbers, figures and session transcript.

Directed by a human, executed by an agent (Astra 6 Max in Codex, about 13 hours).

https://github.com/javiersgjavi/costi-weight-space-programming

---

### Short version (single post / LinkedIn)

> We asked an AI agent to improve CoSTI, our time-series imputation model, **without training it**. Following MIT's "weight-space programming" idea, it read the weights, found idle neurons, and programmed small corrections into them: 965 of 2.4M values. On 4,033 ICU patients, MAE fell 2.57% and MSE 0.61%. Most of the gain is recalibration of easy context columns, not better physiology, and we show every failed idea too. Code, recipe and figures: github.com/javiersgjavi/costi-weight-space-programming
