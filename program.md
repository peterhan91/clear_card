# CLEAR Autonomous Linear Probe Optimization & Concept Analysis

This is an autonomous research program. Once the setup is confirmed, Claude runs
indefinitely — optimizing linear probes to outperform all foundation-model baselines
on MIMIC and PadChest, then producing clinically interpretable concept analysis.

---

## Setup

Work with the user to:

1. **Agree on a run tag** (e.g. `mar15`). Branch: `autoresearch/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read in-scope files** for full context:
   - `CLAUDE.md` — environment, paths, best model info.
   - `concepts/exp_linear_mimic.py` — full pipeline (feature extraction, projection, training). Read-only.
   - `concepts/exp_linear_mimic_foundation.py` — foundation model baselines. Read-only.
   - `concepts/probe.py` — **the file you create and modify**. See below.
4. **Verify cached features exist** (hard prerequisite — do NOT regenerate these).
   Check that `concepts/cache/` contains:
   - `concept_features_*.pt` — CLIP concept features (~492k × 768)
   - `mimic_train_*.pt`, `mimic_validate_*.pt`, `mimic_test_*.pt` — CLIP image features
   - If missing, tell the user to run `exp_linear_mimic.py --cache_only` on the
     cluster first. **Do not attempt to re-extract features yourself.**
5. **Create `concepts/probe.py`**: Extract the probe training logic into a standalone
   script. This is the only file you modify during the experiment loop.
   See [probe.py specification](#probepy-specification) below.
6. **Cache LLM-projected features**: The first thing `probe.py` must do is cache
   `{dataset}_{split}_{model_key}_llm.npy` files in `concepts/cache/`. This makes
   each subsequent experiment take ~1-3 minutes instead of ~10+.
7. **Establish baselines**: Run foundation model baselines if results don't exist.
   Check `concepts/results/` for existing `linear_mimic_foundation_*` directories.
   If missing, run each (note: ark_plus and chexzero require checkpoint paths):
   ```bash
   python concepts/exp_linear_mimic_foundation.py --model rad_dino --dataset mimic
   python concepts/exp_linear_mimic_foundation.py --model biomedclip --dataset mimic
   python concepts/exp_linear_mimic_foundation.py --model ark_plus --dataset mimic \
       --ark_checkpoint /path/to/Ark6_swinLarge768_ep50.pth.tar
   python concepts/exp_linear_mimic_foundation.py --model chexzero --dataset mimic \
       --chexzero_checkpoint /path/to/best_64_5e-05_original_22000_0.864.pt
   ```
   If checkpoint paths are unavailable, skip that model and note it in results.tsv.
   Record their macro AUROC as targets to beat. Also run the CLEAR baseline (step 8).
8. **Initialize results.tsv** in `concepts/results/results.tsv` with just the header.
9. **Confirm and go.**

---

## probe.py Specification

Create `concepts/probe.py` — a self-contained script that:

1. **Reads cached features**: Loads precomputed CLIP features (`concept_features_{tag}.pt`,
   `{dataset}_{split}_{tag}.pt`) and concept LLM embeddings from `concepts/cache/` and
   `concepts/embeddings_output/`. Computes the LLM projection (`scores @ embeddings`)
   and caches the result as `{dataset}_{split}_{model_key}_llm.npy`. On subsequent
   runs, loads the cached LLM-projected features directly. **Never re-run CLIP encoding.**
2. **Trains a classifier**: This is the part you iterate on. Initially a simple
   `nn.Linear` logistic regression, but can evolve into MLPs, etc.
3. **Evaluates**: Per-phenotype AUROC, macro AUROC. Reports both val and test.
4. **Outputs standardized metrics** (see [Output format](#output-format)).

The script should accept these arguments:
```
--dataset       mimic|mimic63|padchest|chexchonet  (default: mimic)
--model_key     kalm_gemma3_12b|sfr_mistral|nemotron_8b|openai_3large
--n_seeds       Number of seeds (default: 3 for speed; final validation uses 5)
--quick         Use 2 seeds + 50 max epochs for rapid iteration
```

**Critical invariants** (do NOT change these in probe.py):
- **Always use precomputed CLIP features** from `concepts/cache/`. Never re-run
  CLIP image encoding or concept encoding — these are expensive and already cached
  as `concept_features_{tag}.pt` and `{dataset}_{split}_{tag}.pt`. Load them directly.
- The LLM projection: `scores = img_feats @ concept_feats.T; llm = scores @ embeddings`
- L2 normalization of LLM representations
- The evaluation function: `sklearn.metrics.roc_auc_score` per label, macro average
- MIN_TEST_POSITIVES = 20
- Train/val/test splits (no patient overlap)
- Image preprocessing (mean=101.48761, std=83.43944, resize 448) — already baked
  into the cached features, do not recompute

**Concept importance compatibility**: The `concept_importance_mimic.py` script
computes importance via `cos_sim(W_j, E_i)` where W_j is the linear layer weight.
To keep this working when using non-linear architectures (MLP, etc.):
- Always include a **final `nn.Linear` layer** mapping to phenotype outputs.
- Save model checkpoints with the key `'state_dict'` containing at minimum
  the final linear layer's weight (e.g., `model.head.weight` or `model.linear.weight`).
- If the architecture is deep, concept importance can alternatively be computed
  via gradient-based attribution (modify `concept_importance_mimic.py` in Phase 3).

**What you CAN change** in probe.py (everything in the training section):
- Classifier architecture (linear, MLP, deeper networks, residual, etc.)
- Optimizer (Adam, AdamW, SGD, LBFGS, etc.)
- Learning rate, weight decay, momentum, and all other hyperparameters
- Learning rate scheduling (warmup, cosine, step decay, etc.)
- Batch size
- Max epochs, patience, early stopping strategy
- Loss function (BCE, focal loss, weighted BCE, asymmetric loss, etc.)
- Regularization (dropout, L1, L2, mixup, label smoothing, etc.)
- Class imbalance handling (sample weights, oversampling, etc.)
- Feature preprocessing (PCA, standardization, whitening, feature selection)
- Concept selection/filtering (top-k by variance, frequency, etc.)
- Ensemble strategies across embedding models
- Multi-task vs per-task training strategies
- **New LLM embeddings**: You can compute concept embeddings from any HuggingFace
  model (see `concepts/get_embed.py` for the pipeline). If you believe a different
  embedding model would help, generate new embeddings and test them. Promising models
  to try: `BAAI/bge-en-icl`, `Alibaba-NLP/gte-Qwen2-7B-instruct`,
  `nvidia/NV-Embed-v2`, `intfloat/e5-mistral-7b-instruct`, or any strong model on
  the MTEB leaderboard. Save new embeddings to `concepts/embeddings_output/` following
  the existing naming convention (`cxr_embeddings_{model_key}.pickle`).
- Any other training trick

---

## Output Format

When `probe.py` finishes, it must print a summary block:

```
---
macro_auroc:       0.7234
val_auroc:         0.7680
n_phenotypes:      1270
n_seeds:           3
training_seconds:  85.2
dataset:           mimic
embedding_model:   kalm_gemma3_12b
---
```

Extract the key metric:
```bash
grep "^macro_auroc:" run.log
```

---

## Logging Results

Log every experiment to `concepts/results/results.tsv` (tab-separated, untracked by git).

Header and columns:

```
commit	macro_auroc	val_auroc	n_pheno	dataset	status	description
```

1. `commit` — short git hash (7 chars)
2. `macro_auroc` — test macro AUROC (0.000000 for crashes)
3. `val_auroc` — best validation AUROC
4. `n_pheno` — number of phenotypes evaluated
5. `dataset` — which dataset (mimic, padchest, etc.)
6. `status` — `keep`, `discard`, or `crash`
7. `description` — short text of what this experiment tried

Example:
```
commit	macro_auroc	val_auroc	n_pheno	dataset	status	description
a1b2c3d	0.7012	0.7530	1270	mimic	keep	baseline logistic regression
b2c3d4e	0.7089	0.7601	1270	mimic	keep	AdamW + cosine LR schedule
c3d4e5f	0.7045	0.7580	1270	mimic	discard	MLP 256 hidden (no improvement)
d4e5f6g	0.7123	0.7650	1270	mimic	keep	focal loss + class weights
e5f6g7h	0.0000	0.0000	0	mimic	crash	3-layer MLP (NaN loss)
```

---

## Phase 1: The Experiment Loop (LOOP FOREVER)

The experiment runs on the `autoresearch/<tag>` branch.

**Use `--quick` mode** (2 seeds, 50 max epochs) for rapid iteration. Each experiment
should take ~2-5 minutes on cached LLM features. At ~2-5 min/experiment you can
run approximately 12-30 experiments per hour, or ~100-200 overnight.

LOOP FOREVER:

1. Look at git state and `results.tsv` — what has been tried, what worked.
2. Choose an experiment. Modify `concepts/probe.py` with your idea.
3. `git add concepts/probe.py && git commit -m "<description>"`
4. Run: `python concepts/probe.py --quick > run.log 2>&1`
5. Read results: `grep "^macro_auroc:\|^val_auroc:" run.log`
6. If grep is empty → crash. Run `tail -n 50 run.log` to debug. Fix if trivial,
   skip if fundamentally broken.
7. Log to `concepts/results/results.tsv` (do NOT commit results.tsv).
8. If `macro_auroc` improved: **keep** the commit, advance the branch.
9. If `macro_auroc` is equal or worse: **discard** — `git reset --hard HEAD~1`.

### Experiment strategy (suggested order)

Start simple, escalate complexity:

**Round 1 — Hyperparameter tuning** (~10 experiments):
- LR sweep: 5e-4, 1e-3, 2e-3, 5e-3, 1e-2
- Weight decay sweep: 0, 1e-6, 1e-4, 1e-2
- Optimizer: Adam → AdamW → SGD+momentum
- Batch size: 256, 512, 1024, 2048

**Round 2 — Architecture** (~10 experiments):
- 1-hidden-layer MLP: 128, 256, 512, 1024 units
- Add BatchNorm and/or LayerNorm
- Add dropout (0.1, 0.2, 0.3)
- 2-hidden-layer MLP with residual connections
- Bottleneck architecture (project down then up)

**Round 3 — Loss & class imbalance** (~10 experiments):
- Focal loss (gamma=1, 2, 3)
- Weighted BCE (inverse frequency)
- Asymmetric loss (ASL)
- Label smoothing (0.01, 0.05)
- Positive-negative ratio weighting

**Round 4 — Feature engineering** (~10 experiments):
- Standardize features (zero-mean, unit-variance per dimension)
- PCA to reduce dimensionality (512, 768, 1024, 2048)
- Top-k concept selection by variance/mutual information
- Temperature scaling on concept similarities before projection
- Non-linear similarity (softmax with temperature)

**Round 5 — Training tricks** (~10 experiments):
- LR warmup (5-10% of training) + cosine annealing
- Gradient clipping
- EMA (exponential moving average) of weights
- Mixup on LLM features
- Progressive label inclusion (start with high-prevalence labels)

**Round 6 — Multi-model ensemble** (~5 experiments):
- Concatenate LLM representations from multiple embedding models
- Late fusion: average predictions from per-model probes
- Learned ensemble weights on validation set
- PCA on concatenated features

**Round 7 — New LLM embeddings** (~5-10 experiments):
- Use `concepts/get_embed.py` to generate concept embeddings from new HuggingFace
  models. Each embedding run takes ~30-60 min for 492k concepts but only needs to
  happen once per model. Good candidates from MTEB leaderboard:
  - `BAAI/bge-en-icl` (English, strong general-purpose)
  - `Alibaba-NLP/gte-Qwen2-7B-instruct` (instruction-tuned, high MTEB)
  - `nvidia/NV-Embed-v2` (strong retrieval model)
  - `intfloat/e5-mistral-7b-instruct` (instruction-tuned Mistral)
- After generating embeddings, add the new model_key to probe.py's config and test.
- Compare new embeddings against existing 4 models (kalm_gemma3, sfr_mistral,
  nemotron, openai_3large). Keep the best.
- Try concatenating embeddings from 2-3 top models.

**Round 8 — Advanced** (~10+ experiments):
- Per-phenotype-group probes (group by ICD category)
- Attention-based concept weighting (learned concept attention)
- Temperature-scaled cosine similarity for concept scoring
- Contrastive pretraining of probe head
- Knowledge distillation from best ensemble to single model

After each round, assess: are we beating the best foundation model baseline?
If yes, move to Phase 2 validation. If no, keep iterating.

### When to transition

Move to Phase 2 when **ALL** of these are true:
- CLEAR macro AUROC on MIMIC > best foundation model baseline by ≥0.5%
- Last 5 experiments produced no improvement (plateau)
- At least 30 experiments have been run

**Periodic PadChest sanity check**: Every ~15 experiments, run the current best
probe on PadChest with `--quick` to check generalization. If MIMIC improves but
PadChest degrades, you may be overfitting to MIMIC-specific patterns — favor
regularization and simpler architectures.

If stuck after 50+ experiments, try radical changes:
- Completely different architecture paradigm
- Re-examine feature projection step (temperature, normalization)
- Combine concepts with direct CLIP features

---

## Phase 2: Cross-Dataset Validation

Once Phase 1 converges:

1. **Full validation on MIMIC**: Run best probe with `--n_seeds 5` (no `--quick`).
   Record final AUROC with mean ± std.
2. **Cache PadChest LLM features**: Run probe.py on PadChest (will auto-cache).
3. **PadChest evaluation**: Run best probe on PadChest with `--n_seeds 5`.
4. **All embedding models**: Run best probe with each of:
   `kalm_gemma3_12b`, `sfr_mistral`, `nemotron_8b`, `openai_3large`
   on both MIMIC and PadChest. Find the best embedding model.
5. **Comparison table**: Generate a comprehensive comparison:

   ```
   Method                  | MIMIC AUROC  | PadChest AUROC
   ========================|==============|===============
   CLEAR (best probe)      | 0.XXXX±0.XX | 0.XXXX±0.XX
   Ark+                    | 0.XXXX±0.XX | 0.XXXX±0.XX
   RAD-DINO                | 0.XXXX±0.XX | 0.XXXX±0.XX
   CheXzero                | 0.XXXX±0.XX | 0.XXXX±0.XX
   BiomedCLIP              | 0.XXXX±0.XX | 0.XXXX±0.XX
   ```

6. If CLEAR does NOT beat all baselines on both datasets, **go back to Phase 1**.
   Analyze which phenotype categories underperform and target them specifically.

Save all results to `concepts/results/phase2_validation/`.

---

## Phase 3: Concept Analysis

Once CLEAR definitively beats all baselines:

### 3a. Concept Importance

Run `concepts/concept_importance_mimic.py` with the best probe model to compute
per-phenotype concept importance (cosine similarity between probe weights and
concept LLM embeddings).

Generate top-10 positive and top-10 negative concepts for each phenotype.

### 3b. Clinical Validation Checklist

For the **top 20 phenotypes by AUROC**, manually verify (print in output):
- Are the top positive concepts clinically plausible?
  (e.g., heart_failure → "cardiomegaly", "pleural effusion", "pacemaker")
- Are the top negative concepts sensible exclusions?
  (e.g., heart_failure → "no cardiomegaly", "normal heart size")
- Flag any suspicious concept-phenotype associations for human review.

### 3c. Cross-Dataset Concept Consistency

Compare concept importance rankings between MIMIC and PadChest:
- Spearman rank correlation of top-100 concepts per phenotype
- Identify concepts that are important on both datasets (robust signals)
- Identify concepts that disagree (dataset-specific artifacts)

### 3d. Visualization

Generate the following figures using the existing plotting scripts:

1. **Macro AUROC comparison** (`plot_auroc_comparison.py`):
   Bar chart of CLEAR vs all baselines on MIMIC and PadChest.

2. **Per-phenotype ROC curves** (`plot_auroc_comparison.py`):
   ROC curves for the top 20 phenotypes by AUROC.

3. **Concept importance** (`plot_concept_importance.py`):
   Horizontal bar charts for the top 20 phenotypes showing top-10 concepts.

4. **New: Concept heatmap**:
   Create a phenotype × concept heatmap showing alignment scores for the
   top 50 phenotypes × top 50 most-frequently-important concepts.

5. **New: Performance by phenotype category**:
   Box plot of AUROC grouped by ICD category (infection, neoplasm, cardiac, etc.)
   showing CLEAR vs baselines per category.

Save all figures to `concepts/results/figures/`.

### 3e. Summary Report

Write `concepts/results/analysis_report.md` with:
- Best probe architecture description
- Comparison tables (CLEAR vs baselines)
- Key concept-phenotype findings
- Clinical interpretation highlights
- Limitations and dataset-specific observations

---

## Rules

**What you CAN do:**
- Modify `concepts/probe.py` — this is your canvas. Architecture, optimizer,
  hyperparameters, loss, regularization, feature engineering — all fair game.
- Run `concepts/get_embed.py` to generate concept embeddings from new HuggingFace
  models (or any model accessible via transformers/sentence-transformers). New
  embeddings are saved to `concepts/embeddings_output/` and added to probe.py config.
- Create new visualization scripts in `concepts/`.
- Read any file in the repository for context.

**What you CANNOT do:**
- Modify `concepts/exp_linear_mimic.py` or `concepts/exp_linear_mimic_foundation.py`.
  These are the fixed feature extraction and baseline pipelines.
- Change the evaluation metric (macro AUROC via sklearn roc_auc_score).
- Change the train/val/test splits or patient-level split integrity.
- Change the CLIP model, concept list, or concept extraction pipeline.
- **Re-run CLIP image/concept encoding.** Always use the precomputed `.pt` feature
  caches in `concepts/cache/`. These are expensive to generate and already done.
- Install new packages beyond what is available in the `ml311` environment
  (note: HuggingFace `transformers` and `sentence-transformers` are available).
- Modify the image preprocessing or feature extraction pipeline.

**Simplicity criterion**: All else being equal, simpler is better. A 0.1% AUROC gain
from a 100-line MLP is not worth it if a 2-line hyperparameter tweak gets 0.08%.
Removing complexity while maintaining performance is a win. When evaluating whether
to keep a change, weigh complexity against improvement magnitude.

**The first run**: Always establish the baseline first — run the current logistic
regression exactly as-is to get a reproducible starting AUROC. For reference, the
current CLEAR baseline with KaLM-Gemma3-12B on MIMIC PheWAS is **0.7012** macro
AUROC (5 seeds, 1,270 phenotypes evaluated). Foundation model baselines have not
yet been run — establishing them is a setup priority.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the user.
Do NOT ask "should I continue?". The user might be asleep. You are autonomous.
If you run out of ideas, re-read the in-scope files, think about what hasn't been
tried, try combining previous near-misses, or try radical departures. The loop runs
until the human interrupts you.

**Timeout**: If an experiment exceeds 10 minutes, kill it and treat as a crash.

**Crashes**: If a run crashes due to a typo or simple bug, fix and re-run. If the
idea itself is broken, log "crash", revert, and move on.

**GPU memory**: Experiments run on a single GPU. Modest memory increases are fine
for meaningful AUROC gains, but don't OOM. If an architecture is too large, scale
it down or use gradient accumulation.

**Environment**: Always activate the environment before running:
```bash
eval "$(micromamba shell hook --shell bash)" && micromamba activate ml311
```
