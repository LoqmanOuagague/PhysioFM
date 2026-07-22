# NormWear + FiLM ablation: improving FiLM's macro-F1

This is an experiment to have the LLM do its own research.

## What this experiment is

`Experiments/NormWear_FiLM/` (see `README.md` there for the full ablation design) linear-probes
frozen NormWear on WESAD baseline/stress/amusement classification, with an optional FiLM layer that
conditions each window on the subject's own resting-state ("baseline") embedding. The full study
runs 8 probes (3 class-holdout splits x {plain, FiLM}, plus leave-one-subject-out (LOSO) x
{plain, FiLM}), but this run uses `run_ablation.py --film_only` to skip the plain (non-FiLM) arm
entirely, leaving the 4 FiLM probes: 3 class-holdout splits (one novel class withheld from training
per split) plus 1 leave-one-subject-out (LOSO) run.

**The goal here is narrower than the original ablation**: don't relitigate whether FiLM helps —
just push the **macro-F1 of the FiLM arms, averaged across both split styles** (class-holdout x 3 +
LOSO) as high as you can. The plain (capacity-matched) arm is not run at all in this setup, so
there's no per-checkpoint sanity-check baseline to compare against beyond the recorded
`results.tsv` history.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jul22`). Follow this repo's
   existing branch-naming style (see `NormWear-fullshot-training-on-Cogload1`): something like
   `NormWear-FiLM-improve-f1-<tag>`. The branch must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b NormWear-FiLM-improve-f1-<tag>` from current `main`.
3. **Read the in-scope files**: Read these for full context:
   - `Experiments/NormWear_FiLM/README.md` — the full ablation design and terminology.
   - `Experiments/NormWear_FiLM/wesad_dataset.py` — data loading and both splitting strategies.
     Do not modify.
   - `Experiments/NormWear_FiLM/normwear_loader.py` — frozen NormWear loading/encoding. Do not modify.
   - `Experiments/NormWear_FiLM/experiment.py` — shared train/eval core (see "What you CANNOT do"
     below for which parts of this specific file are off-limits).
   - `Experiments/NormWear_FiLM/probe_model.py`, `film.py`, `baseline_selector.py` — the model you're
     iterating on.
   - `Experiments/NormWear_FiLM/run_ablation.py`, `train_linear_probe.py` — the CLIs you run.
4. **Confirm the embedding cache is warm**: `data/WESAD/normwear_embed_cache/` should already contain
   one `.pt` file per pooled window (5540 at last check). If it's already complete, every run skips
   the slow NormWear encoding pass entirely — do not delete this directory, it's expensive to rebuild
   on CPU (~2 s/window).
5. **Calibration run**: run the ablation once, as-is, with the fixed budget flags below, and time
   it. This establishes the baseline `film_f1_macro_mean` and confirms the budget flags keep a full
   pass in the 15-20 minute range (see "Fixed evaluation budget"). Adjust `--n_trials` /
   `--search_epochs` if it's wildly off before starting the loop proper.
6. **Initialize results.tsv**: Create `results.tsv` in `Experiments/NormWear_FiLM` with just the header
   row. The baseline is recorded after the calibration run.
7. **Confirm and go**: Confirm setup looks good with the user.

Once you get confirmation, kick off the experimentation.

## Fixed evaluation budget

Unlike a fixed wall-clock budget, every checkpoint here runs the same **fixed hyperparameter-search
budget** so `film_f1_macro_mean` stays comparable across iterations (CPU-only, no GPU on this
machine, so wall-clock varies with epoch cost — pinning the *protocol* instead of the *clock* is what
keeps runs comparable). Use these flags every time (adjust only during the one-time calibration in
Setup step 5, then leave fixed for the rest of the loop):

```bash
python Experiments/NormWear_FiLM/run_ablation.py \
  --film_only --r_minutes_max 5 --n_trials 10 --search_epochs 30 --epochs 300 --patience 20
```

`--seed 42` (the default) stays fixed too — don't tune-by-reseeding. If you want to test seed
sensitivity, do that as its own explicit, labeled experiment (description: "seed sensitivity check"),
not silently mixed into the main loop.

## What you CAN do

Everything about the **model and how it trains** is fair game:

- `probe_model.py` — classifier head architecture, capacity-matching logic, anything downstream of
  the frozen embeddings.
- `film.py` — the FiLM conditioning MLP: width, depth, init, activation, whether gamma/beta are
  predicted jointly or separately, etc.
- `baseline_selector.py` — how the learnable baseline-window selection works (temperature schedule,
  per-channel vs. global selection, alternative gating, etc.).
- `experiment.py` — `HP_SEARCH_SPACE` (add/remove/reweight hyperparameters), `train_probe`'s training
  loop (optimizer, LR schedule, loss function, class weighting, regularization), and
  `hyperparameter_search`'s search algorithm — **except** the parts called out below.
- `train_linear_probe.py`, `run_ablation.py` — CLI defaults, as long as the fixed evaluation budget
  flags above are what actually gets used for logged checkpoints.

## What you CANNOT do

- **Modify `wesad_dataset.py`.** It is read-only. It defines the class-holdout and LOSO splits —
  the ground truth of what's train/test/novel-class/held-out-subject. Changing it would let the
  model cheat by seeing data it's not supposed to.
- **Modify `normwear_loader.py`**, or otherwise unfreeze/fine-tune the NormWear backbone. This is a
  *linear probing* study by design — the whole point is measuring what a lightweight FiLM
  conditioning layer buys on top of frozen embeddings. Fine-tuning the backbone would change the
  experiment into a different one.
- **Modify `experiment.py`'s `evaluate` function** (the metric computation) or the train/test/tune
  boundary contracts inside `run_class_holdout_experiment` / `run_loso_experiment` — specifically:
  the test set/fold must never be visible to `hyperparameter_search`, and `carve_validation_split` /
  `loso_validation_split` must keep carving the validation set out of the *training* pool only. These
  are the anti-cheating guarantees that make macro-F1 numbers trustworthy across iterations.
- **Install new packages or add dependencies.** Only use what's already available.
- **Delete or hand-edit `data/WESAD/normwear_embed_cache/`** or the committed
  `results/ablation_results_green_performance.json` snapshot.

**The goal is simple: get the highest `film_f1_macro_mean`** (see below). Architecture, FiLM design,
baseline-selection strategy, hyperparameters, training loop, loss/regularization — all in play.

**Simplicity criterion**: all else being equal, simpler is better. A small F1 gain that adds ugly
complexity to `film.py` or `baseline_selector.py` is not obviously worth it — weigh it against the
improvement magnitude. Removing something (a hyperparameter, a layer, a search dimension) while
holding `film_f1_macro_mean` steady or better is a great outcome.

**The first run** (the calibration run in Setup) establishes the baseline — run everything as-is.

## Output format

`run_ablation.py` prints a summary table (accuracy/precision/recall/F1/ROC-AUC per experiment) and
saves full results to `Experiments/NormWear_FiLM/results/ablation_results.json`
(`{config, resource_usage, results: {key: run}}`) — this file is overwritten every checkpoint. Kept
checkpoints additionally get a permanent, uniquely-named copy (see "The experiment loop" step 8) so
their results survive later runs overwriting the default file. The target metric isn't printed
directly — compute it from the saved JSON:

```bash
python3 -c "
import json
d = json.load(open('Experiments/NormWear_FiLM/results/ablation_results.json'))['results']
film = [v['metrics']['f1_macro'] if v['mode'] == 'class_holdout' else v['mean']['f1_macro']
        for v in d.values() if v['use_film']]
ch_film = [f for f, v in zip(film, d.values()) if v['mode'] == 'class_holdout']
loso_film = [f for f, v in zip(film, d.values()) if v['mode'] == 'loso']
print(f'film_f1_macro_mean={sum(film)/len(film):.6f}')
print(f'ch_film_f1_mean={sum(ch_film)/len(ch_film):.6f}')
print(f'loso_film_f1={loso_film[0]:.6f}')
"
```

`film_f1_macro_mean` is the unweighted mean of all four FiLM arms' macro-F1 (3 class-holdout + 1
LOSO) — this is the number to optimize.

## Logging results

When an experiment (a full `run_ablation.py` checkpoint) is done, log it to `results.tsv`
(tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	film_f1_macro_mean	ch_film_f1_mean	loso_film_f1	status	description
```

1. git commit hash (short, 7 chars)
2. `film_f1_macro_mean` (e.g. 0.412000) — use 0.000000 for crashes
3. `ch_film_f1_mean`, the class-holdout-only component (average of the 3 novel-class FiLM runs)
4. `loso_film_f1`, the LOSO FiLM run's mean macro-F1 across subject folds
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried — **if the idea is drawn from or directly
   inspired by a specific research paper**, cite it in parens at the end: `(Author et al. Year)`,
   plus an arXiv id/DOI if you have one. Don't cite for generic ML folklore (e.g. "class-weighted
   loss", "deeper MLP") — only when a specific paper's method or finding is what you're actually
   trying.

Example:

```
commit	film_f1_macro_mean	ch_film_f1_mean	loso_film_f1	status	description
a1b2c3d	0.412000	0.398000	0.454000	keep	baseline
b2c3d4e	0.431000	0.415000	0.472000	keep	class-weighted loss for imbalance
c3d4e5f	0.405000	0.390000	0.440000	discard	deeper FiLM MLP (2 hidden layers)
d4e5f6g	0.000000	0.000000	0.000000	crash	per-channel selector temperature (shape mismatch)
e5f6g7h	0.448000	0.430000	0.484000	keep	FiLM gamma/beta predicted from a shared trunk instead of separate heads (Perez et al. 2018, arXiv:1709.07871)
```

## The experiment loop

The experiment runs on the dedicated branch created in Setup.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune the model/training code with an experimental idea by directly hacking the code (see "What
   you CAN do"). If the idea is drawn from or directly inspired by a specific research paper (as
   opposed to general ML folklore), note which paper — you'll cite it when logging the result
   (see "Logging results", column 6).
3. `git commit`.
4. Run the checkpoint: `python Experiments/NormWear_FiLM/run_ablation.py --film_only --r_minutes_max 5 --n_trials 10 --search_epochs 30 --epochs 300 --patience 20 > run.log 2>&1`
   (redirect everything — do NOT use `tee` or let output flood your context).
5. Extract `film_f1_macro_mean` (and its two components) with the one-liner above, reading from
   `results/ablation_results.json`.
6. If extraction fails (missing keys, empty JSON), the run crashed. Run `tail -n 50 run.log` to read
   the Python stack trace and attempt a fix. If you can't get things to work after more than a few
   attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit `results.tsv`, leave it untracked by git).
8. If `film_f1_macro_mean` improved (higher), "advance" the branch, keeping the git commit. Also
   copy `results/ablation_results.json` to `results/ablation_results_<commit>_<slug>.json`, where
   `<commit>` is the short (7-char) commit hash from step 3/the tsv and `<slug>` is the tsv
   description with spaces replaced by hyphens (e.g. `results/ablation_results_b2c3d4e_class-weighted-loss-for-imbalance.json`)
   — this preserves the checkpoint's results before the next iteration overwrites the default file.
9. If `film_f1_macro_mean` is equal or worse, `git reset` back to where you started. Do not keep a
   named snapshot for a discarded/crashed checkpoint.

You are advancing the branch so you can keep iterating; if a change doesn't pan out, discard and try
the next idea. If you feel like you're getting stuck, you can rewind further back, but do this very
sparingly (if ever).

**Timeout**: each checkpoint (the fixed-budget `run_ablation.py` call above) should land somewhere
around 15-20 minutes on this machine, per the calibration run. If a run exceeds 45 minutes, kill it
and treat it as a failure (discard and revert) — something's off (e.g. a change made a training loop
much slower per step, or broke early stopping so it runs to the `--epochs` ceiling every time).

**Crashes**: use your judgment. If it's something dumb and easy to fix (a typo, a shape mismatch),
fix it and re-run. If the idea itself is fundamentally broken, log `crash` in the tsv and move on.

**NEVER STOP**: once the experiment loop has begun (after Setup), do NOT pause to ask the human if
you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The
human might be asleep, or gone from the computer, and expects you to continue working
*indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder:
re-read `README.md` for angles you haven't tried (per-subject vs. global baseline selection,
alternative FiLM conditioning shapes, class-imbalance handling, ensembling across the FiLM arms'
learned `effective_r_minutes`, curriculum on `r_minutes_max`, etc.), or try combining previous
near-misses. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each checkpoint takes
~15-20 minutes, that's roughly 3/hour, for about 25 over the duration of an average night. The user
then wakes up to experimental results, all completed while they slept.
