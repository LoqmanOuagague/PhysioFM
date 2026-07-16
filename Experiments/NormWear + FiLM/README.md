# NormWear + FiLM: does a personal physiological baseline help or hurt?

## What this experiment is

An ablation study of linear probing the frozen [NormWear](https://huggingface.co/mosaic-laboratory/normwear)
foundation model on [WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/) (chest+wrist physiological
signals), classifying the study protocol condition — restricted to **baseline / stress / amusement**
(`wesad_dataset.TASK_CLASSES`; `meditation` is excluded from this study) — with an optional
[FiLM](https://arxiv.org/abs/1709.07871) (Feature-wise Linear Modulation) layer that conditions each window's
embedding on that same subject's own resting-state signal before the classifier head sees it.

## Hypothesis

A physiological signal is ambiguous on its own: the same raw ECG/EDA pattern can mean different things depending
on the person's individual resting physiology (e.g. someone with a naturally elevated resting heart rate
shouldn't be read as "stressed" just because their heart rate is higher than someone else's baseline). A plain
linear probe only sees the task window's embedding and has no way to tell whether a given signal is elevated
*for this person* or just elevated in absolute terms.

**Hypothesis under test:** conditioning the probe on an embedding of the subject's own `baseline`-condition
signal (their resting state, recorded once per subject as part of the WESAD protocol) via FiLM lets the model
calibrate against that individual's personal reference instead of only population-level patterns, and should
improve classification performance over a plain (unconditioned) linear probe on the same frozen embeddings.

This cuts both ways, which is the actual point of running it as an ablation rather than assuming the answer: a
personal baseline could just as easily *hurt* — e.g. if it lets the model shortcut to "does this look like this
subject's own baseline window" instead of learning the general stress/amusement signal, it may overfit to
per-subject idiosyncrasies and generalize worse to subjects (or classes) the classifier hasn't calibrated
against. The experiments below (Section "The experiments") are designed to surface that trade-off from both a
class-generalization and a subject-generalization angle.

## How conditioning works

1. Every pooled baseline/stress/amusement window (10 channels: chest ACC x/y/z, ECG, EMG, EDA, Temp, Resp + wrist
   EDA, Temp; 6 s at 65 Hz) is encoded once with frozen NormWear, giving one 768-dim CLS embedding per channel.
2. For each subject, up to `--r_minutes_max` (a user-specified **upper bound**, not a fixed amount) of their
   earliest `baseline`-condition windows are encoded the same way and kept as an ordered, per-channel candidate
   sequence.
3. **The number of those candidate windows actually used is learned, not fixed.** `baseline_selector.
   LearnableBaselineSelector` holds one scalar parameter per model; a sigmoid maps it to an *effective window
   count* between 0 and the `--r_minutes_max` cap, and a soft cutoff (another sigmoid, this time along the window
   index) turns that into per-window weights — windows well inside the cutoff get weight ≈1, windows well past it
   get weight ≈0. Because the cutoff is soft, gradients flow through it, so this scalar is trained jointly with
   the classifier head by ordinary backprop on the training loss: the model finds whatever amount of baseline
   signal (up to the user's bound) minimizes the loss, rather than a human guessing `r_minutes` up front. The
   converged value is reported as `effective_r_minutes` in every FiLM experiment's results.
4. The resulting weighted-average embedding is the subject's baseline reference. When FiLM is enabled, `FiLMLayer`
   predicts a per-channel `(gamma, beta)` from it and applies `(1 + gamma) * embed + beta` to every window of
   theirs (including, unavoidably, the baseline windows themselves) before the classifier head. One MLP is shared
   across all 10 channels, conditioned on each channel's own baseline embedding, so parameter count doesn't scale
   with channel count.


## The experiments

All experiments use the same pooled baseline/stress/amusement windows and the same probe architecture; they vary
along two independent axes — run together and saved in one pass by `run_ablation.py`:

| Split | FiLM | Runs |
|---|---|---|
| Class-holdout (novel_class ∈ {baseline, stress, amusement}) | No | 3 |
| Class-holdout (novel_class ∈ {baseline, stress, amusement}) | Yes | 3 |
| Leave-one-subject-out (LOSO) cross-validation | No | 1 |
| Leave-one-subject-out (LOSO) cross-validation | Yes | 1 |

- **Class-holdout** (`wesad_dataset.class_holdout_split`): for a chosen `novel_class`, train sees 80% of the
  *other two* classes only (0% of `novel_class`); test gets the remaining 20% of those two classes plus 100% of
  `novel_class`. Subjects are freely mixed across train and test (unlike LOSO — the axis being tested here is
  generalization to an unseen *class*, not an unseen *subject*). Since the model was never given a training
  example of `novel_class`, this measures whether FiLM's personal-baseline conditioning helps the model flag a
  genuinely novel physiological pattern as such, or whether it only helps discriminate between conditions it's
  already seen. Run once per choice of `novel_class` (3 variants) since which class is withheld materially
  changes what's being tested — a novel `stress` class is a very different question from a novel `baseline`.
- **LOSO** (`wesad_dataset.loso_folds`): one fold per subject, trained on every other subject and evaluated on
  the held-out one, metrics averaged (mean ± std) across folds. This is the harder, more honest test of whether
  FiLM conditioning generalizes to a subject the model has never seen calibrated against — exactly the setting
  where "personal baseline helps" or "personal baseline overfits" should actually show up.

Comparing each class-holdout FiLM run against its plain counterpart, and LOSO-FiLM against LOSO-plain, is the
actual ablation: if FiLM helps on some novel classes but not others, or helps class-holdout but not LOSO, that's
evidence about *what kind* of generalization the personal baseline is (or isn't) buying.

## Files

| File | Purpose |
|---|---|
| `normwear_loader.py` | Loads frozen NormWear. Works around a bug where `AutoModel.from_pretrained(..., trust_remote_code=True)` crashes on this repo's transformers version (see docstring) by constructing the model directly and loading `model.safetensors` by hand. Exposes `encode_windows(model, signals)` → per-channel CLS embeddings. |
| `wesad_dataset.py` | Loads `data/WESAD/{train,test}_manifest.csv`, filters to the 3-class task pool (`task_rows`), and provides the two splitting strategies (`class_holdout_split`, `loso_folds`) plus `build_baseline_sequences` (the padded, per-subject candidate baseline window sequences the learnable selector consumes). |
| `baseline_selector.py` | `LearnableBaselineSelector`: the single learnable parameter (bounded above by `--r_minutes_max`) that decides how much baseline signal to average into the FiLM reference. |
| `film.py` | `FiLMLayer`: a small conditioning MLP, zero-initialized so it starts as the identity function. |
| `probe_model.py` | `NormWearFiLMProbe`: per-channel embeddings → optional learnable baseline selection + FiLM → flatten → MLP classifier head. The only trainable parameters (NormWear stays frozen throughout). |
| `experiment.py` | Shared core: builds/caches embeddings, trains+evaluates one probe on a given train/test row pool, and runs a full class-holdout or LOSO experiment. Used by both scripts below so they share one code path. |
| `train_linear_probe.py` | CLI for a *single* (split × FiLM) run — useful for iterating on one configuration. |
| `run_ablation.py` | Runs all 8 experiments in one pass (sharing one embedding cache across them) and saves metrics to `results/ablation_results.json`. |

## Running it

From the repo root:

```bash
# the full ablation study (3 novel-class choices x plain/FiLM + LOSO x plain/FiLM), saved to results/ablation_results.json
python "Experiments/NormWear + FiLM/run_ablation.py" --r_minutes_max 5

# iterate on a single configuration instead
python "Experiments/NormWear + FiLM/train_linear_probe.py" --eval_mode class_holdout --novel_class stress --use_film --r_minutes_max 5
python "Experiments/NormWear + FiLM/train_linear_probe.py" --eval_mode loso --no-use_film
```

Every experiment reports **Accuracy, macro-Precision, macro-Recall, macro-F1, and macro-averaged one-vs-one ROC
AUC** on held-out data (LOSO: mean ± std across subject folds); the FiLM experiments additionally report
`effective_r_minutes`, the baseline duration the learnable selector converged to.

Per-window NormWear embeddings are cached to `data/WESAD/normwear_embed_cache.pt` after the first run (NormWear
is frozen, so there's no reason to re-encode the same windows across experiments); delete that file to force a
re-encode. See `run_ablation.py --help` / `train_linear_probe.py --help` for the full list of hyperparameters
(hidden dim, dropout, epochs, learning rate, batch size, device, selector temperature, ...).

## Caveats

- NormWear is slow on CPU (~2 s/window in testing here); encoding the full pooled dataset the first time can
  take on the order of an hour or more on CPU. Pass `--device cuda` if a GPU is available — this only affects
  the one-off encoding pass, since the classifier head (and the FiLM/selector parameters) are tiny.
- LOSO trains one probe per subject (~15 short training runs); each is fast once embeddings are cached, but it's
  roughly 15x the training cost of a single class-holdout run. `run_ablation.py` runs 8 probes total (3 novel
  classes x 2 (plain/FiLM), plus LOSO x 2 (plain/FiLM), where each LOSO run is itself ~15 folds).
- The learnable selector's soft cutoff has a fixed `--selector_temperature` (not itself learned) controlling how
  sharp the window-count boundary is; very small values make the gradient near the boundary vanish, very large
  values make the effective count barely distinguishable from a uniform average over all candidate windows.
