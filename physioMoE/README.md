# physioMoE

Text-conditioned Mixture-of-Experts that predicts NASA-TLX workload scores
from multichannel physiological signals.

![](mixture_of_experts_architecture%201.png)

- **NormWear** ([mosaic-laboratory/normwear](https://huggingface.co/mosaic-laboratory/normwear)) encodes the raw
  `(channels, time)` signal into a 768-d embedding (mean of per-channel CLS tokens). It stays frozen by default —
  it's a 0.2B-parameter foundation model, not something to fine-tune on a small NASA-TLX set.
- A small pretrained sentence encoder (`sentence-transformers/all-MiniLM-L6-v2` by default) encodes the task
  description string.
- The **router** sees both embeddings and outputs a distribution over experts (dense softmax, or top-k sparse
  gating). It only decides *how much to trust each expert* — the routing decision is task-conditioned.
- Each **expert** is an independent MLP that reads the physiological embedding only and predicts the 6 NASA-TLX
  dimensions (mental demand, physical demand, temporal demand, performance, effort, frustration). Expert outputs
  are combined with the router's gate weights into the final prediction.

## project layout
  - **`config.py`** — `ModelConfig` / `TrainConfig` dataclasses and the canonical NASA-TLX dimension list/order
    used everywhere else in the code.
  - **`metrics.py`** — regression metrics (MAE/RMSE/R²) shared by training and evaluation.
  - **`train.py`** — CLI training loop (also installed as the `physioMoE-train` script).
  - **`evaluate.py`** — CLI checkpoint evaluation (also installed as `physioMoE-evaluate`).
  - **`models/`** — the architecture, one file per block of the diagram above:
    - `text_encoder.py` — `TextEncoder`, wraps any Hugging Face encoder model with mean pooling.
    - `normwear_encoder.py` — `NormWearEncoder`, wraps the NormWear foundation model.
    - `router.py` — `Router` (dense or top-k gating) and `load_balancing_loss` (keeps expert usage balanced).
    - `experts.py` — `Expert` / `ExpertBank`, the parallel NASA-TLX prediction heads and their weighted combination.
    - `physio_moe.py` — `PhysioMoE`, wires the four pieces together and defines the training loss.
  - **`data/`** — dataset loading and synthesis:
    - `dataset.py` — `PhysioTLXDataset` (reads a manifest CSV + `.npy` signal files) and a `collate_fn` that
      zero-pads variable-length signals within a batch.
- **`tests/`** — network-free unit tests (router gating, expert combination, full forward/backward pass using
  lightweight fake encoders instead of downloading real models). Run with `uv run pytest`.
- **`checkpoints/`** *(git-ignored, created on demand)* — training output: `best.pt` (model weights + config +
  validation metrics) written by `train.py`.

## Tests

```bash
uv run pytest
```

Tests inject lightweight fake text/NormWear encoders so they run without network access or GPU; they check
router gating math, expert combination, and the full model's forward/backward pass.

## Notes on NormWear compatibility

NormWear's `trust_remote_code=True` modeling code runs eager tensor ops inside `__init__` to infer shapes, which
is incompatible with `transformers`' newer meta-device fast-init path. This project pins
`transformers==4.46.3` (see `pyproject.toml`) to avoid a `NotImplementedError: Cannot copy out of meta tensor`
crash on load.
## Contributors
The code is initially generated using AI, then reviewed and edited by Loqman OUAGAGUE [loqman.ouagague@gmail.com](mailto:loqman.ouagague@gmail.com).