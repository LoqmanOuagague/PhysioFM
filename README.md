# PhysioFM — Cognitive Workload Estimation (Internship Project)

This repository is my internship workspace, built on top of **[NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)**, a foundation model for multivariate wearable physiological sensing ([Luo et al., ACM Trans. Comput. Healthcare, 2026](https://dl.acm.org/doi/10.1145/3803808)). NormWear is pretrained on large collections of PPG, ECG, EEG, GSR, and IMU signals and produces general-purpose embeddings that transfer to many downstream health applications, either **zero-shot** (via a text-aligned fusion head) or **fullshot** (by training a small task-specific head on frozen embeddings).

My internship goal is to train a model model for **cognitive workload estimation using physiological signals and contextual data**: predicting how mentally demanding a task is for a person, directly from wearable physiological signals (heart rate, GSR, respiration, temperature, wrist acceleration, ...), without relying on self-report during the task itself.

- **Intern:** Loqman OUAGAGUE ([loqman.ouagague@gmail.com](mailto:loqman.ouagague@gmail.com))
- **Host:** LS2N Lab, École Centrale Nantes
- **Tutors**: 
    - AUGEREAU Olivier — Associate professor in Ecole Centrale Nantes, [olivier.augereau@ec-nantes.fr](mailto:olivier.augereau@ec-nantes.fr)
    - BAGHDADI Sarra — Phd Candidate, [Sarra.BAGHDADI@b-com.com](mailto:Sarra.BAGHDADI@b-com.com)

> This is a fork/derivative of the [original NormWear repository](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear). The core model, pretraining pipeline, and zero-shot fusion mechanism are the authors' work I just adapted some snippets of it — see [`NormWear/README.md`](./NormWear/README.md) for the original documentation. Everything else is my contribution as part of this internship.

## Why cognitive workload estimation?

Cognitive workload is the amount of mental effort a task demands relative to a person's capacity. It is commonly measured post-hoc with the **NASA Task Load Index (NASA-TLX)**, a self-report questionnaire scoring six dimensions: *mental demand, physical demand, temporal demand, performance, effort,* and *frustration*. Self-report is retrospective and disruptive, which limits its use in real-time or safety-critical settings (driving, aviation, control rooms, human-robot interaction).

The hypothesis behind this project: a wearable sensing foundation model pretrained on generic physiological signals should already encode features (heart-rate variability, skin conductance responses, breathing patterns, micro-movements) that correlate with mental workload.

## Repository layout

```
NormWear/                      # Core model, forked from the original repo
├── main_model.py               # NormWearModel: signal → patch embeddings
├── modules/                    # Encoder, patch embedding, positional encoding, preprocessing
├── pretrain_main.py            # Self-supervised pretraining entry point
├── pretrain_pipeline/          # Pretraining dataset & training loop
├── zero_shot/                  # Text-aligned zero-shot inference (MSiTF fusion)
├── downstream_main.py          # Fullshot evaluation entry point (embeddings + linear/shallow probe)
├── downstream_pipeline/        # Task specs, embedding extraction, probing, finetuning
├── baseline_models/            # Baselines used for comparison (CrossViT, TFC, ...)
├── weights/                    # Model checkpoints (gitignored, see below)
└── README.md                   # Original upstream documentation

data/                           # Downstream & pretraining datasets (gitignored, see below)

utils/
├── process_cogload.py          # My preprocessing script: raw CogLoad1 CSVs → NormWear-ready windows
└── process_wesad.py            # My preprocessing script: raw WESAD .pkl files → NormWear-ready windows

physioMoE/                      # Text-conditioned Mixture-of-Experts for NASA-TLX prediction (see below)
├── models/                     # Router, experts, text/NormWear encoders, PhysioMoE wiring
├── data/                       # PhysioTLXDataset (manifest CSV + .npy signals) used by train.py/evaluate.py
├── train.py / evaluate.py      # CLI training & checkpoint evaluation
├── tests/                      # Network-free unit tests (router gating, expert combination, forward/backward)
└── Design_notes.md             # Design rationale and open questions

CogLoad1 fullshot evaluation/  # Autonomous-agent ("autoresearch") LoRA fine-tuning of NormWear on CogLoad1
├── train.py                    # LoRA-adapted NormWear backbone + NASA-TLX head (the file the agent edits)
├── prepare.py                  # Fixed CWT spectrogram caching + held-out test evaluation (not modified)
└── program.md                  # Agent instructions for the autonomous experiment loop

docs/
├── Running_zeroshot_evaluation_of_NormWear.md   # Detailed zero-shot / HPC walkthrough
├── evaluation_NormWear.md                       # Fullshot results & hyperparameters (CogLoad1)
├── job_script_example.sh                        # Annotated Slurm job script
└── Tutorial on supercomputing[FR]/               # French-language HPC cluster guide

job_script.sh                   # My working Slurm script (gitignored, machine-specific)
sweep.yaml                      # Weights & Biases hyperparameter sweep config
.env                            # Local paths to model weights (see below)
```

## Installation

### Using uv (Recommended)

```sh
# Creating the virtual environment
uv venv .venv
# Activating the virtual environment
source .venv/bin/activate
# Installing dependencies
uv sync
```

### Using pip

```sh
# Creating the virtual environment
python -m venv .venv
# Activating the virtual environment
source .venv/bin/activate
# Upgrading pip [Optional]
python -m pip install --upgrade pip
# Installing dependencies
pip install -r requirements.txt
```

## Model weights

The pretrained NormWear backbone and the zero-shot MSiTF fusion checkpoint are available from the [GitHub release of the original repo](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha), or from [HuggingFace](https://huggingface.co/mosaic-laboratory/normwear):

```sh
hf download mosaic-laboratory/normwear
```

Point the `.env` file at wherever the weights end up (see below). To use a custom directory:

```sh
hf download mosaic-laboratory/normwear --local-dir YOUR_DIRECTORY
```

## Datasets

### Downstream datasets (WESAD, UCI-HAR, ...)

The processed downstream datasets from the original paper can be downloaded from [Google Drive](https://drive.google.com/file/d/1Mojf_iby8FnUydogwUE-b321fB1V6SGK/view?usp=sharing).

### Pretraining dataset

Available from [Google Drive](https://drive.google.com/file/d/1WBlyweezkYm16PR3UFrO85XrZCKjqWmP/view?usp=sharing).

### CogLoad1

CogLoad1 is not part of the original NormWear release; it needs to be preprocessed locally with [`utils/process_cogload.py`](utils/process_cogload.py). Starting from raw per-participant sensor CSVs (`hr`, `gsr`, `rr`, `temperature`, and 3-axis wrist acceleration) plus a `personality_performance.csv` file with NASA-TLX scores per (task, level) segment, the script:

1. Resamples each segment from ~1 Hz to **65 Hz** (NormWear's pretraining sampling rate);
2. Splits it into non-overlapping **6-second windows** (matching NormWear's pretraining segment length), dropping a trailing partial window;
3. Randomly assigns each segment (not individual windows, so no leakage across the split) to train/test, default 80/20;
4. Writes one `.npy` file per window plus a manifest CSV (`sample_id`, `task_text`, `signal_path`, the six NASA-TLX scores, and personality/demographic columns), in the format `physioMoE.data.dataset.PhysioTLXDataset` expects.

```sh
python3 utils/process_cogload.py \
    --raw_dir path/to/train/raw \
    --out_dir data/Cogload \
    --performance_csv path/to/personality_performance.csv \
    --train_split 0.8
```

This currently yields ~3,900 windows across all participants/tasks/levels (roughly 3,100 train / 800 test). Pass `--split_mode subject_independent` to isolate one random participant as the entire test set (leave-one-subject-out) instead of the default per-segment random split.

### WESAD (from raw)

As an alternative to the pre-processed Google Drive download above, WESAD can also be built directly from the raw per-subject release (`SX.pkl`, `SX_readme.txt`, `SX_quest.csv`, ... per subject; place it under `data/WESAD_RAW/`) with [`utils/process_wesad.py`](utils/process_wesad.py). Starting from the synchronised chest (RespiBAN, 700 Hz) and wrist (Empatica E4) signals in each subject's `SX.pkl`, the script:

1. Locates the contiguous baseline/stress/amusement segments from the per-sample study-protocol label (meditation and the transient/undefined labels are dropped, matching the standard WESAD 3-class stress-detection setup);
2. Resamples the chest channels (700 Hz) and the wrist EDA/TEMP channels (4 Hz) independently to **65 Hz** and stacks them into 10 channels (chest ACC x/y/z, ECG, EMG, EDA, Temp, Resp + wrist EDA, TEMP);
3. Splits each segment into non-overlapping **6-second windows** (matching NormWear's pretraining segment length), dropping a trailing partial window;
4. Randomly assigns each segment to train/test and writes one `.npy` file per window plus a manifest CSV — the same `subject_dependent` / `subject_independent` (leave-one-subject-out) split modes as `process_cogload.py`.

**Note**: this second alternative is provided because in the preprocessed dataset provided by the authors segments **are not** assigned to the participant. Thus, it is impossible to evaluate the model by leaving one subject out. Moreover, it does not allow the calculation of the true accuracy because it evaluate accuracy on windows where as the real senario is to evaluate on the whole recorded signal.
```sh
python3 utils/process_wesad.py \
    --raw_dir data/WESAD_RAW \
    --out_dir data/WESAD \
    --train_split 0.8
```

## Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `NORMWEAR_PATH` | Local directory of the NormWear backbone weights (HuggingFace download) |
| `MSITF_CKPT_PATH` | Local path to the zero-shot MSiTF fusion checkpoint |
| `TINYLLAMA_PATH` | Local directory of [`muzammil-eds/tinyllama-2.5T-Clinical-v2`](https://huggingface.co/muzammil-eds/tinyllama-2.5T-Clinical-v2), used as the text encoder for zero-shot inference |
| `MODEL_CKPT_PATH` | Path to a fullshot/finetuned model checkpoint |

## Running zero-shot evaluation

Zero-shot inference matches signal embeddings against text descriptions of candidate outcomes (e.g. "the person is under stress") without any task-specific training. See [`docs/Running_zeroshot_evaluation_of_NormWear.md`](docs/Running_zeroshot_evaluation_of_NormWear.md) for the full walkthrough (weights, TinyLlama setup, HPC job script). Quick example:

```sh
CUDA_VISIBLE_DEVICES=0 python3 -m NormWear.zero_shot.zero_shot_inference_parallel normwear --dataset wesad --times 3
```

`zero_shot_inference.py` uses GitHub-released weights, `zero_shot_inference_HF.py` uses the HuggingFace weights, and `_parallel` runs faster and is the recommended entry point on HPC.

## Fullshot LoRA fine-tuning on CogLoad1 (autoresearch)

[`CogLoad1 fullshot evaluation/`](CogLoad1%20fullshot%20evaluation/) fine-tunes the NormWear backbone with LoRA adapters plus a small NASA-TLX regression head directly on CogLoad1's raw signals (as opposed to the frozen-embedding + linear-probe path used by `NormWear/downstream_main.py`). It follows the ["autoresearch"](https://github.com/karpathy/nanochat) methodology (adapted from Karpathy's nanochat repo): a coding agent is pointed at [`program.md`](CogLoad1%20fullshot%20evaluation/program.md) and left to iterate on `train.py` autonomously — each run trains for a fixed 10-minute wall-clock budget, is evaluated on the held-out test split via `prepare.py`'s frozen `evaluate()`, and is kept or discarded based on `overall_mae`, with every attempt logged to an untracked `results.tsv`. `prepare.py` (CWT spectrogram caching + the ground-truth evaluation) is off-limits to the agent; only `train.py` (LoRA rank/targets, head architecture, optimizer, hyperparameters, ...) is fair game. See that folder's [`README.md`](CogLoad1%20fullshot%20evaluation/README.md) for the full methodology and `program.md` for the exact experiment loop.

## Roadmap: physioMoE

[`physioMoE/`](physioMoE/) is a **Mixture-of-Experts** extension inspired by [Mixtral](https://arxiv.org/abs/2401.04088) that already has an initial implementation (router, expert bank, text/NormWear encoders, training/evaluation CLIs, unit tests — see [`physioMoE/README.md`](physioMoE/README.md)): it routes physiological embeddings through multiple expert sub-networks (each specializing in patterns from different datasets/tasks), conditioned by a text encoding of the task and its context, to predict NASA-TLX workload scores while keeping the number of active parameters small. Open design questions still being iterated on, tracked in [`physioMoE/Design_notes.md`](physioMoE/Design_notes.md), include:

- **Resampling robustness** — making the model less dependent on a fixed 65 Hz input rate.
- **Fusion mechanism** for combining per-channel signal embeddings (averaging, CNN, CLS-token fusion).
- **Router network** architecture and the text encoder used to condition it.
- **Task-context prompting** — how the text description of a task/environment is phrased, and how much it matters.
- **Dataset splitting strategy** — classic train/test vs. leave-one-participant-out.
- **Training strategy** — joint end-to-end training vs. training each expert independently then freezing it while training the router.

## Citation

If you use NormWear in your research, please cite the original paper:

```bibtex
@article{10.1145/3803808,
  author = {Luo, Yunfei and Chen, Yuliang and Salekin, Asif and Rahman, Tauhidur},
  title = {Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals},
  year = {2026},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3803808},
  doi = {10.1145/3803808},
  journal = {ACM Trans. Comput. Healthcare},
  month = mar,
  keywords = {Machine Learning, Deep Learning, Digital Health, Time Series Modeling, Signal Processing}
}
```

