# MambaInterp

Train and compare Mamba-style audio classifiers on ESC-50 with a shared evaluation script.

This repo implements:
- `mamba_spectrogram`: Mamba encoder on log-mel spectrogram patches.

## 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Prepare ESC-50

Download and extract the Kaggle dataset:

- https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50

Expected structure (`--data-root` points to this folder):

```text
ESC-50-master/
  audio/
  esc50.csv
```

## 3) Small validation runs

Run a small validation experiment:

```bash
python -m mambainterp.run_small_experiments ^
  --data-root "E:\path\to\ESC-50-master" ^
  --output-dir "runs\small"
```

To explicitly set models (spectrogram only):

```bash
python -m mambainterp.run_small_experiments ^
  --data-root "E:\path\to\ESC-50-master" ^
  --models mamba_spectrogram ^
  --output-dir "runs\small"
```

## 4) Single training run

```bash
python -m mambainterp.train ^
  --data-root "E:\path\to\ESC-50-master" ^
  --model mamba_spectrogram ^
  --val-fold 1 ^
  --epochs 20 ^
  --batch-size 16
```

## Notes

- The Mamba block follows the core paper structure: input projection, depthwise convolution, selective state-space scan, gated output, residual path.
- This code is designed for fast prototype runs (small runs first, then scaling model size/compute).
- Each training run saves:
  - `history.json` and `history.csv`
  - `training_curves.png` (loss/accuracy/F1/LR vs epoch)
- `run_small_experiments` also saves:
  - `small_experiments_curves.png` (validation curves overlay)
  - `small_experiments_best_accuracy.png` (best-accuracy ranking bar chart)
