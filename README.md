# MambaInterp

Train and compare Mamba-style audio classifiers on ESC-50 with a shared evaluation script.

This repo implements:
- `mamba_waveform`: Mamba encoder on raw waveform patches.
- `mamba_spectrogram`: Mamba encoder on log-mel spectrogram patches.
- `mamba_hybrid`: Late-fusion waveform + spectrogram Mamba.
- `cnn_waveform` and `cnn_spectrogram`: standard CNN baselines.

## 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Prepare ESC-50

Download and extract the Kaggle dataset:

- https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50

Expected structure (any parent directory works as long as `meta/esc50.csv` and `audio/` are inside):

```text
ESC-50-master/
  audio/
  meta/
    esc50.csv
```

## 3) Small validation runs

Run all small experiments (same fold, same evaluation logic):

```bash
python -m mambainterp.run_small_experiments ^
  --data-root "E:\path\to\ESC-50-master" ^
  --output-dir "runs\small"
```

This runs:
- `cnn_waveform`
- `cnn_spectrogram`
- `mamba_waveform`
- `mamba_spectrogram`

Use `--include-hybrid` to also run `mamba_hybrid`.

To run only one model (example: spectrogram Mamba):

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
  --model mamba_waveform ^
  --val-fold 1 ^
  --epochs 20 ^
  --batch-size 16
```

## Notes

- The Mamba block follows the core paper structure: input projection, depthwise convolution, selective state-space scan, gated output, residual path.
- This code is designed for fast prototype comparisons (small runs first, then scaling model size/compute).
- Each training run saves:
  - `history.json` and `history.csv`
  - `training_curves.png` (loss/accuracy/F1/LR vs epoch)
- `run_small_experiments` also saves:
  - `small_experiments_curves.png` (validation curves overlay)
  - `small_experiments_best_accuracy.png` (best-accuracy ranking bar chart)
