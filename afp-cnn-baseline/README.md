# AFP CNN Baseline

A **minimal 1D convolutional neural network** for predicting an atom's element from its atomic fingerprint (AFP). No VAE, residual blocks, attention, or auxiliary prediction heads.

## Architecture

`(batch, 12, 256) AFP → Conv1d(12→32, k=7) → ReLU → MaxPool → Conv1d(32→64, k=5) → ReLU → MaxPool → Conv1d(64→128, k=3) → ReLU → global average pool → Linear(128→number of elements)`

The number of classes is discovered from atomic numbers (Z) in the dataset (e.g., 34 for your condensed dataset). `r_i0` and `r_min_ratio` in the binary record are **not used** in this initial baseline.

## Data you already have

Point `--data-dir` at the **existing** `data/afps` folder, which contains `manifest.csv` and its `.bin` files; do **not** copy this data into Git.

- Manifest **column 0** (zero-indexed): relative filename; **column 6**: target atomic number Z.
- Every `.bin` record contains **3074 little-endian float32s**: first 3072 AFP values reshape to `(12,256)`; last two are `r_i0` and `r_min_ratio`.
- If your manifest columns differ, override `--filename-col` and `--z-col`.
- Only elements actually present in this dataset are modeled. The saved checkpoint includes the Z ↔ class-index mapping.

## Setup (Windows CMD or PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**GPU note:** On older GPUs (e.g. NVIDIA GTX 960), use a PyTorch CUDA build compatible with your GPU/driver. If PyTorch reports no CUDA device, training runs on CPU. You can reuse your already-working CUDA environment instead of installing a new one. `numpy<2` avoids the incompatibility you saw with older PyTorch.

## Train

```powershell
python train.py --data-dir "C:\path\to\data\afps" --output-dir "C:\path\to\cnn-results\baseline" --epochs 30 --batch-size 64
```

- Deterministic **60% train / 20% validation / 20% test**, selected by seeded shuffle, by default.
- To prevent samples from the same structure/composition appearing in multiple splits, pass `--group-col N` for the manifest column containing a **CIF/structure identifier or composition**. If the CSV does not contain such a column, default site-level splitting does **not** protect against structure leakage.
- Scales inputs using the **99th percentile absolute AFP value from training samples only** (one scalar, preserving amplitude differences).
- Trains using cross-entropy and Adam. Selects the best checkpoint by **validation macro-F1** (useful for class imbalance).
- Output folder contains `best.pt`, `latest.pt`, `splits.npz`, `history.csv`, `test_metrics.json`, `per_class_metrics.csv`, `confusion_matrix.csv`, `test_predictions.csv`, `config.json`.

## Resume after stopping

```powershell
python train.py --data-dir "C:\path\to\data\afps" --output-dir "C:\path\to\cnn-results\baseline" --epochs 60 --resume
```

This resumes at the **last completed epoch**, restoring model and optimizer; it is not a bit-for-bit repeat of an uninterrupted run because minibatch RNG state is not restored. Keep the same dataset, batch size, and architecture. `--epochs 60` means 60 total epochs.

## Push code only to GitHub

```bash
git init
git add .gitignore README.md requirements.txt model.py data.py metrics.py train.py
git commit -m "Add minimal AFP CNN baseline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/afp-cnn-baseline.git
git push -u origin main
```

Create the empty GitHub repo first. Do **not** commit AFP binaries, manifests, result folders, or virtual environments. `.gitignore` prevents *new* matching files from being added; it cannot erase files already present in Git history.

## Next iterations

1. `cnn-baseline` (this implementation)
2. `cnn-deep` (more convolutional layers)
3. `cnn-residual` (skip connections)
4. `cnn-attention` (attention over AFP perspectives)

For a fair comparison, **reuse the same dataset, group definitions, and splits** across experiments.
