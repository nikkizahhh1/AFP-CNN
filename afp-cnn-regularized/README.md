# AFP CNN — Regularized Experiment

Separate experiment copied from `afp-cnn-baseline`. The baseline folder remains untouched.

## Architecture
Input `(batch, 12, 256)` → Conv1d(12→32, k7) → ReLU → MaxPool → Conv1d(32→64, k5) → ReLU → MaxPool → Conv1d(64→128, k3) → ReLU → global average pooling → **Dropout(0.3)** → Linear(128→number of elements).

## Training changes
- Default 75 epochs (previously 30).
- Dropout probability 0.3 before classification.
- Training loss is cross-entropy with moderate inverse-square-root class-frequency weights computed **only from training examples**. Set `--class-weighting none` to disable.
- Early stopping monitors validation macro F1, stopping after 12 epochs without an improvement. Best model is selected by highest validation macro F1.
- Validation/test cross-entropy loss remains unweighted for easier comparison.
- The same 60/20/20 split algorithm, Adam optimizer, learning rate 0.001, and batch size 64.
- Uses a new results directory; don't resume checkpoints from the baseline architecture.

## Run in PowerShell

```powershell
cd afp-cnn-regularized
python -m pip install -r requirements.txt
python train.py --data-dir "C:\Users\Kevin\Desktop\AFP-Proj\Datasets\data\afps" --output-dir "C:\Users\Kevin\Desktop\AFP-Proj\Results\regularized" --epochs 75 --dropout 0.3 --patience 12 --class-weighting sqrt-inverse --batch-size 64
```

For an isolated dropout experiment, use `--class-weighting none` and a different output directory. To compare class weights alone, use `--dropout 0`.

**Research note:** The existing site-level split may place fingerprints from the same crystal structure in both training and validation/testing. If manifest.csv includes a real CIF/structure identifier column, pass `--group-col N` to reduce leakage. Grouped splits may be missing rare classes in training, in which case the script raises an error rather than silently training incorrect weights.

Outputs: `best.pt`, `latest.pt`, `history.csv`, `history.json`, `config.json`, `test_metrics.json`, `per_class_metrics.csv`, `confusion_matrix.csv`, and `test_predictions.csv`. No training has been run just by adding this folder.
