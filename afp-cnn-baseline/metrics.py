"""Per-class precision/recall/F1 without scikit-learn dependency."""
import csv
from pathlib import Path

import numpy as np


def classification_metrics(truth, pred, classes):
    truth, pred = np.asarray(truth), np.asarray(pred)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (truth, pred), 1)
    tp = cm.diagonal()
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(len(classes)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(classes)), where=(precision + recall) != 0)
    report = {
        "accuracy": float(tp.sum() / max(1, cm.sum())),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    rows = [{"atomic_number": z, "precision": float(p), "recall": float(r),
             "f1": float(f), "support": int(s)}
            for z, p, r, f, s in zip(classes, precision, recall, f1, support)]
    return report, rows, cm


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
