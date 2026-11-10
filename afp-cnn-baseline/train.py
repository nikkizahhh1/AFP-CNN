"""Train a basic 1D CNN to classify the element at an AFP site."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import AFPDataset, load_dataset, make_split
from metrics import classification_metrics, save_csv
from model import AFPConvNet


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Directory containing manifest.csv and .bin AFP files")
    p.add_argument("--output-dir", default="results/baseline")
    p.add_argument("--filename-col", type=int, default=0)
    p.add_argument("--z-col", type=int, default=6)
    p.add_argument("--group-col", type=int, default=None,
                   help="Manifest column for CIF/composition ID; enables leakage-safe group split")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)  # Windows safe
    p.add_argument("--resume", action="store_true", help="Resume from latest.pt")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def inference(model, loader, loss_fn, device):
    model.eval()
    total_loss, total, correct = 0., 0, 0
    truths, predictions = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = logits.argmax(1)
            total_loss += loss.item() * len(y)
            total += len(y)
            correct += (preds == y).sum().item()
            truths.extend(y.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
    return total_loss / max(total, 1), correct / max(total, 1), truths, predictions


def main():
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot see a compatible GPU")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    x, y, classes, groups, filenames = load_dataset(args.data_dir, args.filename_col,
                                                      args.z_col, args.group_col)
    split_file = output / "splits.npz"
    if args.resume:
        if not split_file.exists():
            raise FileNotFoundError(f"Cannot resume without {split_file}")
        splits = np.load(split_file)
        train_idx, val_idx, test_idx = (splits[k] for k in ("train", "val", "test"))
    else:
        train_idx, val_idx, test_idx = make_split(len(y), groups, args.seed, args.group_col is not None)
        np.savez(split_file, train=train_idx, val=val_idx, test=test_idx)
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Empty train/validation/test split. Add more samples/groups.")
    # Use robust training-set scaling (one scalar for all AFP values).
    scale = max(float(np.percentile(np.abs(x[train_idx]), 99)), 1e-6)
    train_loader = DataLoader(AFPDataset(x, y, train_idx, scale), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(AFPDataset(x, y, val_idx, scale), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(AFPDataset(x, y, test_idx, scale), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    model = AFPConvNet(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best_val = 0, -1.
    history = []
    if args.resume:
        checkpoint = torch.load(output / "latest.pt", map_location=device, weights_only=False) if _supports_weights_only() else torch.load(output / "latest.pt", map_location=device)
        if checkpoint["classes"] != classes or checkpoint["filenames"] != filenames:
            raise RuntimeError("Dataset changed since checkpoint: cannot safely resume")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_val = float(checkpoint["best_val"])
        scale = float(checkpoint["scale"])
        # reset loaders to saved scale if training set percentile changed
        for loader in (train_loader, val_loader, test_loader):
            loader.dataset.scale = scale
        history_path = output / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        print(f"Resuming after epoch {start_epoch}")
    print(f"Device: {device}; split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    if args.group_col is None:
        print("Note: site-level split; samples from the same CIF may occur in multiple splits.")
    config = vars(args) | {"element_atomic_numbers": classes, "scale": scale,
                           "split_sizes": [len(train_idx), len(val_idx), len(test_idx)]}
    (output / "config.json").write_text(json.dumps(config, indent=2))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, count = 0., 0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; stop training to avoid corrupt weights")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
            count += len(target)
        val_loss, val_acc, truth, predictions = inference(model, val_loader, loss_fn, device)
        val_report, _, _ = classification_metrics(truth, predictions, classes)
        record = {"epoch": epoch + 1, "train_loss": total_loss / count,
                  "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_report["macro_f1"]}
        history.append(record)
        save_csv(output / "history.csv", history)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch+1:3d}/{args.epochs}: train loss={record['train_loss']:.4f}, "
              f"val loss={val_loss:.4f}, val acc={val_acc:.3f}, macro F1={val_report['macro_f1']:.3f}")
        improved = val_report["macro_f1"] > best_val
        if improved:
            best_val = val_report["macro_f1"]
        checkpoint = {"epoch": epoch + 1, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "classes": classes,
                      "filenames": filenames, "scale": scale, "best_val": best_val}
        torch.save(checkpoint, output / "latest.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")

    # Always evaluate only the best validation-selected model on the untouched test set.
    best = torch.load(output / "best.pt", map_location=device, weights_only=False) if _supports_weights_only() else torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_loss, test_acc, truth, predictions = inference(model, test_loader, loss_fn, device)
    report, rows, cm = classification_metrics(truth, predictions, classes)
    report["loss"] = test_loss
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2))
    save_csv(output / "per_class_metrics.csv", rows)
    np.savetxt(output / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_csv(output / "test_predictions.csv", [
        {"filename": filenames[int(idx)], "actual_Z": classes[int(y[int(idx)])],
         "predicted_Z": classes[int(p)]}
        for idx, p in zip(test_idx, predictions)])
    print(f"TEST (best validation checkpoint): accuracy={test_acc:.3f}, macro F1={report['macro_f1']:.3f}")
    print(f"Saved outputs to {output.resolve()}")


def _supports_weights_only():
    import inspect
    return "weights_only" in inspect.signature(torch.load).parameters


if __name__ == "__main__":
    main()
