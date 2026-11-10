"""Load the existing AFP .bin format + manifest.csv (no data conversion)."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

AFP_SHAPE = (12, 256)
RECORD_FLOATS = 3074  # 3072 AFP + r_i0 + r_min_ratio


def load_dataset(data_dir: str, filename_col: int = 0, z_col: int = 6,
                 group_col: int | None = None):
    root = Path(data_dir).expanduser().resolve()
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"No manifest.csv in {root}")

    data, zs, groups, filenames = [], [], [], []
    skipped = {"header_or_invalid": 0, "missing_file": 0, "bad_binary": 0}
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            needed = [filename_col, z_col] + ([] if group_col is None else [group_col])
            if not row or len(row) <= max(needed):
                skipped["header_or_invalid"] += 1
                continue
            try:
                z = int(float(row[z_col]))
            except ValueError:  # header or malformed row
                skipped["header_or_invalid"] += 1
                continue
            if not 1 <= z <= 118:
                skipped["header_or_invalid"] += 1
                continue
            name = row[filename_col].strip().replace('\\', '/')
            path = root / name
            if not path.is_file():
                skipped["missing_file"] += 1
                continue
            values = np.fromfile(path, dtype="<f4")
            if values.size != RECORD_FLOATS or not np.isfinite(values).all():
                skipped["bad_binary"] += 1
                continue
            data.append(values[:3072].reshape(AFP_SHAPE))
            zs.append(z)
            filenames.append(name)
            groups.append(row[group_col].strip() if group_col is not None else name)

    if not data:
        raise RuntimeError(f"No valid AFP files found. Skips: {skipped}. "
                           "Check --data-dir, --filename-col, and --z-col.")
    x = np.stack(data).astype(np.float32, copy=False)
    # Fixed scale across samples; avoid per-sample normalization erasing amplitude.
    classes = sorted(set(zs))
    target_map = {z: i for i, z in enumerate(classes)}
    y = np.array([target_map[z] for z in zs], dtype=np.int64)
    print(f"Loaded {len(y):,} AFPs, {len(classes)} element classes. Skipped: {skipped}")
    return x, y, classes, np.asarray(groups), filenames


class AFPDataset(Dataset):
    def __init__(self, x, y, indices, scale):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices)
        self.scale = float(scale)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ix):
        j = int(self.indices[ix])
        # Global training-only normalization makes AMP and learning stable.
        return torch.from_numpy(np.ascontiguousarray(self.x[j] / self.scale)), int(self.y[j])


def make_split(n: int, groups: np.ndarray, seed: int, split_by_group: bool):
    """60/20/20 deterministic split. Optional group isolation for composition/CIF."""
    rng = np.random.default_rng(seed)
    if not split_by_group:
        order = rng.permutation(n)
        n_train, n_val = int(n * 0.60), int(n * 0.20)
        return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    unique, inv = np.unique(groups, return_inverse=True)
    if len(unique) < 3:
        raise ValueError("Group split needs at least 3 distinct group IDs.")
    # Balance splits by sample counts while keeping every group entirely in one split.
    grouped_indices = [np.flatnonzero(inv == i) for i in range(len(unique))]
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=len, reverse=True)
    buckets = [[], [], []]
    counts = np.zeros(3, dtype=np.int64)
    target = np.array([0.60, 0.20, 0.20]) * n
    for idx in grouped_indices:
        ratios = counts / target
        choice = int(np.argmin(ratios))
        buckets[choice].extend(idx.tolist())
        counts[choice] += len(idx)
    return tuple(rng.permutation(np.asarray(b, dtype=np.int64)) for b in buckets)
