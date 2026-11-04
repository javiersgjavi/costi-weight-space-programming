"""Inference data matching the repository and TSL 0.9.5 preprocessing.

No TSL sparse/CUDA dependencies are necessary. Source-specific splits and the
original CustomScaler (including its additional 70% slice) are preserved.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sample_mask(shape, p, p_noise, min_seq, max_seq, seed):
    """Same generator consumption and per-column fault lengths as TSL."""
    rng = np.random.default_rng(seed)
    if p == 0:
        # default_rng is PCG64: one float64 sample consumes one generator step.
        rng.bit_generator.advance(int(np.prod(shape)))
        mask = np.zeros(shape, dtype=bool)
    else:
        mask = rng.random(shape) < p
    for col in range(shape[1]):
        idx = np.flatnonzero(mask[:, col])
        if not len(idx):
            continue
        length = min_seq + int(rng.integers(max_seq - min_seq)) if max_seq > min_seq else min_seq
        expanded = np.unique(np.concatenate([np.arange(i, i + length) for i in idx]))
        mask[np.clip(expanded, 0, shape[0] - 1), col] = True
    return mask | (rng.random(shape) < p_noise)


def temporal_mean(frame):
    x = frame.copy()
    idx = x.index
    conditions = [[idx.year, idx.isocalendar().week, idx.hour],
                  [idx.year, idx.month, idx.hour], [idx.month, idx.hour], [idx.hour]]
    for groups in conditions:
        if not x.isna().values.any():
            break
        x = x.fillna(x.groupby(groups).transform("mean"))
    return x.ffill().bfill()


def air_splits(index, starts, window, months=(3, 6, 9, 12)):
    first, last = index[starts].month, index[starts + window - 1].month
    test = np.flatnonzero(np.isin(first, months) & np.isin(last, months))
    train = np.flatnonzero(~np.isin(first, months) & ~np.isin(last, months))
    val_len = int(0.1 * len(train)) // len(months)
    delta = np.diff(test)
    boundaries = test[1:][np.flatnonzero(delta > delta.min())]
    if len(boundaries) < len(months):
        boundaries = np.insert(boundaries, 0, test[0])
    val = np.concatenate([np.arange(i - val_len, i) - window for i in boundaries]) % len(starts)
    # With stride==window the windows never overlap, so this is exact.
    train = train[~np.isin(train, val)]
    return train, val, test


@dataclass
class BenchmarkData:
    values: np.ndarray
    observed: np.ndarray
    condition: np.ndarray
    starts: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    window: int
    bias: torch.Tensor
    scale: torch.Tensor
    name: str

    def indices(self, split, limit=None):
        if split == "calibration":
            ids = self.validation[::2]
        elif split == "confirmation":
            ids = self.validation[1::2]
        elif split == "test":
            ids = self.test
        else:
            raise ValueError(split)
        if limit is not None and limit < len(ids):
            ids = ids[np.linspace(0, len(ids) - 1, limit, dtype=int)]
        return ids

    def batch(self, ids):
        rows = self.starts[np.asarray(ids)][:, None] + np.arange(self.window)
        target = torch.from_numpy(self.values[rows, :, None])
        condition = torch.from_numpy(self.condition[rows, :, None])
        observed = torch.from_numpy(self.observed[rows, :, None])
        x = torch.where(condition, (target - self.bias) / self.scale, 0)
        masked = torch.where(condition, x, torch.nan).squeeze(-1).numpy()
        interpolation = np.stack([
            pd.DataFrame(v).interpolate(axis=0).bfill().ffill().fillna(0).values
            for v in masked])
        return {"x": x, "mask": condition, "eval_mask": observed & ~condition,
                "x_interpolated": torch.from_numpy(interpolation[..., None]),
                "target": target}


def load_data(name, root=Path("artifacts/data")):
    root = Path(root)
    if name.startswith("mimic"):
        paths = [root / "clinical" / s / "x.npy" for s in ("train", "val", "test")]
        arrays = [np.load(p, mmap_mode="r") for p in paths]
        lengths = [len(a) for a in arrays]
        values = np.concatenate([a.transpose(0, 2, 1).astype(np.float32).reshape(-1, 40)
                                 for a in arrays])
        window = 48
        starts = np.arange(0, len(values) - window + 1, window)
        observed = ~np.isnan(values)
        values = np.nan_to_num(values, nan=0)
        missing = sample_mask((*values.shape, 1), 0, .25, 48, 96, 42)[..., 0]
        ntrain, nval = lengths[:2]
        train = np.arange(ntrain)
        val = np.arange(ntrain, ntrain + nval)
        test = np.arange(ntrain + nval, len(starts))
    elif name.startswith("aqi"):
        raw = np.load(root / "air.npz")
        index = pd.DatetimeIndex(raw["pm25_index"].astype("datetime64[ns]"))
        frame = pd.DataFrame(raw["pm25"], index=index)
        observed = ~np.isnan(frame.values)
        values = temporal_mean(frame).values.astype(np.float32)
        missing = raw["eval_mask"].astype(bool)
        window = 36
        starts = np.arange(0, len(values) - window + 1, window)
        train, val, test = air_splits(index, starts, window)
    elif name.startswith("metr"):
        raw = np.load(root / "metr.npz")
        index = pd.DatetimeIndex(raw["df_index"].astype("datetime64[ns]"))
        frame = pd.DataFrame(raw["df"], index=index)
        frame = frame.reindex(pd.date_range(index[0], index[-1], freq="5min"))
        values = frame.values.astype(np.float32)
        observed = values != 0
        if not np.isfinite(values).all():
            raise ValueError("METR-LA has non-finite values after source reindex")
        window = 24
        starts = np.arange(0, len(values) - window + 1, window)
        point = name.endswith("point")
        missing = sample_mask((*values.shape, 1), 0 if point else .0015,
                              .25 if point else .05, 12, 48, 9101112)[..., 0]
        nt = int(.2 * len(starts))
        nv = int(.1 * (len(starts) - nt))
        test_start, val_start = len(starts) - nt, len(starts) - nt - nv
        train = np.arange(val_start - 1)
        val = np.arange(val_start, test_start - 1)
        test = np.arange(test_start, len(starts))
    else:
        raise ValueError(f"Out-of-scope dataset {name}")
    rows = (starts[train, None] + np.arange(window)).reshape(-1)
    rows = np.unique(rows)
    fitting_rows = rows[:int(len(rows) * .7)]
    fitting_values = torch.from_numpy(values[fitting_rows])
    bias = fitting_values.mean(0).reshape(1, 1, -1, 1)
    scale = fitting_values.std(0).reshape(1, 1, -1, 1)
    if not bool((scale > 0).all()) or not bool(scale.isfinite().all()):
        raise ValueError("Source scaler produced invalid scale")
    condition = observed & ~missing
    if np.intersect1d(val, test).size or np.intersect1d(train, val).size:
        raise ValueError("Overlapping data partitions")
    return BenchmarkData(values, observed, condition, starts, train, val, test,
                         window, bias, scale, name)
