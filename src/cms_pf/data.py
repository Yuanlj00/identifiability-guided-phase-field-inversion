from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self, path: str, indices: np.ndarray, parameter_mean=None, parameter_std=None,
        descriptor_mean=None, descriptor_std=None,
    ):
        self.path = path
        self.indices = np.asarray(indices, dtype=np.int64)
        with h5py.File(path, "r") as h5:
            params = h5["parameters"][:]
            self.times = h5["times"][:].astype(np.float32)
            # The formal dataset is small relative to host memory. Preloading
            # each split avoids reopening a gzip-compressed HDF5 file for every
            # sample and keeps all accelerators fed during parallel runs.
            self.fields = h5["fields"][self.indices]
            self.descriptors = h5["descriptors"][self.indices].astype(np.float32)
            self.parameters = params[self.indices].astype(np.float32)
            if descriptor_mean is None:
                train_desc = self.descriptors.copy()
                train_desc = np.sign(train_desc) * np.log1p(np.abs(train_desc))
                descriptor_mean = train_desc.mean(axis=(0, 1))
                descriptor_std = train_desc.std(axis=(0, 1)) + 1e-8
        log_params = np.log(params)
        self.parameter_mean = (
            log_params[self.indices].mean(0) if parameter_mean is None else parameter_mean
        )
        self.parameter_std = (
            log_params[self.indices].std(0) + 1e-8 if parameter_std is None else parameter_std
        )
        self.descriptor_mean = np.asarray(descriptor_mean, dtype=np.float32)
        self.descriptor_std = np.asarray(descriptor_std, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        fields = self.fields[item].astype(np.float32)
        desc = self.descriptors[item].copy()
        params = self.parameters[item]
        fields = (fields - 0.5) / 0.15
        desc = np.sign(desc) * np.log1p(np.abs(desc))
        desc = (desc - self.descriptor_mean) / self.descriptor_std
        target = (np.log(params) - self.parameter_mean) / self.parameter_std
        times = np.log1p(self.times)
        times = times / max(times.max(), 1e-8)
        return {
            "fields": torch.from_numpy(fields[:, None]),
            "descriptors": torch.from_numpy(desc),
            "times": torch.from_numpy(times.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "index": idx,
        }


def grouped_split(parameter_ids: np.ndarray, seed: int, train=0.70, validation=0.15):
    unique = np.unique(parameter_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(len(unique) * train)
    n_val = int(len(unique) * validation)
    train_ids = unique[:n_train]
    val_ids = unique[n_train:n_train + n_val]
    test_ids = unique[n_train + n_val:]
    return tuple(
        np.flatnonzero(np.isin(parameter_ids, ids))
        for ids in (train_ids, val_ids, test_ids)
    )
