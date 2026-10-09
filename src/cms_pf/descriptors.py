from __future__ import annotations

import numpy as np


def radial_profile(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    yy, xx = np.indices(values.shape)
    rr = np.sqrt((xx - n // 2) ** 2 + (yy - n // 2) ** 2).astype(int)
    sums = np.bincount(rr.ravel(), weights=values.ravel())
    counts = np.bincount(rr.ravel())
    return sums / np.maximum(counts, 1)


def field_descriptors(field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    centered = field - field.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    radial = radial_profile(spectrum)
    radii = np.arange(radial.size, dtype=float)
    valid = radii > 0
    denom = radial[valid].sum()
    mean_k = (radii[valid] * radial[valid]).sum() / max(denom, 1e-12)
    characteristic_length = field.shape[0] * dx / max(mean_k, 1e-8)
    gx, gy = np.gradient(field, dx)
    interface_density = np.mean(np.sqrt(gx**2 + gy**2))
    amplitude = np.percentile(field, 95) - np.percentile(field, 5)
    variance = np.var(field)
    return np.asarray(
        [characteristic_length, interface_density, amplitude, variance],
        dtype=np.float32,
    )


def trajectory_descriptors(fields: np.ndarray, dx: float = 1.0) -> np.ndarray:
    return np.stack([field_descriptors(f, dx=dx) for f in fields])

