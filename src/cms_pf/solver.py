from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CHConfig:
    n: int = 64
    dx: float = 1.0
    dt: float = 0.2
    n_steps: int = 900
    n_snapshots: int = 12
    mean_composition: float = 0.5
    initial_noise: float = 0.025


def free_energy_density(c: np.ndarray, A: float) -> np.ndarray:
    return A * c**2 * (1.0 - c) ** 2


def free_energy_derivative(c: np.ndarray, A: float) -> np.ndarray:
    return 2.0 * A * c * (1.0 - c) * (1.0 - 2.0 * c)


def wave_numbers(n: int, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    return kx, ky, k2


def total_energy(c: np.ndarray, A: float, kappa: float, dx: float) -> float:
    kx, ky, _ = wave_numbers(c.shape[0], dx)
    c_hat = np.fft.fft2(c)
    gx = np.fft.ifft2(1j * kx * c_hat).real
    gy = np.fft.ifft2(1j * ky * c_hat).real
    density = free_energy_density(c, A) + 0.5 * kappa * (gx**2 + gy**2)
    return float(density.sum() * dx**2)


def logarithmic_snapshot_steps(n_steps: int, n_snapshots: int) -> np.ndarray:
    raw = np.geomspace(1, n_steps, n_snapshots)
    steps = np.unique(np.rint(raw).astype(int))
    if steps[-1] != n_steps:
        steps = np.append(steps, n_steps)
    return steps


def simulate(
    A: float,
    kappa: float,
    mobility: float,
    seed: int,
    config: CHConfig,
) -> dict[str, np.ndarray]:
    """Semi-implicit Fourier-spectral constant-mobility Cahn-Hilliard solver."""
    rng = np.random.default_rng(seed)
    c = config.mean_composition + config.initial_noise * rng.standard_normal(
        (config.n, config.n)
    )
    c -= c.mean() - config.mean_composition
    _, _, k2 = wave_numbers(config.n, config.dx)
    k4 = k2**2
    denominator = 1.0 + config.dt * mobility * kappa * k4
    save_steps = logarithmic_snapshot_steps(config.n_steps, config.n_snapshots)
    save_set = set(save_steps.tolist())
    fields: list[np.ndarray] = []
    times: list[float] = []
    energies: list[float] = []
    masses: list[float] = []

    for step in range(1, config.n_steps + 1):
        c_hat = np.fft.fft2(c)
        nonlinear_hat = np.fft.fft2(free_energy_derivative(c, A))
        c_hat = (c_hat - config.dt * mobility * k2 * nonlinear_hat) / denominator
        c = np.fft.ifft2(c_hat).real
        if step in save_set:
            fields.append(c.astype(np.float32))
            times.append(step * config.dt)
            energies.append(total_energy(c, A, kappa, config.dx))
            masses.append(float(c.mean()))

    return {
        "fields": np.stack(fields),
        "times": np.asarray(times, dtype=np.float32),
        "energies": np.asarray(energies, dtype=np.float64),
        "masses": np.asarray(masses, dtype=np.float64),
    }


def symmetry_transform(
    A: float, kappa: float, mobility: float, lam: float
) -> tuple[float, float, float]:
    return lam * A, lam * kappa, mobility / lam

