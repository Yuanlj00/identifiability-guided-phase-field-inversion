import numpy as np

from cms_pf.solver import CHConfig, simulate, symmetry_transform


def test_mass_energy_and_symmetry():
    cfg = CHConfig(n=32, dt=0.08, n_steps=160, n_snapshots=8)
    theta = (1.0, 1.4, 0.7)
    a = simulate(*theta, 7, cfg)
    b = simulate(*symmetry_transform(*theta, 3.0), 7, cfg)
    assert np.max(np.abs(a["masses"] - a["masses"][0])) < 1e-10
    assert np.all(np.diff(a["energies"]) <= 1e-8)
    assert np.sqrt(np.mean((a["fields"] - b["fields"]) ** 2)) < 2e-6

