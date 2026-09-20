import numpy as np

from cfrp_gw.config import PreprocessingConfig
from cfrp_gw.preprocessing import repeated_path_energy


def test_repeated_path_energy_is_finite():
    cfg = PreprocessingConfig()
    t = np.arange(2000) / cfg.sampling_rate_hz
    base = np.sin(2 * np.pi * 100_000.0 * t)
    traces = np.vstack([(1.0 + 0.01 * k) * base for k in range(5)])
    median, energies = repeated_path_energy(traces, cfg)
    assert energies.shape == (5,)
    assert np.all(np.isfinite(energies))
    assert median > 0.0
