import numpy as np
import pandas as pd

from cfrp_gw.config import GridConfig, ReconstructionConfig
from cfrp_gw.geometry import build_ray_matrix
from cfrp_gw.reconstruction import sirt, tv_sirt


def test_ray_matrix_and_reconstruction_shapes():
    table = pd.DataFrame(
        [
            {"Tx_X": 50.0, "Tx_Y": 50.0, "Rx_X": 100.0, "Rx_Y": 50.0},
            {"Tx_X": 100.0, "Tx_Y": 50.0, "Rx_X": 150.0, "Rx_Y": 75.0},
            {"Tx_X": 150.0, "Tx_Y": 50.0, "Rx_X": 200.0, "Rx_Y": 0.0},
        ]
    )
    gcfg = GridConfig(nx=40, ny=8, ray_samples=50)
    A, _ = build_ray_matrix(table, gcfg)
    assert A.shape == (3, 320)
    b = np.array([0.05, 0.08, 0.04])
    rcfg = ReconstructionConfig(iterations=3, tv_inner_iterations=2)
    s = sirt(A, b, (8, 40), rcfg)
    t = tv_sirt(A, b, (8, 40), rcfg)
    assert s.shape == (8, 40)
    assert t.shape == (8, 40)
    assert np.min(s) >= 0.0
    assert np.min(t) >= 0.0
