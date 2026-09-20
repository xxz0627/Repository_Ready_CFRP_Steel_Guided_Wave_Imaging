import numpy as np
import pandas as pd

from cfrp_gw.config import GridConfig, PDIConfig, ReconstructionConfig
from cfrp_gw.geometry import make_grid
from cfrp_gw.pdi import probabilistic_diagnostic_imaging


def test_pdi_output_shape():
    table = pd.DataFrame([
        {"Tx_X": 50.0, "Tx_Y": 50.0, "Rx_X": 100.0, "Rx_Y": 50.0},
        {"Tx_X": 100.0, "Tx_Y": 50.0, "Rx_X": 150.0, "Rx_Y": 50.0},
    ])
    grid = make_grid(GridConfig(nx=40, ny=8))
    field = probabilistic_diagnostic_imaging(table, np.array([0.1, 0.2]), grid, PDIConfig(), ReconstructionConfig())
    assert field.shape == (8, 40)
    assert np.min(field) >= 0.0
