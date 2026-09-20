import numpy as np
import pandas as pd

from cfrp_gw.config import SSBConfig
from cfrp_gw.ssb import build_screened_observations, construct_ssb_observations


def _table():
    rows = []
    for i, e in enumerate(np.linspace(1.0, 2.0, 10)):
        rows.append({"Tx_X": float(i), "Tx_Y": 0.0, "Rx_X": float(i + 50), "Rx_Y": 0.0, "Energy": e})
    return pd.DataFrame(rows)


def test_ssb_reference_is_current_scan_quantile():
    cfg = SSBConfig(percentile=0.70)
    out = construct_ssb_observations(_table(), cfg)
    expected = np.quantile(np.linspace(1.0, 2.0, 10), 0.70)
    assert np.allclose(out["SSB_Reference_Energy"], expected)


def test_mad_screen_adds_screened_observation():
    out = build_screened_observations(_table(), SSBConfig())
    assert "Attenuation" in out.columns
    assert np.all(out["Attenuation"].to_numpy() >= 0.0)
