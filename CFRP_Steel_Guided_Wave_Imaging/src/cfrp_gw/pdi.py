from __future__ import annotations

import numpy as np

from .config import PDIConfig, ReconstructionConfig
from .geometry import Grid
from .reconstruction import suppress_background


def probabilistic_diagnostic_imaging(path_table, observations: np.ndarray, grid: Grid, pdi_cfg: PDIConfig, recon_cfg: ReconstructionConfig) -> np.ndarray:
    """Elliptical path-influence PDI using the same path observations as SIRT/TV-SIRT."""
    beta = float(pdi_cfg.shape_factor_beta)
    if beta <= 1.0:
        raise ValueError("PDI shape factor beta must be greater than 1")

    obs = np.asarray(observations, dtype=float).ravel()
    if obs.size != len(path_table):
        raise ValueError("number of observations must match the number of paths")

    xx, yy = np.meshgrid(grid.x_centers_mm, grid.y_centers_mm)
    numerator = np.zeros_like(xx, dtype=float)
    denominator = np.zeros_like(xx, dtype=float)

    for value, row in zip(obs, path_table.itertuples(index=False)):
        tx, ty = float(row.Tx_X), float(row.Tx_Y)
        rx, ry = float(row.Rx_X), float(row.Rx_Y)
        direct = np.hypot(rx - tx, ry - ty)
        if direct <= 0.0:
            continue
        d1 = np.hypot(xx - tx, yy - ty)
        d2 = np.hypot(xx - rx, yy - ry)
        ratio = (d1 + d2) / direct
        weight = np.clip((beta - ratio) / (beta - 1.0), 0.0, 1.0)
        numerator += float(value) * weight
        denominator += weight

    field = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
    return suppress_background(field, recon_cfg)
