"""Same-scan statistical-background (SSB) path-observation construction.

This module mirrors the manuscript definition: paths are grouped by nominal
propagation length and the q-th percentile (q=0.70 by default) of the current
scan's energy population is used as the internal reference for each group.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NOMINAL_LENGTHS_MM = np.array([50.0, np.hypot(50.0, 25.0), np.hypot(50.0, 50.0)], dtype=float)

def construct_ssb_observations(df: pd.DataFrame, q: float = 0.70, energy_floor: float = 1e-10) -> pd.DataFrame:
    required = {'Tx_X','Tx_Y','Rx_X','Rx_Y'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing coordinate columns: {sorted(missing)}')
    energy_col = 'Energy_mean' if 'Energy_mean' in df.columns else 'Energy'
    if energy_col not in df.columns:
        raise ValueError('Expected Energy or Energy_mean column.')

    out = df.copy()
    dx = out['Rx_X'].to_numpy(float) - out['Tx_X'].to_numpy(float)
    dy = out['Rx_Y'].to_numpy(float) - out['Tx_Y'].to_numpy(float)
    length = np.hypot(dx, dy)
    group_idx = np.argmin(np.abs(length[:, None] - NOMINAL_LENGTHS_MM[None, :]), axis=1)
    out['Path_Length_mm'] = length
    out['SSB_Length_Group_mm'] = NOMINAL_LENGTHS_MM[group_idx]

    energy = np.maximum(out[energy_col].to_numpy(float), energy_floor)
    obs = np.zeros(len(out), dtype=float)
    ref = np.zeros(len(out), dtype=float)
    for g in np.unique(group_idx):
        mask = group_idx == g
        e0 = float(np.quantile(energy[mask], q))
        ref[mask] = e0
        obs[mask] = np.maximum(np.log((e0 + energy_floor) / (energy[mask] + energy_floor)), 0.0)
    out['SSB_Reference_Energy'] = ref
    out['Raw_Attenuation_Integral'] = obs
    return out

def mad_screen(values, k: float = 1.0, max_threshold: float = 0.04):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    threshold = min(med + k * 1.4826 * mad, max_threshold)
    screened = values.copy()
    screened[screened < threshold] = 0.0
    return screened, threshold
