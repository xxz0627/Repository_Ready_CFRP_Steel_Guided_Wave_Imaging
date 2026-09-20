from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SSBConfig

COORD_COLUMNS = ("Tx_X", "Tx_Y", "Rx_X", "Rx_Y")


def _energy_column(df: pd.DataFrame) -> str:
    if "Energy" in df.columns:
        return "Energy"
    if "Energy_mean" in df.columns:
        return "Energy_mean"
    repeat_cols = [f"Energy_{i}" for i in range(1, 6)]
    if all(c in df.columns for c in repeat_cols):
        return "__repeat_median__"
    raise ValueError(
        "energy table must contain Energy, Energy_mean, or Energy_1 ... Energy_5"
    )


def prepare_path_energies(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing coordinate columns: {missing}")

    out = df.copy()
    source = _energy_column(out)
    if source == "__repeat_median__":
        repeat_cols = [f"Energy_{i}" for i in range(1, 6)]
        out["Energy"] = out[repeat_cols].median(axis=1)
    else:
        out["Energy"] = out[source].astype(float)

    if not np.all(np.isfinite(out["Energy"].to_numpy(float))):
        raise ValueError("energy column contains non-finite values")
    if np.any(out["Energy"].to_numpy(float) < 0.0):
        raise ValueError("path energy must be non-negative")
    return out


def assign_nominal_length_groups(df: pd.DataFrame, cfg: SSBConfig) -> pd.DataFrame:
    out = df.copy()
    dx = out["Rx_X"].to_numpy(float) - out["Tx_X"].to_numpy(float)
    dy = out["Rx_Y"].to_numpy(float) - out["Tx_Y"].to_numpy(float)
    lengths = np.hypot(dx, dy)
    nominal = np.asarray(cfg.nominal_lengths_mm, dtype=float)
    group_index = np.argmin(np.abs(lengths[:, None] - nominal[None, :]), axis=1)
    out["Path_Length_mm"] = lengths
    out["SSB_Length_Group_mm"] = nominal[group_index]
    return out


def construct_ssb_observations(df: pd.DataFrame, cfg: SSBConfig) -> pd.DataFrame:
    """Construct Eq. (15) using a current-scan percentile reference by path length."""
    out = assign_nominal_length_groups(prepare_path_energies(df), cfg)
    energy = np.maximum(out["Energy"].to_numpy(float), cfg.energy_floor)

    refs = np.zeros(len(out), dtype=float)
    raw = np.zeros(len(out), dtype=float)
    group_values = out["SSB_Length_Group_mm"].to_numpy(float)

    for group in np.unique(group_values):
        mask = np.isclose(group_values, group)
        reference = float(np.quantile(energy[mask], cfg.percentile))
        refs[mask] = reference
        raw[mask] = np.log((reference + cfg.energy_floor) / (energy[mask] + cfg.energy_floor))

    out["SSB_Reference_Energy"] = refs
    out["Raw_Attenuation"] = raw
    return out


def mad_screen(observations: np.ndarray, cfg: SSBConfig) -> tuple[np.ndarray, float]:
    """Apply Eqs. (16)-(17) to path-level attenuation observations."""
    b = np.asarray(observations, dtype=float)
    if b.ndim != 1 or b.size == 0:
        raise ValueError("observations must be a non-empty one-dimensional array")
    median = float(np.median(b))
    mad = float(np.median(np.abs(b - median)))
    threshold = median + cfg.mad_k * 1.4826 * mad
    threshold = float(np.clip(threshold, 0.0, cfg.mad_threshold_cap))
    screened = np.where(b >= threshold, b, 0.0)
    return screened, threshold


def build_screened_observations(df: pd.DataFrame, cfg: SSBConfig) -> pd.DataFrame:
    out = construct_ssb_observations(df, cfg)
    screened, threshold = mad_screen(out["Raw_Attenuation"].to_numpy(float), cfg)
    out["Attenuation"] = screened
    out["Path_MAD_Threshold"] = threshold
    return out
