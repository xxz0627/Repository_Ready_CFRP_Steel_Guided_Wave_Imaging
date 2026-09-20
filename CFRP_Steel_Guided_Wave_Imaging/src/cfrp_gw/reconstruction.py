from __future__ import annotations

import numpy as np
from scipy import sparse
from skimage.restoration import denoise_tv_chambolle

from .config import ReconstructionConfig


def _normalization_vectors(A: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    col_sum = np.asarray(A.sum(axis=0)).ravel()
    row_norm = np.zeros_like(row_sum, dtype=float)
    col_norm = np.zeros_like(col_sum, dtype=float)
    row_norm[row_sum > 0.0] = 1.0 / row_sum[row_sum > 0.0]
    col_norm[col_sum > 1.0e-12] = 1.0 / col_sum[col_sum > 1.0e-12]
    return row_norm, col_norm


def suppress_background(field: np.ndarray, cfg: ReconstructionConfig) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    background = np.percentile(arr[np.isfinite(arr)], cfg.background_percentile)
    out = arr - cfg.background_strength * background
    return np.clip(out, cfg.attenuation_min, cfg.attenuation_max)


def reconstruct_sirt(
    A: sparse.csr_matrix,
    b: np.ndarray,
    shape: tuple[int, int],
    cfg: ReconstructionConfig,
    use_tv: bool,
) -> np.ndarray:
    """SIRT with optional Chambolle-TV denoising after each update."""
    obs = np.asarray(b, dtype=float).ravel()
    if A.shape[0] != obs.size:
        raise ValueError("number of observations must match the ray-matrix rows")
    if A.shape[1] != shape[0] * shape[1]:
        raise ValueError("shape is inconsistent with the ray-matrix columns")

    row_norm, col_norm = _normalization_vectors(A)
    x = np.zeros(A.shape[1], dtype=float)
    span = cfg.attenuation_max - cfg.attenuation_min

    for iteration in range(cfg.iterations):
        projected = A @ x
        residual = obs - projected
        correction = A.T @ (row_norm * residual)
        update = col_norm * np.asarray(correction).ravel()
        relaxation = cfg.relaxation_initial * (1.0 - iteration / cfg.iterations)
        x = np.clip(x + relaxation * update, cfg.attenuation_min, cfg.attenuation_max)

        if use_tv and cfg.tv_weight > 0.0:
            image = x.reshape(shape)
            normalized = (image - cfg.attenuation_min) / max(span, 1.0e-12)
            normalized = denoise_tv_chambolle(
                normalized,
                weight=cfg.tv_weight,
                max_num_iter=cfg.tv_inner_iterations,
            )
            x = np.clip(
                normalized.ravel() * span + cfg.attenuation_min,
                cfg.attenuation_min,
                cfg.attenuation_max,
            )

    return suppress_background(x.reshape(shape), cfg)


def sirt(A: sparse.csr_matrix, b: np.ndarray, shape: tuple[int, int], cfg: ReconstructionConfig) -> np.ndarray:
    return reconstruct_sirt(A, b, shape, cfg, use_tv=False)


def tv_sirt(A: sparse.csr_matrix, b: np.ndarray, shape: tuple[int, int], cfg: ReconstructionConfig) -> np.ndarray:
    return reconstruct_sirt(A, b, shape, cfg, use_tv=True)
