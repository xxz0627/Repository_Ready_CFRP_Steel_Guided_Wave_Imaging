from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from .config import ReconstructionConfig
from .geometry import Grid


def interpolate_for_display(field: np.ndarray, grid: Grid, cfg: ReconstructionConfig):
    scale = max(int(cfg.display_interpolation_scale), 1)
    x_new = np.linspace(grid.x_centers_mm[0], grid.x_centers_mm[-1], grid.nx * scale)
    y_new = np.linspace(grid.y_centers_mm[0], grid.y_centers_mm[-1], grid.ny * scale)

    if cfg.display_interpolation_method.lower() == "spline":
        interpolator = RectBivariateSpline(grid.y_centers_mm, grid.x_centers_mm, field, kx=3, ky=3)
        image = interpolator(y_new, x_new)
    else:
        interpolator = RegularGridInterpolator(
            (grid.y_centers_mm, grid.x_centers_mm),
            field,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        yy, xx = np.meshgrid(y_new, x_new, indexing="ij")
        image = interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(yy.shape)
    return image, x_new, y_new


def normalized_display(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def save_attenuation_map(field: np.ndarray, grid: Grid, cfg: ReconstructionConfig, output: str | Path, title: str = "Attenuation map") -> None:
    image, x, y = interpolate_for_display(field, grid, cfg)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    mesh = ax.imshow(
        image,
        origin="lower",
        extent=(x.min(), x.max(), y.min(), y.max()),
        aspect="auto",
    )
    ax.set_xlabel("X coordinate (mm)")
    ax.set_ylabel("Y coordinate (mm)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Equivalent attenuation index")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
