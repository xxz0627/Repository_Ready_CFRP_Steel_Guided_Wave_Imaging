from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import GridConfig


@dataclass(frozen=True)
class Grid:
    x_edges_mm: np.ndarray
    y_edges_mm: np.ndarray
    x_centers_mm: np.ndarray
    y_centers_mm: np.ndarray

    @property
    def nx(self) -> int:
        return self.x_centers_mm.size

    @property
    def ny(self) -> int:
        return self.y_centers_mm.size

    @property
    def cell_area_mm2(self) -> float:
        dx = float(np.mean(np.diff(self.x_edges_mm)))
        dy = float(np.mean(np.diff(self.y_edges_mm)))
        return dx * dy


def make_grid(cfg: GridConfig) -> Grid:
    x_edges = np.linspace(cfg.x_range_mm[0], cfg.x_range_mm[1], cfg.nx + 1)
    y_edges = np.linspace(cfg.y_range_mm[0], cfg.y_range_mm[1], cfg.ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return Grid(x_edges, y_edges, x_centers, y_centers)


def _midpoint_cell_indices(xm: np.ndarray, ym: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    ix = np.searchsorted(grid.x_edges_mm, xm, side="right") - 1
    iy = np.searchsorted(grid.y_edges_mm, ym, side="right") - 1
    return ix, iy


def build_ray_matrix(path_table, cfg: GridConfig, grid: Grid | None = None) -> tuple[sparse.csr_matrix, Grid]:
    """Assemble a straight-ray system matrix using the manuscript's 200-sample path discretization."""
    grid = make_grid(cfg) if grid is None else grid
    required = ("Tx_X", "Tx_Y", "Rx_X", "Rx_Y")
    missing = [c for c in required if c not in path_table.columns]
    if missing:
        raise ValueError(f"missing coordinate columns: {missing}")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    t_edges = np.linspace(0.0, 1.0, cfg.ray_samples + 1)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])

    for row_index, row in enumerate(path_table.itertuples(index=False)):
        tx = float(getattr(row, "Tx_X"))
        ty = float(getattr(row, "Tx_Y"))
        rx = float(getattr(row, "Rx_X"))
        ry = float(getattr(row, "Rx_Y"))

        dx = rx - tx
        dy = ry - ty
        path_length_m = np.hypot(dx, dy) * 1.0e-3
        ds_m = path_length_m / cfg.ray_samples
        xm = tx + t_mid * dx
        ym = ty + t_mid * dy
        ix, iy = _midpoint_cell_indices(xm, ym, grid)

        valid = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        flat = iy[valid] * grid.nx + ix[valid]
        if flat.size == 0:
            continue
        unique, count = np.unique(flat, return_counts=True)
        rows.extend([row_index] * unique.size)
        cols.extend(unique.tolist())
        vals.extend((count.astype(float) * ds_m).tolist())

    matrix = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(len(path_table), grid.nx * grid.ny),
        dtype=float,
    ).tocsr()
    return matrix, grid
