import numpy as np

from cfrp_gw.config import GridConfig, MultiAreaConfig, SegmentationConfig, SingleAreaConfig
from cfrp_gw.geometry import make_grid
from cfrp_gw.quantification import quantify_multiple, quantify_single


def _gaussian_field(grid, centers):
    xx, yy = np.meshgrid(grid.x_centers_mm, grid.y_centers_mm)
    out = np.zeros_like(xx)
    for x0, y0, amp in centers:
        out += amp * np.exp(-((xx - x0) ** 2 / (2 * 25.0**2) + (yy - y0) ** 2 / (2 * 12.0**2)))
    return out


def test_single_quantification_returns_finite_estimate():
    grid = make_grid(GridConfig())
    field = _gaussian_field(grid, [(450.0, 50.0, 4.0)])
    estimate = quantify_single(field, grid, SegmentationConfig(), SingleAreaConfig())
    assert np.isfinite(estimate.centroid_x_mm)
    assert np.isfinite(estimate.centroid_y_mm)
    assert estimate.equivalent_area_mm2 > 0.0


def test_multiple_quantification_finds_separated_responses():
    grid = make_grid(GridConfig())
    field = _gaussian_field(grid, [(150.0, 50.0, 3.0), (350.0, 50.0, 3.5), (550.0, 50.0, 4.0), (750.0, 50.0, 4.5)])
    estimates = quantify_multiple(field, grid, SegmentationConfig(), MultiAreaConfig())
    assert len(estimates) >= 3
