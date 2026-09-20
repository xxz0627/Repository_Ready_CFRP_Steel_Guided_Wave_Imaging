from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .geometry import Grid, build_ray_matrix
from .pdi import probabilistic_diagnostic_imaging
from .quantification import DefectEstimate, quantify_multiple, quantify_single
from .reconstruction import sirt, tv_sirt
from .ssb import build_screened_observations


@dataclass
class ImagingResult:
    paths: pd.DataFrame
    grid: Grid
    field: np.ndarray
    estimates: list[DefectEstimate]
    method: str


def run_imaging(
    path_energy_table: pd.DataFrame,
    method: str = "TV-SIRT",
    defect_mode: str = "single",
    config: PipelineConfig = DEFAULT_CONFIG,
) -> ImagingResult:
    paths = build_screened_observations(path_energy_table, config.ssb)
    b = paths["Attenuation"].to_numpy(float)
    A, grid = build_ray_matrix(paths, config.grid)
    shape = (config.grid.ny, config.grid.nx)

    method_key = method.upper().replace("_", "-")
    if method_key == "TV-SIRT":
        field = tv_sirt(A, b, shape, config.reconstruction)
    elif method_key == "SIRT":
        field = sirt(A, b, shape, config.reconstruction)
    elif method_key == "PDI":
        field = probabilistic_diagnostic_imaging(paths, b, grid, config.pdi, config.reconstruction)
    else:
        raise ValueError("method must be one of: TV-SIRT, SIRT, PDI")

    if defect_mode == "single":
        estimates = [quantify_single(field, grid, config.segmentation, config.single_area)]
    elif defect_mode == "multiple":
        estimates = quantify_multiple(field, grid, config.segmentation, config.multi_area)
    elif defect_mode == "none":
        estimates = []
    else:
        raise ValueError("defect_mode must be one of: single, multiple, none")

    return ImagingResult(paths, grid, field, estimates, method_key)
