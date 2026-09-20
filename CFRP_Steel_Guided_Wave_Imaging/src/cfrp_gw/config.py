from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PreprocessingConfig:
    sampling_rate_hz: float = 2_000_000.0
    low_cut_hz: float = 80_000.0
    high_cut_hz: float = 120_000.0
    butterworth_order: int = 4
    record_duration_s: float = 1.0e-3
    repeats_per_path: int = 5


@dataclass(frozen=True)
class SSBConfig:
    percentile: float = 0.70
    nominal_lengths_mm: Tuple[float, float, float] = (
        50.0,
        55.90169943749474,
        70.71067811865476,
    )
    energy_floor: float = 1.0e-10
    mad_k: float = 1.0
    mad_threshold_cap: float = 0.04


@dataclass(frozen=True)
class GridConfig:
    x_range_mm: Tuple[float, float] = (-10.0, 910.0)
    y_range_mm: Tuple[float, float] = (-10.0, 110.0)
    nx: int = 160
    ny: int = 24
    ray_samples: int = 200


@dataclass(frozen=True)
class ReconstructionConfig:
    iterations: int = 100
    relaxation_initial: float = 0.30
    attenuation_min: float = 0.0
    attenuation_max: float = 10.0
    tv_weight: float = 0.003
    tv_inner_iterations: int = 20
    background_percentile: float = 60.0
    background_strength: float = 1.0
    display_interpolation_scale: int = 7
    display_interpolation_method: str = "spline"


@dataclass(frozen=True)
class SegmentationConfig:
    quantify_x_range_mm: Tuple[float, float] = (0.0, 900.0)
    quantify_y_range_mm: Tuple[float, float] = (0.0, 100.0)
    background_percentile: float = 70.0
    background_mad_k: float = 3.5
    high_percentile: float = 95.0
    high_peak_ratio: float = 0.45
    low_peak_ratio: float = 0.18


@dataclass(frozen=True)
class SingleAreaConfig:
    phi_c0: float = 0.26
    phi_c1: float = 0.05
    phi_c2: float = 0.018
    phi_min: float = 0.35
    phi_max: float = 0.85
    mu_intercept: float = 1.026
    mu_slope: float = 0.073
    mu_high_peak_threshold: float = 3.0
    mu_high_peak_extra_slope: float = 0.15
    mu_min: float = 0.45
    mu_max: float = 0.95


@dataclass(frozen=True)
class MultiAreaConfig:
    peak_min_distance_mm: float = 70.0
    peak_min_global_ratio: float = 0.10
    max_peaks: int = 8
    local_component_peak_ratio: float = 0.25
    min_component_area_mm2: float = 100.0
    max_component_area_mm2: float = 9000.0

    profile_base_ratio: float = 0.70
    strong_peak_threshold: float = 4.8
    strong_profile_ratio: float = 0.60
    tall_aspect_threshold: float = 0.85
    tall_profile_ratio: float = 0.75
    wide_aspect_threshold: float = 1.18
    wide_y_profile_ratio: float = 0.75

    weak_peak_upper: float = 2.6
    intermediate_peak_upper: float = 4.2
    soft_delta_weak: float = 0.08
    soft_delta_intermediate: float = 0.06
    soft_delta_strong: float = 0.04
    kappa_weak: float = 0.34
    kappa_intermediate: float = 0.34
    kappa_strong: float = 0.22

    soft_wide_aspect_threshold: float = 1.15
    soft_tall_aspect_threshold: float = 0.88
    soft_extra_major: float = 0.03
    soft_extra_minor: float = 0.01
    soft_wide_kappa_extra: float = 0.06
    soft_tall_kappa_extra: float = -0.16
    soft_ratio_min: float = 0.52
    kappa_min: float = 0.10
    kappa_max: float = 0.65


@dataclass(frozen=True)
class PDIConfig:
    shape_factor_beta: float = 1.04


@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ssb: SSBConfig = field(default_factory=SSBConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    single_area: SingleAreaConfig = field(default_factory=SingleAreaConfig)
    multi_area: MultiAreaConfig = field(default_factory=MultiAreaConfig)
    pdi: PDIConfig = field(default_factory=PDIConfig)


DEFAULT_CONFIG = PipelineConfig()
