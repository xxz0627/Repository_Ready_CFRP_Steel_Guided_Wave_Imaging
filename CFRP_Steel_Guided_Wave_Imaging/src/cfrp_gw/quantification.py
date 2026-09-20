from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from skimage import measure, morphology
from skimage.feature import peak_local_max

from .config import MultiAreaConfig, SegmentationConfig, SingleAreaConfig
from .geometry import Grid


@dataclass
class DefectEstimate:
    centroid_x_mm: float
    centroid_y_mm: float
    equivalent_area_mm2: float
    binary_area_mm2: float
    weighted_area_mm2: float
    peak_value: float
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def _crop_to_quantification_region(field: np.ndarray, grid: Grid, cfg: SegmentationConfig) -> tuple[np.ndarray, Grid]:
    x_mask = (grid.x_centers_mm >= cfg.quantify_x_range_mm[0]) & (grid.x_centers_mm <= cfg.quantify_x_range_mm[1])
    y_mask = (grid.y_centers_mm >= cfg.quantify_y_range_mm[0]) & (grid.y_centers_mm <= cfg.quantify_y_range_mm[1])
    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError("quantification range does not overlap the reconstruction grid")
    ix = np.where(x_mask)[0]
    iy = np.where(y_mask)[0]
    cropped = np.asarray(field, dtype=float)[np.ix_(iy, ix)]
    subgrid = Grid(
        x_edges_mm=grid.x_edges_mm[ix[0] : ix[-1] + 2],
        y_edges_mm=grid.y_edges_mm[iy[0] : iy[-1] + 2],
        x_centers_mm=grid.x_centers_mm[ix],
        y_centers_mm=grid.y_centers_mm[iy],
    )
    return cropped, subgrid


def robust_background_threshold(field: np.ndarray, cfg: SegmentationConfig) -> float:
    arr = np.asarray(field, dtype=float)
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return 0.0
    cutoff = np.percentile(values, cfg.background_percentile)
    background = values[values < cutoff]
    if background.size == 0:
        background = values
    median = float(np.median(background))
    mad = float(np.median(np.abs(background - median)))
    sigma = 1.4826 * mad if mad > 0.0 else float(np.std(background))
    return median + cfg.background_mad_k * sigma


def segmentation_thresholds(field: np.ndarray, cfg: SegmentationConfig) -> tuple[float, float]:
    arr = np.asarray(field, dtype=float)
    peak = float(np.max(arr))
    t_img_mad = robust_background_threshold(arr, cfg)
    low = max(t_img_mad, cfg.low_peak_ratio * peak)
    high = max(float(np.percentile(arr, cfg.high_percentile)), cfg.high_peak_ratio * peak, low)
    return float(low), float(high)


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    if not mask[seed]:
        return np.zeros_like(mask, dtype=bool)
    labels = measure.label(mask, connectivity=2)
    label = labels[seed]
    return labels == label if label else np.zeros_like(mask, dtype=bool)


def hysteresis_peak_component(field: np.ndarray, cfg: SegmentationConfig) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(field, dtype=float)
    low, high = segmentation_thresholds(arr, cfg)
    peak_rc = np.unravel_index(np.argmax(arr), arr.shape)
    high_mask = arr >= high
    low_mask = arr >= low
    seed = np.zeros_like(high_mask, dtype=np.uint8)
    seed[peak_rc] = 1
    seed &= high_mask.astype(np.uint8)
    if not np.any(seed):
        return np.zeros_like(arr, dtype=bool), low, high
    high_connected = morphology.reconstruction(seed, high_mask.astype(np.uint8), method="dilation")
    grown = morphology.reconstruction(high_connected.astype(np.uint8), low_mask.astype(np.uint8), method="dilation")
    return grown.astype(bool), low, high


def _weighted_metrics(field: np.ndarray, mask: np.ndarray, grid: Grid, low_threshold: float, peak: float) -> tuple[float, float, float, float, float]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("empty defect mask")
    vals = field[rows, cols]
    dx = float(np.mean(np.diff(grid.x_edges_mm)))
    dy = float(np.mean(np.diff(grid.y_edges_mm)))
    cell_area = dx * dy
    weights_area = np.clip((vals - low_threshold) / max(peak - low_threshold, 1.0e-12), 0.0, 1.0)
    weighted_area = float(np.sum(weights_area) * cell_area)
    binary_area = float(rows.size * cell_area)
    weights_centroid = np.maximum(vals, 1.0e-12)
    cx = float(np.average(grid.x_centers_mm[cols], weights=weights_centroid))
    cy = float(np.average(grid.y_centers_mm[rows], weights=weights_centroid))
    return weighted_area, binary_area, cx, cy, float(np.max(vals))


def _single_phi(peak: float, cfg: SingleAreaConfig) -> float:
    value = cfg.phi_c0 + cfg.phi_c1 * peak + cfg.phi_c2 * peak * peak
    return float(np.clip(value, cfg.phi_min, cfg.phi_max))


def _single_mu(peak: float, cfg: SingleAreaConfig) -> float:
    value = cfg.mu_intercept - cfg.mu_slope * peak
    if peak > cfg.mu_high_peak_threshold:
        value -= cfg.mu_high_peak_extra_slope * (peak - cfg.mu_high_peak_threshold)
    return float(np.clip(value, cfg.mu_min, cfg.mu_max))


def quantify_single(field: np.ndarray, grid: Grid, seg_cfg: SegmentationConfig, area_cfg: SingleAreaConfig) -> DefectEstimate:
    arr, grid = _crop_to_quantification_region(field, grid, seg_cfg)
    mask, low, _ = hysteresis_peak_component(arr, seg_cfg)
    if not np.any(mask):
        raise RuntimeError("no single-defect response was segmented")
    peak = float(np.max(arr))
    weighted_area, binary_area, cx, cy, local_peak = _weighted_metrics(arr, mask, grid, low, peak)
    constrained_area = _single_phi(local_peak, area_cfg) * weighted_area
    mu = _single_mu(local_peak, area_cfg)
    final_area = mu * constrained_area + (1.0 - mu) * weighted_area
    return DefectEstimate(cx, cy, float(final_area), binary_area, weighted_area, local_peak, low)


def _contiguous_span(profile: np.ndarray, center: int, threshold: float, step_mm: float) -> float:
    if profile.size == 0:
        return 0.0
    center = int(np.clip(center, 0, profile.size - 1))
    if profile[center] < threshold:
        center = int(np.argmax(profile))
        if profile[center] < threshold:
            return 0.0
    left = center
    right = center
    while left > 0 and profile[left - 1] >= threshold:
        left -= 1
    while right + 1 < profile.size and profile[right + 1] >= threshold:
        right += 1
    return float((right - left + 1) * step_mm)


def _profile_area(field: np.ndarray, mask: np.ndarray, grid: Grid, ratio_x: float, ratio_y: float) -> tuple[float, float, float]:
    rows, cols = np.where(mask)
    vals = field[rows, cols]
    row_center = int(round(np.average(rows, weights=np.maximum(vals, 1.0e-12))))
    col_center = int(round(np.average(cols, weights=np.maximum(vals, 1.0e-12))))
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    px = np.asarray(field[row_center, c0 : c1 + 1], dtype=float)
    py = np.asarray(field[r0 : r1 + 1, col_center], dtype=float)
    dx = float(np.mean(np.diff(grid.x_edges_mm)))
    dy = float(np.mean(np.diff(grid.y_edges_mm)))
    width = _contiguous_span(px, col_center - c0, float(np.max(px)) * ratio_x, dx)
    height = _contiguous_span(py, row_center - r0, float(np.max(py)) * ratio_y, dy)
    return width * height, width, height


def _base_profile_ratios(local_peak: float, bbox_aspect: float, cfg: MultiAreaConfig) -> tuple[float, float]:
    ratio_x = cfg.strong_profile_ratio if local_peak >= cfg.strong_peak_threshold else cfg.profile_base_ratio
    ratio_y = ratio_x
    if local_peak < cfg.strong_peak_threshold:
        if bbox_aspect <= cfg.tall_aspect_threshold:
            ratio_x = max(ratio_x, cfg.tall_profile_ratio)
            ratio_y = max(ratio_y, cfg.tall_profile_ratio)
        elif bbox_aspect >= cfg.wide_aspect_threshold:
            ratio_y = max(ratio_y, cfg.wide_y_profile_ratio)
    return float(ratio_x), float(ratio_y)


def _soft_parameters(local_peak: float, profile_aspect: float, cfg: MultiAreaConfig) -> tuple[float, float, float]:
    if local_peak < cfg.weak_peak_upper:
        delta = cfg.soft_delta_weak
        kappa = cfg.kappa_weak
    elif local_peak < cfg.intermediate_peak_upper:
        delta = cfg.soft_delta_intermediate
        kappa = cfg.kappa_intermediate
    else:
        delta = cfg.soft_delta_strong
        kappa = cfg.kappa_strong

    delta_x = delta
    delta_y = delta
    if profile_aspect >= cfg.soft_wide_aspect_threshold:
        delta_x += cfg.soft_extra_major
        delta_y += cfg.soft_extra_minor
        kappa += cfg.soft_wide_kappa_extra
    elif profile_aspect <= cfg.soft_tall_aspect_threshold:
        delta_y += cfg.soft_extra_major
        delta_x += cfg.soft_extra_minor
        kappa += cfg.soft_tall_kappa_extra
    return float(delta_x), float(delta_y), float(np.clip(kappa, cfg.kappa_min, cfg.kappa_max))


def _find_multi_peaks(field: np.ndarray, grid: Grid, threshold: float, cfg: MultiAreaConfig) -> list[tuple[int, int]]:
    peak = float(np.max(field))
    dx = float(np.mean(np.diff(grid.x_edges_mm)))
    dy = float(np.mean(np.diff(grid.y_edges_mm)))
    min_distance_px = max(1, int(round(cfg.peak_min_distance_mm / max(min(dx, dy), 1.0e-12))))
    coords = peak_local_max(
        field,
        min_distance=min_distance_px,
        threshold_abs=max(threshold, cfg.peak_min_global_ratio * peak),
        exclude_border=False,
    )
    ordered = sorted(
        [(int(r), int(c)) for r, c in coords],
        key=lambda rc: float(field[rc]),
        reverse=True,
    )
    return ordered[: cfg.max_peaks]


def quantify_multiple(field: np.ndarray, grid: Grid, seg_cfg: SegmentationConfig, area_cfg: MultiAreaConfig) -> list[DefectEstimate]:
    arr, grid = _crop_to_quantification_region(field, grid, seg_cfg)
    background = robust_background_threshold(arr, seg_cfg)
    global_peak = float(np.max(arr))
    peaks = _find_multi_peaks(arr, grid, background, area_cfg)
    cell_area = grid.cell_area_mm2
    estimates: list[DefectEstimate] = []
    accepted_masks: list[np.ndarray] = []

    for pr, pc in peaks:
        local_peak = float(arr[pr, pc])
        local_threshold = max(background, area_cfg.local_component_peak_ratio * local_peak)
        component = _component_from_seed(arr >= local_threshold, (pr, pc))
        if not np.any(component):
            continue

        duplicate = False
        for previous in accepted_masks:
            intersection = np.logical_and(component, previous).sum()
            union = np.logical_or(component, previous).sum()
            if union and intersection / union > 0.5:
                duplicate = True
                break
        if duplicate:
            continue

        binary_area = float(component.sum() * cell_area)
        if not (area_cfg.min_component_area_mm2 <= binary_area <= area_cfg.max_component_area_mm2):
            continue

        weighted_area, _, cx, cy, _ = _weighted_metrics(arr, component, grid, local_threshold, local_peak)
        rows, cols = np.where(component)
        dx = float(np.mean(np.diff(grid.x_edges_mm)))
        dy = float(np.mean(np.diff(grid.y_edges_mm)))
        bbox_w = (cols.max() - cols.min() + 1) * dx
        bbox_h = (rows.max() - rows.min() + 1) * dy
        bbox_aspect = bbox_w / max(bbox_h, 1.0e-12)

        ratio_x, ratio_y = _base_profile_ratios(local_peak, bbox_aspect, area_cfg)
        basic_area, basic_w, basic_h = _profile_area(arr, component, grid, ratio_x, ratio_y)
        profile_aspect = basic_w / max(basic_h, 1.0e-12)
        delta_x, delta_y, kappa = _soft_parameters(local_peak, profile_aspect, area_cfg)
        soft_ratio_x = float(np.clip(ratio_x - delta_x, area_cfg.soft_ratio_min, 0.95))
        soft_ratio_y = float(np.clip(ratio_y - delta_y, area_cfg.soft_ratio_min, 0.95))
        soft_area, _, _ = _profile_area(arr, component, grid, soft_ratio_x, soft_ratio_y)
        final_area = (1.0 - kappa) * basic_area + kappa * soft_area

        estimates.append(
            DefectEstimate(cx, cy, float(final_area), binary_area, weighted_area, local_peak, local_threshold)
        )
        accepted_masks.append(component)

    estimates.sort(key=lambda item: item.centroid_x_mm)
    return estimates


def evaluate_estimates(estimates: Iterable[DefectEstimate], truth: Iterable[dict]) -> list[dict]:
    """Match detections to supplied reference geometry only for post-detection evaluation."""
    estimates = list(estimates)
    references = list(truth)
    used: set[int] = set()
    rows: list[dict] = []

    for estimate in estimates:
        best = None
        best_distance = np.inf
        for index, ref in enumerate(references):
            if index in used:
                continue
            distance = np.hypot(estimate.centroid_x_mm - ref["x"], estimate.centroid_y_mm - ref["y"])
            if distance < best_distance:
                best_distance = distance
                best = index
        if best is None:
            continue
        used.add(best)
        ref = references[best]
        area_error = abs(estimate.equivalent_area_mm2 - ref["area"]) / ref["area"] * 100.0
        rows.append(
            {
                "reference_index": best + 1,
                "centroid_offset_mm": float(best_distance),
                "reference_area_mm2": float(ref["area"]),
                "estimated_area_mm2": float(estimate.equivalent_area_mm2),
                "equivalent_area_error_percent": float(area_error),
            }
        )
    return rows
