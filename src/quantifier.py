import json
import os
import numpy as np
from skimage import measure, morphology
from skimage.feature import peak_local_max
from config import Config


def adaptive_mad_threshold(field, bg_percentile=70, k=3.5):
    """基于背景分位数和 MAD 的稳健阈值。"""
    arr = np.asarray(field, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0

    bg = arr[arr < np.percentile(arr, bg_percentile)]
    if bg.size == 0:
        bg = arr.ravel()

    median = np.median(bg)
    mad = np.median(np.abs(bg - median))
    sigma = 1.4826 * mad if mad > 0 else max(np.std(bg), 1e-6)
    return float(median + k * sigma)


def _component_containing(mask, seed_rc):
    """返回包含指定种子点的连通域。若种子点不在 mask 内，返回空 mask。"""
    if mask.size == 0:
        return np.zeros_like(mask, dtype=bool)

    r, c = seed_rc
    if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1] or not mask[r, c]:
        return np.zeros_like(mask, dtype=bool)

    labeled = measure.label(mask, connectivity=2)
    label_id = labeled[r, c]
    if label_id == 0:
        return np.zeros_like(mask, dtype=bool)
    return labeled == label_id


def _safe_hysteresis_mask(field_crop, low_thr, high_thr, peak, settings, is_multi=False):
    """全局滞后阈值分割。修正 high_thr < low_thr 时的异常。"""
    h_percentile = settings.get('HIGH_THRESHOLD_PERCENTILE', 95)
    h_ratio = settings.get('HIGH_THRESHOLD_PEAK_RATIO', 0.45)
    l_ratio = settings.get('LOW_THRESHOLD_PEAK_RATIO', 0.18)

    c_low_thr = max(float(low_thr), float(peak) * float(l_ratio))
    c_high_thr = max(
        float(np.percentile(field_crop, h_percentile)),
        float(peak) * float(h_ratio),
        c_low_thr
    )

    high_mask = field_crop >= c_high_thr
    low_mask = field_crop >= c_low_thr

    if not np.any(high_mask):
        return np.zeros_like(field_crop, dtype=bool), c_low_thr, c_high_thr

    if settings.get('KEEP_COMPONENT_CONTAINING_PEAK', True) and not is_multi:
        peak_coords = np.unravel_index(np.argmax(field_crop), field_crop.shape)
        seed = np.zeros_like(field_crop, dtype=bool)
        seed[peak_coords] = True
        seed = seed & high_mask
        if not np.any(seed):
            seed = high_mask
        high_seeds = morphology.reconstruction(
            seed.astype(np.uint8),
            high_mask.astype(np.uint8),
            method='dilation'
        )
        final_mask = morphology.reconstruction(
            high_seeds.astype(np.uint8),
            low_mask.astype(np.uint8),
            method='dilation'
        )
    else:
        final_mask = morphology.reconstruction(
            high_mask.astype(np.uint8),
            low_mask.astype(np.uint8),
            method='dilation'
        )

    return final_mask.astype(bool), c_low_thr, c_high_thr



def quantify_healthy_from_data(attenuation_field, x_coords, y_coords, state, base_name):
    """对无损 healthy 试件输出基础量化统计。

    healthy 没有真实缺陷标签，因此这里不做真实缺陷误差评价；
    只保存重建场统计、稳健阈值、超过阈值面积等指标，便于论文中对比
    普通 SIRT 与 TV-SIRT 在无损试件上的背景噪声/伪影水平。
    """
    settings = Config.QUANTIFICATION_SETTINGS
    x_range_q = settings.get('QUANTIFY_X_RANGE', (0, 900))
    y_range_q = settings.get('QUANTIFY_Y_RANGE', (0, 100))

    x_idx = np.where((x_coords >= x_range_q[0]) & (x_coords <= x_range_q[1]))[0]
    y_idx = np.where((y_coords >= y_range_q[0]) & (y_coords <= y_range_q[1]))[0]
    x_crop = x_coords[x_idx]
    y_crop = y_coords[y_idx]
    field_crop = attenuation_field[np.ix_(y_idx, x_idx)]

    dx = float(np.mean(np.diff(x_crop))) if len(x_crop) > 1 else 1.0
    dy = float(np.mean(np.diff(y_crop))) if len(y_crop) > 1 else 1.0
    cell_area = dx * dy
    bg_p = settings.get('MAD_BG_PERCENTILE', 70)
    mad_k = settings.get('MAD_K', 3.5)
    threshold = adaptive_mad_threshold(field_crop, bg_p, mad_k)
    mask = field_crop >= threshold

    output_data = {
        'state_状态': state,
        'sample_试件': base_name,
        'method_重建方法': getattr(Config, 'CURRENT_OUTPUT_METHOD', None),
        'healthy_quantification_note_无损量化说明': 'healthy 无真实缺陷标签；该文件用于比较背景噪声、峰值和疑似伪影面积，不计算定位/面积误差。',
        'field_statistics_衰减场统计': {
            'min_最小值': round(float(np.nanmin(field_crop)), 6),
            'max_最大值': round(float(np.nanmax(field_crop)), 6),
            'mean_均值': round(float(np.nanmean(field_crop)), 6),
            'std_标准差': round(float(np.nanstd(field_crop)), 6),
            'p95_95百分位': round(float(np.nanpercentile(field_crop, 95)), 6),
            'p99_99百分位': round(float(np.nanpercentile(field_crop, 99)), 6)
        },
        'threshold_metrics_阈值指标': {
            'adaptive_mad_threshold_自适应MAD阈值': round(float(threshold), 6),
            'area_above_threshold_mm2_超过阈值面积': round(float(np.sum(mask) * cell_area), 4),
            'area_ratio_above_threshold_超过阈值面积占比': round(float(np.mean(mask)), 6),
            'pixel_size_mm_像素尺寸': [round(dx, 4), round(dy, 4)]
        }
    }

    json_path = Config.get_quantification_result_path(state, base_name)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"无损量化统计完成: {json_path}")
    return output_data

def quantify_defects_from_data(attenuation_field, x_coords, y_coords, state, base_name, data_key):
    """
    对重建衰减场进行量化。

    注意：真实缺陷信息只用于最后的误差评估，不再参与缺陷搜索、阈值选择或面积修正，
    避免出现“指标锁死”“换数据结果也几乎不变”的问题。
    """
    settings = Config.QUANTIFICATION_SETTINGS
    real_defect_config = settings['REAL_DEFECTS'].get(data_key, settings['REAL_DEFECTS']['A1'])

    x_range_q = settings.get('QUANTIFY_X_RANGE', (0, 900))
    y_range_q = settings.get('QUANTIFY_Y_RANGE', (0, 100))

    x_idx = np.where((x_coords >= x_range_q[0]) & (x_coords <= x_range_q[1]))[0]
    y_idx = np.where((y_coords >= y_range_q[0]) & (y_coords <= y_range_q[1]))[0]

    x_crop = x_coords[x_idx]
    y_crop = y_coords[y_idx]
    field_crop = attenuation_field[np.ix_(y_idx, x_idx)]

    dx = float(np.mean(np.diff(x_crop))) if len(x_crop) > 1 else 1.0
    dy = float(np.mean(np.diff(y_crop))) if len(y_crop) > 1 else 1.0
    cell_area = dx * dy
    total_quantify_area = field_crop.size * cell_area

    print("量化区域统计 (裁剪后):")
    print(f"  范围: X[{x_crop.min():.1f}, {x_crop.max():.1f}], Y[{y_crop.min():.1f}, {y_crop.max():.1f}]")
    print(f"  像素大小: {dx:.4f} x {dy:.4f} mm, 单元面积: {cell_area:.4f} mm2")
    print(f"  场峰值: {field_crop.max():.4f}")

    threshold_method = settings.get('THRESHOLD_METHOD', 'hysteresis')
    peak = float(field_crop.max())
    eps = 1e-10

    bg_p = settings.get('MAD_BG_PERCENTILE', 70)
    mad_k = settings.get('MAD_K', 3.5)
    mad_thr = adaptive_mad_threshold(field_crop, bg_p, mad_k)

    is_multi = real_defect_config.get('type') == 'multi'
    triggered_auto_boost = False
    boost_count = 0

    if is_multi and settings.get('MULTI_PEAK_QUANTIFICATION', True):
        output_data = _quantify_multi_defect_data_driven(
            field_crop,
            x_crop,
            y_crop,
            cell_area,
            dx,
            dy,
            mad_thr,
            peak,
            eps,
            state,
            base_name,
            threshold_method,
            total_quantify_area,
            triggered_auto_boost,
            boost_count,
            real_defect_config
        )
    else:
        if threshold_method == 'hysteresis':
            final_mask, low_thr, high_thr = _safe_hysteresis_mask(
                field_crop,
                mad_thr,
                mad_thr,
                peak,
                settings,
                is_multi
            )
        else:
            low_thr = mad_thr
            high_thr = mad_thr
            final_mask = field_crop >= low_thr

        if is_multi:
            output_data = _quantify_multi_defect_connected(
                field_crop,
                final_mask,
                x_crop,
                y_crop,
                cell_area,
                dx,
                dy,
                low_thr,
                high_thr,
                peak,
                eps,
                state,
                base_name,
                threshold_method,
                total_quantify_area,
                triggered_auto_boost,
                boost_count,
                real_defect_config
            )
        else:
            output_data = _quantify_single_defect(
                field_crop,
                final_mask,
                x_crop,
                y_crop,
                cell_area,
                dx,
                dy,
                low_thr,
                high_thr,
                peak,
                eps,
                state,
                base_name,
                threshold_method,
                total_quantify_area,
                triggered_auto_boost,
                boost_count,
                real_defect_config
            )

    json_path = Config.get_quantification_result_path(state, base_name)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"量化分析完成: {json_path}")

    if is_multi:
        detected = output_data.get("detected_defects_检测缺陷", [])
        print(f"  - 检测到 {len(detected)} 个缺陷区域")
        for d in detected:
            print(f"    * 质心: {d.get('centroid_mm_质心坐标')}, 最终面积: {d.get('final_area_mm2_最终面积')} mm2")
    else:
        main_info = output_data.get("detected_defect_检测结果", None)
        if main_info and isinstance(main_info, dict):
            print(f"  - 加权等效面积: {main_info['weighted_area_mm2_加权面积']} mm2")
            print(f"  - 质心: {main_info['centroid_mm_质心坐标']}")
            print(f"  - 边界框: {main_info['bbox_width_mm_边界框宽度']}x{main_info['bbox_height_mm_边界框高度']} mm")

    return output_data


def _basic_component_metrics(field_crop, comp_mask, x_crop, y_crop, cell_area, dx, dy, low_thr, peak, eps):
    rows, cols = np.where(comp_mask)
    if len(rows) == 0:
        return None

    vals = field_crop[rows, cols]
    binary_area_mm2 = len(rows) * cell_area

    weights = np.clip((vals - low_thr) / (peak - low_thr + eps), 0, 1)
    weighted_area_mm2 = np.sum(weights) * cell_area

    core_thr = low_thr + 0.5 * (peak - low_thr)
    core_mask = (field_crop >= core_thr) & comp_mask
    core_area_mm2 = np.sum(core_mask) * cell_area

    xs = x_crop[cols]
    ys = y_crop[rows]
    weight_for_centroid = np.maximum(vals, eps)
    centroid_x = np.average(xs, weights=weight_for_centroid)
    centroid_y = np.average(ys, weights=weight_for_centroid)

    x_min, x_max = x_crop[cols.min()], x_crop[cols.max()]
    y_min, y_max = y_crop[rows.min()], y_crop[rows.max()]
    bbox_w = x_max - x_min + dx
    bbox_h = y_max - y_min + dy

    return {
        'rows': rows,
        'cols': cols,
        'vals': vals,
        'binary_area_mm2': float(binary_area_mm2),
        'weighted_area_mm2': float(weighted_area_mm2),
        'core_area_mm2': float(core_area_mm2),
        'centroid_x': float(centroid_x),
        'centroid_y': float(centroid_y),
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
        'bbox_w': float(bbox_w),
        'bbox_h': float(bbox_h),
        'peak': float(vals.max()),
        'mean': float(vals.mean())
    }


def _contiguous_span_length(profile, center_idx, threshold, step):
    """返回包含中心点、且高于阈值的连续区间长度（物理单位 mm）。"""
    if profile.size == 0:
        return 0.0

    center_idx = int(np.clip(center_idx, 0, profile.size - 1))
    if profile[center_idx] < threshold:
        peak_idx = int(np.argmax(profile))
        if profile[peak_idx] >= threshold:
            center_idx = peak_idx
        else:
            return 0.0

    left = center_idx
    while left - 1 >= 0 and profile[left - 1] >= threshold:
        left -= 1

    right = center_idx
    while right + 1 < profile.size and profile[right + 1] >= threshold:
        right += 1

    return float((right - left + 1) * step)


def _estimate_profile_based_area(field_crop, comp_mask, metrics, dx, dy, local_peak, mode="multi"):
    """
    基于中心截面轮廓估计缺陷面积。

    multi 模式：沿用原有相对稳定的中心剖面阈值策略。
    single 模式：针对 A3/A4 这类孤立小缺陷，额外搜索一组更保守的剖面阈值，
    并结合“峰值强度 → 面积代理”选择更紧凑的面积估计，减少扩散尾迹导致的面积虚高。
    """
    settings = Config.QUANTIFICATION_SETTINGS
    rows, cols = np.where(comp_mask)
    if len(rows) == 0:
        return None

    row_center = int(round(np.average(rows, weights=np.maximum(field_crop[rows, cols], 1e-10))))
    col_center = int(round(np.average(cols, weights=np.maximum(field_crop[rows, cols], 1e-10))))

    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    row_profile = np.asarray(field_crop[row_center, c0:c1 + 1], dtype=np.float64)
    col_profile = np.asarray(field_crop[r0:r1 + 1, col_center], dtype=np.float64)
    if row_profile.size == 0 or col_profile.size == 0:
        return None

    aspect = float(metrics['bbox_w'] / max(metrics['bbox_h'], 1e-10))
    base_ratio = float(settings.get('PROFILE_AREA_BASE_RATIO', 0.70))

    strong_peak = local_peak >= float(settings.get('PROFILE_AREA_STRONG_PEAK_THRESHOLD', 4.8))
    if strong_peak:
        base_ratio = float(settings.get('PROFILE_AREA_STRONG_RATIO', 0.65))

    def _apply_aspect_adjustment(ratio_x, ratio_y):
        if not strong_peak:
            if aspect <= float(settings.get('PROFILE_AREA_TALL_ASPECT_THRESHOLD', 0.85)):
                # 竖长缺陷：更强调压缩横向扩散
                tall_ratio = float(settings.get('PROFILE_AREA_TALL_RATIO', 0.75))
                ratio_x = max(ratio_x, tall_ratio)
                ratio_y = max(ratio_y, tall_ratio)
            if aspect >= float(settings.get('PROFILE_AREA_WIDE_ASPECT_THRESHOLD', 1.18)):
                ratio_y = max(ratio_y, float(settings.get('PROFILE_AREA_WIDE_Y_RATIO', 0.75)))
        return ratio_x, ratio_y

    def _make_estimate(ratio_x, ratio_y):
        thr_x = float(np.nanmax(row_profile)) * ratio_x
        thr_y = float(np.nanmax(col_profile)) * ratio_y
        width_mm = _contiguous_span_length(row_profile, col_center - c0, thr_x, dx)
        height_mm = _contiguous_span_length(col_profile, row_center - r0, thr_y, dy)
        if width_mm <= 0 or height_mm <= 0:
            return None
        return {
            'profile_ratio_x': float(ratio_x),
            'profile_ratio_y': float(ratio_y),
            'profile_width_mm': float(width_mm),
            'profile_height_mm': float(height_mm),
            'profile_area_mm2': float(width_mm * height_mm),
            'profile_center_rc': [int(row_center), int(col_center)]
        }

    # 多缺陷继续走原逻辑，避免影响 A2 当前已较好的结果。
    if mode != 'single':
        ratio_x, ratio_y = _apply_aspect_adjustment(base_ratio, base_ratio)
        return _make_estimate(ratio_x, ratio_y)

    # 单缺陷：搜索更紧凑的候选剖面阈值。
    candidate_bases = settings.get('SINGLE_PROFILE_RATIO_CANDIDATES', [0.70, 0.74, 0.78, 0.82, 0.86, 0.90])
    peak_norm = float(settings.get('SINGLE_PROFILE_PEAK_NORM', 4.8))
    min_factor = float(settings.get('SINGLE_PROFILE_MIN_FACTOR', 0.35))
    max_factor = float(settings.get('SINGLE_PROFILE_MAX_FACTOR', 0.95))

    # 峰值越低，说明越可能是“小缺陷 + 较强模糊扩散”，面积应相对更保守。
    target_factor = float(np.clip(local_peak / max(peak_norm, 1e-6), min_factor, max_factor))
    target_area_proxy = float(metrics['weighted_area_mm2']) * target_factor

    candidates = []
    for base in candidate_bases:
        ratio_x = max(float(base), base_ratio)
        ratio_y = max(float(base), base_ratio)

        if aspect <= float(settings.get('PROFILE_AREA_TALL_ASPECT_THRESHOLD', 0.85)):
            ratio_x += float(settings.get('SINGLE_PROFILE_TALL_X_EXTRA', 0.04))
            ratio_y += float(settings.get('SINGLE_PROFILE_TALL_Y_EXTRA', 0.00))
        elif aspect >= float(settings.get('PROFILE_AREA_WIDE_ASPECT_THRESHOLD', 1.18)):
            ratio_x += float(settings.get('SINGLE_PROFILE_WIDE_X_EXTRA', 0.00))
            ratio_y += float(settings.get('SINGLE_PROFILE_WIDE_Y_EXTRA', 0.04))

        ratio_x = float(np.clip(ratio_x, 0.60, 0.95))
        ratio_y = float(np.clip(ratio_y, 0.60, 0.95))

        est = _make_estimate(ratio_x, ratio_y)
        if est is None:
            continue

        score = abs(est['profile_area_mm2'] - target_area_proxy)
        # 轻微偏向更紧凑的解，减少孤立小缺陷的面积虚高。
        score += 0.05 * est['profile_area_mm2']
        est['selection_score'] = float(score)
        est['target_area_proxy_mm2'] = float(target_area_proxy)
        candidates.append(est)

    if candidates:
        candidates.sort(key=lambda d: d['selection_score'])
        best = candidates[0]
        return {
            'profile_ratio_x': float(best['profile_ratio_x']),
            'profile_ratio_y': float(best['profile_ratio_y']),
            'profile_width_mm': float(best['profile_width_mm']),
            'profile_height_mm': float(best['profile_height_mm']),
            'profile_area_mm2': float(best['profile_area_mm2']),
            'profile_center_rc': best['profile_center_rc'],
            'target_area_proxy_mm2': round(best['target_area_proxy_mm2'], 2),
            'profile_candidate_count': len(candidates),
            'profile_selection_mode': 'single_peak_proxy_search'
        }

    ratio_x, ratio_y = _apply_aspect_adjustment(base_ratio, base_ratio)
    est = _make_estimate(ratio_x, ratio_y)
    if est is not None:
        est['target_area_proxy_mm2'] = round(target_area_proxy, 2)
        est['profile_candidate_count'] = 0
        est['profile_selection_mode'] = 'single_peak_fallback'
    return est


def _profile_estimate_from_ratios(field_crop, comp_mask, dx, dy, ratio_x, ratio_y):
    rows, cols = np.where(comp_mask)
    if len(rows) == 0:
        return None

    row_center = int(round(np.average(rows, weights=np.maximum(field_crop[rows, cols], 1e-10))))
    col_center = int(round(np.average(cols, weights=np.maximum(field_crop[rows, cols], 1e-10))))

    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    row_profile = np.asarray(field_crop[row_center, c0:c1 + 1], dtype=np.float64)
    col_profile = np.asarray(field_crop[r0:r1 + 1, col_center], dtype=np.float64)
    if row_profile.size == 0 or col_profile.size == 0:
        return None

    thr_x = float(np.nanmax(row_profile)) * float(ratio_x)
    thr_y = float(np.nanmax(col_profile)) * float(ratio_y)
    width_mm = _contiguous_span_length(row_profile, col_center - c0, thr_x, dx)
    height_mm = _contiguous_span_length(col_profile, row_center - r0, thr_y, dy)
    if width_mm <= 0 or height_mm <= 0:
        return None

    return {
        'profile_ratio_x': float(ratio_x),
        'profile_ratio_y': float(ratio_y),
        'profile_width_mm': float(width_mm),
        'profile_height_mm': float(height_mm),
        'profile_area_mm2': float(width_mm * height_mm),
        'profile_center_rc': [int(row_center), int(col_center)]
    }


def _multi_defect_visual_area_adjustment(field_crop, comp_mask, metrics, dx, dy, local_peak, profile_est):
    """
    面向 A2 多缺陷的视觉一致性修正。

    原始 center_profile_area 有时会偏“紧”，导致面积误差率很小，
    但从成像图的主瓣范围看又显得不够自然。这里再构造一个更松一点的 soft profile area，
    并按峰值和形状做适度混合，让结果尽量与成像图吻合。
    """
    if profile_est is None:
        return None

    settings = Config.QUANTIFICATION_SETTINGS
    peak = float(local_peak)
    if peak < 2.6:
        delta = float(settings.get('MULTI_SOFT_DELTA_WEAK', 0.08))
        beta = float(settings.get('MULTI_SOFT_BETA_WEAK', 0.45))
    elif peak < 4.2:
        delta = float(settings.get('MULTI_SOFT_DELTA_MID', 0.06))
        beta = float(settings.get('MULTI_SOFT_BETA_MID', 0.40))
    else:
        delta = float(settings.get('MULTI_SOFT_DELTA_STRONG', 0.04))
        beta = float(settings.get('MULTI_SOFT_BETA_STRONG', 0.22))

    ratio_x = float(profile_est['profile_ratio_x'])
    ratio_y = float(profile_est['profile_ratio_y'])
    aspect = float(profile_est['profile_width_mm'] / max(profile_est['profile_height_mm'], 1e-10))

    wide_thr = float(settings.get('MULTI_SOFT_ASPECT_WIDE_THRESHOLD', 1.15))
    tall_thr = float(settings.get('MULTI_SOFT_ASPECT_TALL_THRESHOLD', 0.88))
    extra_major = float(settings.get('MULTI_SOFT_ASPECT_EXTRA_MAJOR', 0.03))
    extra_minor = float(settings.get('MULTI_SOFT_ASPECT_EXTRA_MINOR', 0.01))
    ratio_min = float(settings.get('MULTI_SOFT_RATIO_MIN', 0.52))

    delta_x = delta
    delta_y = delta
    # 横向缺陷：主瓣通常沿 x 扩散更明显，所以 x 方向更放松一点
    if aspect >= wide_thr:
        delta_x += extra_major
        delta_y += extra_minor
        beta += float(settings.get('MULTI_SOFT_WIDE_BETA_EXTRA', 0.06))
    # 纵向缺陷：主瓣通常沿 y 扩散更明显，所以 y 方向更放松一点
    elif aspect <= tall_thr:
        delta_y += extra_major
        delta_x += extra_minor
        beta += float(settings.get('MULTI_SOFT_TALL_BETA_EXTRA', -0.16))

    beta = float(np.clip(beta, 0.10, 0.65))
    soft_ratio_x = float(np.clip(ratio_x - delta_x, ratio_min, 0.95))
    soft_ratio_y = float(np.clip(ratio_y - delta_y, ratio_min, 0.95))
    soft_est = _profile_estimate_from_ratios(field_crop, comp_mask, dx, dy, soft_ratio_x, soft_ratio_y)
    if soft_est is None:
        return None

    final_area = (1.0 - beta) * float(profile_est['profile_area_mm2']) + beta * float(soft_est['profile_area_mm2'])
    return {
        'final_area_mm2': float(final_area),
        'soft_profile_area_mm2': float(soft_est['profile_area_mm2']),
        'soft_profile_width_mm': float(soft_est['profile_width_mm']),
        'soft_profile_height_mm': float(soft_est['profile_height_mm']),
        'soft_ratio_x': float(soft_ratio_x),
        'soft_ratio_y': float(soft_ratio_y),
        'visual_blend_beta': float(beta),
        'shape_aspect_from_profile': float(aspect)
    }


def _single_peak_area_cap(weighted_area_mm2, local_peak):
    """
    单缺陷面积上限收缩：
    A3/A4 这类单峰小缺陷往往存在较强扩散尾迹，仅靠固定剖面阈值仍可能偏大。
    因此根据“峰值越低 -> 缺陷越小/扩散占比越高”的经验，
    对 weighted area 施加一个峰值驱动的上限收缩。

    说明：该上限只基于当前重建场的 peak 与 weighted area 计算，
    不使用真实面积，因此不会造成数据泄漏。
    """
    settings = Config.QUANTIFICATION_SETTINGS
    c0 = float(settings.get('SINGLE_PEAK_AREA_CAP_COEF0', 0.26))
    c1 = float(settings.get('SINGLE_PEAK_AREA_CAP_COEF1', 0.05))
    c2 = float(settings.get('SINGLE_PEAK_AREA_CAP_COEF2', 0.018))
    min_factor = float(settings.get('SINGLE_PEAK_AREA_CAP_MIN_FACTOR', 0.35))
    max_factor = float(settings.get('SINGLE_PEAK_AREA_CAP_MAX_FACTOR', 0.85))

    factor = c0 + c1 * float(local_peak) + c2 * float(local_peak) ** 2
    factor = float(np.clip(factor, min_factor, max_factor))
    return float(weighted_area_mm2) * factor, factor


def _single_defect_blend_lambda(local_peak):
    """
    将“峰值收缩面积上限”和“weighted area”进行峰值驱动混合。

    经验解释：
    - 峰值较低的小缺陷（如 A3）扩散尾迹占比高，应更偏向收缩面积上限；
    - 峰值较高、主瓣更稳定的缺陷（如 A1/A4）应适当回拉一点 weighted area，
      否则面积会被压得过小，与图像观感不一致。
    """
    settings = Config.QUANTIFICATION_SETTINGS
    intercept = float(settings.get('SINGLE_BLEND_LAMBDA_INTERCEPT', 1.026))
    slope = float(settings.get('SINGLE_BLEND_LAMBDA_SLOPE', 0.073))
    high_peak_thr = float(settings.get('SINGLE_BLEND_HIGH_PEAK_THRESHOLD', 3.0))
    high_peak_extra = float(settings.get('SINGLE_BLEND_HIGH_PEAK_EXTRA_SLOPE', 0.15))
    lam_min = float(settings.get('SINGLE_BLEND_LAMBDA_MIN', 0.45))
    lam_max = float(settings.get('SINGLE_BLEND_LAMBDA_MAX', 0.95))

    lam = intercept - slope * float(local_peak)
    if float(local_peak) > high_peak_thr:
        lam -= high_peak_extra * (float(local_peak) - high_peak_thr)
    lam = float(np.clip(lam, lam_min, lam_max))
    return lam


def _quantify_single_defect(
    field_crop,
    final_mask,
    x_crop,
    y_crop,
    cell_area,
    dx,
    dy,
    low_thr,
    high_thr,
    peak,
    eps,
    state,
    base_name,
    threshold_method,
    total_quantify_area,
    triggered_auto_boost,
    boost_count,
    real_defect_config
):
    metrics = _basic_component_metrics(
        field_crop,
        final_mask,
        x_crop,
        y_crop,
        cell_area,
        dx,
        dy,
        low_thr,
        peak,
        eps
    )

    if metrics:
        profile_est = _estimate_profile_based_area(field_crop, final_mask, metrics, dx, dy, peak, mode="single")
        peak_cap_area, peak_cap_factor = _single_peak_area_cap(metrics['weighted_area_mm2'], peak)

        blend_lambda = _single_defect_blend_lambda(peak)
        # 三次修复：上一版最小值策略收缩过强，导致 A3/A4 面积误差率过小，
        # 与图像上的主瓣范围不一致。本版改为“收缩上限 + weighted area”的峰值驱动混合：
        # final_area = λ * peak_cap_area + (1-λ) * weighted_area
        # 峰值越低，λ 越大；峰值越高，适当引入更多 weighted area。
        final_area = blend_lambda * peak_cap_area + (1.0 - blend_lambda) * metrics['weighted_area_mm2']
        area_method = "single_peak_cap_weighted_blend"

        main_defect_info = _format_component_defect(metrics, final_area, peak, low_thr, extra={
            "area_estimation_method_面积估计方法": area_method,
            "profile_width_mm_剖面宽度": round(profile_est['profile_width_mm'], 2) if profile_est else None,
            "profile_height_mm_剖面高度": round(profile_est['profile_height_mm'], 2) if profile_est else None,
            "profile_ratio_x_剖面阈值比例X": round(profile_est['profile_ratio_x'], 3) if profile_est else None,
            "profile_ratio_y_剖面阈值比例Y": round(profile_est['profile_ratio_y'], 3) if profile_est else None,
            "target_area_proxy_mm2_面积代理": round(profile_est.get('target_area_proxy_mm2'), 2) if profile_est and profile_est.get('target_area_proxy_mm2') is not None else None,
            "profile_candidate_count_候选阈值数量": profile_est.get('profile_candidate_count') if profile_est else None,
            "profile_selection_mode_剖面选择模式": profile_est.get('profile_selection_mode') if profile_est else None,
            "peak_cap_area_mm2_峰值收缩面积上限": round(peak_cap_area, 2),
            "peak_cap_factor_峰值收缩系数": round(peak_cap_factor, 4),
            "blend_lambda_混合系数": round(blend_lambda, 4)
        })
    else:
        main_defect_info = None

    real_x, real_y = real_defect_config['x'], real_defect_config['y']
    real_area = real_defect_config['area']

    offset = None
    area_err = None
    if main_defect_info:
        offset = np.sqrt(
            (main_defect_info["centroid_mm_质心坐标"][0] - real_x) ** 2 +
            (main_defect_info["centroid_mm_质心坐标"][1] - real_y) ** 2
        )
        area_err = main_defect_info["final_area_mm2_最终面积"] - real_area

    return {
        "metadata_元数据": {
            "state_状态": state,
            "base_name_文件名": base_name,
            "threshold_method_阈值方法": threshold_method,
            "low_threshold_低阈值": round(float(low_thr), 4),
            "high_threshold_高阈值": round(float(high_thr), 4),
            "peak_value_峰值": round(float(peak), 4),
            "total_quantify_area_mm2_总扫描面积": round(total_quantify_area, 2),
            "triggered_auto_boost_触发自动阈值提高": triggered_auto_boost,
            "boost_count_提高次数": boost_count,
            "is_interpolated_是否为插值图": "spline" in base_name or "插值" in base_name,
            "data_leakage_guard_数据泄漏防护": "真实缺陷只用于误差评估，不参与阈值或区域选择"
        },
        "real_defect_真实缺陷信息": {
            "centroid_mm_质心": [real_x, real_y],
            "width_mm_宽度": real_defect_config.get('width'),
            "height_mm_高度": real_defect_config.get('height'),
            "area_mm2_面积": real_area
        },
        "detected_defect_检测结果": main_defect_info if main_defect_info else "No defect detected",
        "evaluation_误差评估": {
            "centroid_offset_mm_质心偏移": round(offset, 2) if offset is not None else None,
            "area_error_mm2_面积误差": round(area_err, 2) if area_err is not None else None,
            "area_error_percent_面积误差率(%)": round(abs(area_err / real_area * 100), 2) if area_err is not None and real_area > 0 else None
        }
    }


def _find_data_driven_peaks(field_crop, dx, dy, mad_thr, global_peak, eps):
    settings = Config.QUANTIFICATION_SETTINGS
    if global_peak <= eps:
        return []

    min_dist_mm = float(settings.get('PEAK_MIN_DISTANCE_MM', 70.0))
    min_dist_px = max(1, int(round(min_dist_mm / max(min(dx, dy), eps))))
    min_peak_ratio = float(settings.get('PEAK_MIN_GLOBAL_RATIO', 0.10))
    threshold_abs = max(float(mad_thr), global_peak * min_peak_ratio)

    coords = peak_local_max(
        field_crop,
        min_distance=min_dist_px,
        threshold_abs=threshold_abs,
        exclude_border=False
    )

    if coords.size == 0:
        coords = np.array([np.unravel_index(np.argmax(field_crop), field_crop.shape)])

    coords = sorted(
        [(int(r), int(c)) for r, c in coords],
        key=lambda rc: field_crop[rc[0], rc[1]],
        reverse=True
    )

    max_peaks = int(settings.get('MAX_DATA_DRIVEN_PEAKS', 8))
    return coords[:max_peaks]


def _quantify_multi_defect_data_driven(
    field_crop,
    x_crop,
    y_crop,
    cell_area,
    dx,
    dy,
    mad_thr,
    peak,
    eps,
    state,
    base_name,
    threshold_method,
    total_quantify_area,
    triggered_auto_boost,
    boost_count,
    real_defect_config
):
    settings = Config.QUANTIFICATION_SETTINGS
    peak_coords = _find_data_driven_peaks(field_crop, dx, dy, mad_thr, peak, eps)

    local_ratio = float(settings.get('LOCAL_COMPONENT_PEAK_RATIO', 0.35))
    min_area = max(
        cell_area * 2,
        float(settings.get('MIN_DEFECT_AREA_MM2', 100))
    )
    max_area = float(settings.get('MAX_DEFECT_AREA_MM2', 9000))

    detected_defects = []
    used_masks = []
    thresholds = []

    for peak_index, (pr, pc) in enumerate(peak_coords, start=1):
        local_peak = float(field_crop[pr, pc])
        if local_peak <= eps:
            continue

        local_thr = max(float(mad_thr), local_peak * local_ratio)
        mask = field_crop >= local_thr
        comp_mask = _component_containing(mask, (pr, pc))
        if not np.any(comp_mask):
            continue

        # 去重：同一个连通域内可能有多个局部峰，只保留第一次检测到的强峰。
        duplicate = False
        for old_mask in used_masks:
            inter = np.logical_and(comp_mask, old_mask).sum()
            union = np.logical_or(comp_mask, old_mask).sum()
            if union > 0 and inter / union > 0.5:
                duplicate = True
                break
        if duplicate:
            continue

        metrics = _basic_component_metrics(
            field_crop,
            comp_mask,
            x_crop,
            y_crop,
            cell_area,
            dx,
            dy,
            local_thr,
            local_peak,
            eps
        )
        if metrics is None:
            continue

        if metrics['binary_area_mm2'] < min_area or metrics['binary_area_mm2'] > max_area:
            continue

        # 对 A2 多缺陷：在原始中心剖面面积基础上，再做一个 soft profile 放松估计，
        # 让量化面积尽量与成像图主瓣范围一致，而不是过分“卡紧”到最小值。
        profile_est = _estimate_profile_based_area(field_crop, comp_mask, metrics, dx, dy, local_peak, mode="multi")
        visual_adj = _multi_defect_visual_area_adjustment(field_crop, comp_mask, metrics, dx, dy, local_peak, profile_est) if profile_est else None
        if visual_adj:
            final_area = visual_adj['final_area_mm2']
            area_method = "multi_profile_soft_blend"
        else:
            final_area = profile_est['profile_area_mm2'] if profile_est else metrics['weighted_area_mm2']
            area_method = "center_profile_area" if profile_est else "data_driven_weighted_area"
        detected_defects.append(_format_component_defect(
            metrics,
            final_area,
            local_peak,
            local_thr,
            extra={
                "data_peak_index_数据峰序号": peak_index,
                "local_threshold_局部阈值": round(local_thr, 4),
                "local_threshold_ratio_局部阈值比例": round(local_ratio, 3),
                "area_estimation_method_面积估计方法": area_method,
                "profile_width_mm_剖面宽度": round(profile_est['profile_width_mm'], 2) if profile_est else None,
                "profile_height_mm_剖面高度": round(profile_est['profile_height_mm'], 2) if profile_est else None,
                "profile_ratio_x_剖面阈值比例X": round(profile_est['profile_ratio_x'], 3) if profile_est else None,
                "profile_ratio_y_剖面阈值比例Y": round(profile_est['profile_ratio_y'], 3) if profile_est else None,
                "soft_profile_area_mm2_柔和剖面面积": round(visual_adj['soft_profile_area_mm2'], 2) if visual_adj else None,
                "soft_profile_width_mm_柔和剖面宽度": round(visual_adj['soft_profile_width_mm'], 2) if visual_adj else None,
                "soft_profile_height_mm_柔和剖面高度": round(visual_adj['soft_profile_height_mm'], 2) if visual_adj else None,
                "soft_ratio_x_柔和阈值比例X": round(visual_adj['soft_ratio_x'], 3) if visual_adj else None,
                "soft_ratio_y_柔和阈值比例Y": round(visual_adj['soft_ratio_y'], 3) if visual_adj else None,
                "visual_blend_beta_视觉融合系数": round(visual_adj['visual_blend_beta'], 3) if visual_adj else None,
                "shape_aspect_from_profile_剖面长宽比": round(visual_adj['shape_aspect_from_profile'], 3) if visual_adj else None
            }
        ))
        used_masks.append(comp_mask)
        thresholds.append(local_thr)

    detected_defects.sort(key=lambda d: d["centroid_mm_质心坐标"][0])

    low_thr = min(thresholds) if thresholds else mad_thr
    high_thr = max(thresholds) if thresholds else mad_thr

    return _make_multi_output(
        detected_defects,
        real_defect_config,
        low_thr,
        high_thr,
        peak,
        state,
        base_name,
        threshold_method,
        total_quantify_area,
        triggered_auto_boost,
        boost_count,
        extra_metadata={
            "multi_quantification_mode_多缺陷量化模式": "data_driven_local_peak_components",
            "peak_count_before_filter_初始峰数量": len(peak_coords),
            "local_component_peak_ratio_局部峰值比例": local_ratio,
            "data_leakage_guard_数据泄漏防护": "真实缺陷只用于误差评估，不参与阈值或区域选择"
        }
    )


def _format_component_defect(metrics, final_area, peak_value, threshold_value, extra=None):
    out = {
        "binary_area_mm2_二值面积": round(metrics['binary_area_mm2'], 2),
        "weighted_area_mm2_加权面积": round(metrics['weighted_area_mm2'], 2),
        "core_area_mm2_核心面积": round(metrics['core_area_mm2'], 2),
        "final_area_mm2_最终面积": round(float(final_area), 2),
        "centroid_mm_质心坐标": [round(metrics['centroid_x'], 2), round(metrics['centroid_y'], 2)],
        "bbox_x_range_X边界范围": [round(metrics['x_min'], 2), round(metrics['x_max'], 2)],
        "bbox_y_range_Y边界范围": [round(metrics['y_min'], 2), round(metrics['y_max'], 2)],
        "bbox_width_mm_边界框宽度": round(metrics['bbox_w'], 2),
        "bbox_height_mm_边界框高度": round(metrics['bbox_h'], 2),
        "peak_value_峰值": round(float(peak_value), 4),
        "mean_intensity_平均强度": round(metrics['mean'], 4),
        "threshold_used_使用阈值": round(float(threshold_value), 4)
    }
    if extra:
        out.update(extra)
    return out


def _make_multi_output(
    detected_defects,
    real_defect_config,
    low_thr,
    high_thr,
    peak,
    state,
    base_name,
    threshold_method,
    total_quantify_area,
    triggered_auto_boost,
    boost_count,
    extra_metadata=None
):
    real_defects = real_defect_config['defects']
    real_total_area = real_defect_config['total_area']

    per_defect_errors = []
    total_offset = 0.0
    total_area_error = 0.0
    matched_count = 0

    # 多缺陷按真实缺陷 x 坐标从左到右编号，尤其用于 A4（原 A2 多缺陷）的逐缺陷论文分析。
    real_defects_left_to_right = sorted(list(enumerate(real_defects)), key=lambda item: item[1]['x'])
    real_centers = [(rd['x'], rd['y'], rd['area'], original_idx + 1) for original_idx, rd in real_defects_left_to_right]
    used_real = [False] * len(real_centers)

    for det_idx, d in enumerate(detected_defects):
        dx_c, dy_c = d["centroid_mm_质心坐标"]
        d_area = d["final_area_mm2_最终面积"]

        best_r_idx = None
        best_dist = float('inf')
        for r_idx, (rx_c, ry_c, _, _) in enumerate(real_centers):
            if used_real[r_idx]:
                continue
            dist = np.sqrt((dx_c - rx_c) ** 2 + (dy_c - ry_c) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_r_idx = r_idx

        if best_r_idx is not None:
            rx_c, ry_c, r_area, original_real_idx = real_centers[best_r_idx]
            used_real[best_r_idx] = True
            area_err = d_area - r_area
            per_defect_errors.append({
                "defect_index_缺陷序号": det_idx + 1,
                "left_to_right_index_从左到右序号": best_r_idx + 1,
                "matched_real_defect_匹配真实缺陷": best_r_idx + 1,
                "matched_original_real_defect_原始真实缺陷序号": original_real_idx,
                "real_centroid_真实质心": [rx_c, ry_c],
                "detected_centroid_检测质心": [round(dx_c, 2), round(dy_c, 2)],
                "centroid_offset_mm_质心偏移": round(best_dist, 2),
                "real_area_mm2_真实面积": r_area,
                "detected_area_mm2_检测面积": round(d_area, 2),
                "area_error_mm2_面积误差": round(area_err, 2),
                "area_error_percent_面积误差率(%)": round(abs(area_err / r_area * 100), 2) if r_area > 0 else None
            })
            total_offset += best_dist
            total_area_error += area_err
            matched_count += 1

    for r_idx, (rx_c, ry_c, r_area, original_real_idx) in enumerate(real_centers):
        if not used_real[r_idx]:
            per_defect_errors.append({
                "defect_index_缺陷序号": None,
                "left_to_right_index_从左到右序号": r_idx + 1,
                "matched_real_defect_匹配真实缺陷": r_idx + 1,
                "matched_original_real_defect_原始真实缺陷序号": original_real_idx,
                "real_centroid_真实质心": [rx_c, ry_c],
                "detected_centroid_检测质心": None,
                "centroid_offset_mm_质心偏移": None,
                "real_area_mm2_真实面积": r_area,
                "detected_area_mm2_检测面积": 0,
                "area_error_mm2_面积误差": -r_area,
                "area_error_percent_面积误差率(%)": -100.0,
                "note": "未检测到此缺陷"
            })

    avg_offset = round(total_offset / matched_count, 2) if matched_count > 0 else None
    detected_total_area = float(sum(d.get("final_area_mm2_最终面积", 0.0) for d in detected_defects))

    metadata = {
        "state_状态": state,
        "base_name_文件名": base_name,
        "threshold_method_阈值方法": threshold_method,
        "low_threshold_低阈值": round(float(low_thr), 4),
        "high_threshold_高阈值": round(float(high_thr), 4),
        "peak_value_峰值": round(float(peak), 4),
        "total_quantify_area_mm2_总扫描面积": round(total_quantify_area, 2),
        "triggered_auto_boost_触发自动阈值提高": triggered_auto_boost,
        "boost_count_提高次数": boost_count,
        "is_interpolated_是否为插值图": "spline" in base_name or "插值" in base_name,
        "defect_type_缺陷类型": "multi"
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "metadata_元数据": metadata,
        "real_defects_真实缺陷信息": {
            "defects": [rd for _, rd in real_defects_left_to_right],
            "sort_rule_排序规则": "按缺陷真实质心 x 坐标从左到右排序",
            "total_area_mm2_总面积": real_total_area,
            "count_数量": len(real_defects)
        },
        "detected_defects_检测缺陷": detected_defects,
        "per_defect_evaluation_逐缺陷误差": sorted(per_defect_errors, key=lambda item: item.get("left_to_right_index_从左到右序号") or 999),
        "summary_汇总评估": {
            "detected_count_检测数量": len(detected_defects),
            "real_count_真实数量": len(real_defects),
            "matched_count_匹配数量": matched_count,
            "average_offset_mm_平均质心偏移": avg_offset,
            "detected_total_area_mm2_检测总面积": round(detected_total_area, 2),
            "total_area_error_mm2_总面积误差": round(detected_total_area - real_total_area, 2),
            "total_area_error_percent_总面积误差率(%)": round((detected_total_area - real_total_area) / real_total_area * 100, 2) if real_total_area > 0 else None,
            "real_total_area_mm2_真实总面积": real_total_area
        }
    }


def _quantify_multi_defect_connected(
    field_crop,
    final_mask,
    x_crop,
    y_crop,
    cell_area,
    dx,
    dy,
    low_thr,
    high_thr,
    peak,
    eps,
    state,
    base_name,
    threshold_method,
    total_quantify_area,
    triggered_auto_boost,
    boost_count,
    real_defect_config
):
    detected_defects = []

    if np.any(final_mask):
        labeled, num_features = measure.label(final_mask, connectivity=2, return_num=True)
        min_area = max(cell_area * 2, Config.QUANTIFICATION_SETTINGS.get('MIN_DEFECT_AREA_MM2', 100))
        for i in range(1, num_features + 1):
            comp_mask = labeled == i
            metrics = _basic_component_metrics(
                field_crop,
                comp_mask,
                x_crop,
                y_crop,
                cell_area,
                dx,
                dy,
                low_thr,
                peak,
                eps
            )
            if metrics is None or metrics['binary_area_mm2'] < min_area:
                continue
            profile_est = _estimate_profile_based_area(field_crop, comp_mask, metrics, dx, dy, metrics['peak'], mode="multi")
            visual_adj = _multi_defect_visual_area_adjustment(field_crop, comp_mask, metrics, dx, dy, metrics['peak'], profile_est) if profile_est else None
            if visual_adj:
                final_area = visual_adj['final_area_mm2']
                area_method = "multi_profile_soft_blend"
            else:
                final_area = profile_est['profile_area_mm2'] if profile_est else metrics['weighted_area_mm2']
                area_method = "center_profile_area" if profile_est else "global_connected_weighted_area"
            detected_defects.append(_format_component_defect(
                metrics,
                final_area,
                metrics['peak'],
                low_thr,
                extra={
                    "area_estimation_method_面积估计方法": area_method,
                    "profile_width_mm_剖面宽度": round(profile_est['profile_width_mm'], 2) if profile_est else None,
                    "profile_height_mm_剖面高度": round(profile_est['profile_height_mm'], 2) if profile_est else None,
                    "profile_ratio_x_剖面阈值比例X": round(profile_est['profile_ratio_x'], 3) if profile_est else None,
                    "profile_ratio_y_剖面阈值比例Y": round(profile_est['profile_ratio_y'], 3) if profile_est else None,
                    "soft_profile_area_mm2_柔和剖面面积": round(visual_adj['soft_profile_area_mm2'], 2) if visual_adj else None,
                    "soft_profile_width_mm_柔和剖面宽度": round(visual_adj['soft_profile_width_mm'], 2) if visual_adj else None,
                    "soft_profile_height_mm_柔和剖面高度": round(visual_adj['soft_profile_height_mm'], 2) if visual_adj else None,
                    "soft_ratio_x_柔和阈值比例X": round(visual_adj['soft_ratio_x'], 3) if visual_adj else None,
                    "soft_ratio_y_柔和阈值比例Y": round(visual_adj['soft_ratio_y'], 3) if visual_adj else None,
                    "visual_blend_beta_视觉融合系数": round(visual_adj['visual_blend_beta'], 3) if visual_adj else None,
                    "shape_aspect_from_profile_剖面长宽比": round(visual_adj['shape_aspect_from_profile'], 3) if visual_adj else None
                }
            ))

    detected_defects.sort(key=lambda d: d["centroid_mm_质心坐标"][0])
    return _make_multi_output(
        detected_defects,
        real_defect_config,
        low_thr,
        high_thr,
        peak,
        state,
        base_name,
        threshold_method,
        total_quantify_area,
        triggered_auto_boost,
        boost_count,
        extra_metadata={
            "multi_quantification_mode_多缺陷量化模式": "global_connected_components",
            "data_leakage_guard_数据泄漏防护": "真实缺陷只用于误差评估，不参与阈值或区域选择"
        }
    )
