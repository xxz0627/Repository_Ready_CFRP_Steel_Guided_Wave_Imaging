"""可视化模块"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from config import Config


class Visualizer:
    """可视化器:生成各种成像结果图"""

    _font_prop_en = None
    _font_prop_zh = None

    @staticmethod
    def _resolve_font_properties():
        """优先使用项目内自带中文字体,其次回退系统字体,避免中文变方框"""
        if Visualizer._font_prop_en is None:
            Visualizer._font_prop_en = fm.FontProperties(family=Config.FONT_FAMILY)

        if Visualizer._font_prop_zh is not None:
            return

        # 1) 优先使用项目内自带字体,这样项目拷贝到别的电脑也能正常显示中文
        bundled_font = getattr(Config, 'PROJECT_ZH_FONT_FILE', None)
        if bundled_font and os.path.exists(bundled_font):
            try:
                Visualizer._font_prop_zh = fm.FontProperties(fname=bundled_font)
                print(f"中文字体已加载(项目内置): {bundled_font}")
                return
            except Exception as exc:
                print(f"项目内置中文字体加载失败: {exc}")

        # 2) 按family 名称寻找系统中文字体
        for family in Config.FONT_FAMILY_ZH:
            try:
                font_path = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
                if font_path and os.path.exists(font_path):
                    Visualizer._font_prop_zh = fm.FontProperties(fname=font_path)
                    print(f"中文字体已加载 {family} -> {font_path}")
                    return
            except Exception:
                continue

        # 3) 再按常见中文字体文件名兜底搜索
        candidates = [
            'notosanscjk', 'notoserifcjk', 'sourcehansans', 'sourcehanserif',
            'msyh', 'msyhbd', 'simhei', 'simsun', 'simkai', 'arialuni', 'uming', 'ukai', 'wqy'
        ]
        for font_path in fm.findSystemFonts(fontext='ttf') + fm.findSystemFonts(fontext='ttc'):
            lower_name = os.path.basename(font_path).lower()
            if any(key in lower_name for key in candidates):
                try:
                    Visualizer._font_prop_zh = fm.FontProperties(fname=font_path)
                    print(f"中文字体已加载(兜底): {font_path}")
                    return
                except Exception:
                    continue

        # 4) 最后退到family 列表
        Visualizer._font_prop_zh = fm.FontProperties(family=Config.FONT_FAMILY_ZH)
        print("警告: 未找到明确的中文字体文件,已退到family 列表;若仍显示方框,请检查fonts/NotoSansCJK-Regular.ttc 是否存在")

    @staticmethod
    def initialize():
        """初始化matplotlib绘图设置"""
        Visualizer._resolve_font_properties()
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': Config.FONT_FAMILY,
            'axes.unicode_minus': False,
            'figure.dpi': Config.PLOT_DPI
        })

    @staticmethod
    def _get_font_prop(language='en', size=None):
        Visualizer._resolve_font_properties()
        base_prop = Visualizer._font_prop_zh if language == 'zh' else Visualizer._font_prop_en
        file_path = None
        try:
            file_path = base_prop.get_file()
        except Exception:
            file_path = None

        if file_path:
            return fm.FontProperties(fname=file_path, size=size)

        families = None
        try:
            families = base_prop.get_family()
        except Exception:
            families = Config.FONT_FAMILY_ZH if language == 'zh' else Config.FONT_FAMILY
        return fm.FontProperties(family=families, size=size)

    @staticmethod
    def _setup_axes(ax, language='en'):
        """设置坐标轴样式"""
        major_ticks_x = np.arange(*Config.COORD_RANGE_X, Config.MAJOR_TICK_INTERVAL)
        major_ticks_y = np.arange(*Config.COORD_RANGE_Y, Config.MAJOR_TICK_INTERVAL)

        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)

        minor_ticks_x = np.arange(*Config.COORD_RANGE_X, Config.MINOR_TICK_INTERVAL)
        minor_ticks_y = np.arange(*Config.COORD_RANGE_Y, Config.MINOR_TICK_INTERVAL)

        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(minor_ticks_y, minor=True)

        ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlim(Config.COORD_RANGE_X)
        ax.set_ylim(Config.COORD_RANGE_Y)
        ax.set_aspect('equal')

        ax.set_xlabel(
            'X坐标 (mm)' if language == 'zh' else 'X Coordinate (mm)',
            fontproperties=Visualizer._get_font_prop(language, Config.FONT_SIZE_AXIS)
        )
        ax.set_ylabel(
            'Y坐标 (mm)' if language == 'zh' else 'Y Coordinate (mm)',
            fontproperties=Visualizer._get_font_prop(language, Config.FONT_SIZE_AXIS)
        )

        ax.tick_params(axis='both', labelsize=Config.FONT_SIZE_TICK)

    @staticmethod
    def _get_sensor_positions(data):
        tx_positions = data[['Tx_X', 'Tx_Y']].drop_duplicates().values
        rx_positions = data[['Rx_X', 'Rx_Y']].drop_duplicates().values
        return tx_positions, rx_positions

    @staticmethod
    def _create_sensor_mask(x_coords, y_coords):
        """创建试块边界掩码"""
        sample_polygon = Path(Config.SAMPLE_BOUNDARY)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        return sample_polygon.contains_points(points).reshape(len(y_coords), len(x_coords))

    @staticmethod
    def _plot_sensor_boundary(ax, all_sensors):
        boundary = Config.SAMPLE_BOUNDARY
        bx = [p[0] for p in boundary] + [boundary[0][0]]
        by = [p[1] for p in boundary] + [boundary[0][1]]
        ax.plot(bx, by, 'k-', alpha=0.7, lw=1.5, zorder=4)

    @staticmethod
    def generate_image_path(image_type, data_filename, language=None):
        """生成图像保存路径"""
        if not data_filename:
            return f'{image_type}.png'

        folder_name = os.path.basename(os.path.dirname(data_filename))
        base_name = os.path.splitext(os.path.basename(data_filename))[0]

        return Config.get_image_path(image_type, folder_name, base_name, language)

    @staticmethod
    def _smooth_display_field(field, x_coords, y_coords, title_key):
        """仅用于显示的轻微平滑,不改变量化输入"""
        settings = getattr(Config, 'DISPLAY_SMOOTHING', {})
        if not settings.get('enabled', False):
            return np.array(field, dtype=np.float64, copy=True)

        apply_to_iter = bool(settings.get('apply_to_iterated', False))
        apply_to_interp = bool(settings.get('apply_to_interpolated', True))
        if title_key == 'iterated_attenuation' and not apply_to_iter:
            return np.array(field, dtype=np.float64, copy=True)
        if title_key == 'spline_attenuation' and not apply_to_interp:
            return np.array(field, dtype=np.float64, copy=True)

        arr = np.array(field, dtype=np.float64, copy=True)
        if arr.size == 0:
            return arr

        dx = float(np.mean(np.diff(x_coords))) if len(x_coords) > 1 else 1.0
        dy = float(np.mean(np.diff(y_coords))) if len(y_coords) > 1 else 1.0

        sigma_x_mm = float(settings.get('sigma_x_mm', 0.0))
        sigma_y_mm = float(settings.get('sigma_y_mm', 0.0))
        sigma_x = max(sigma_x_mm / max(dx, 1e-9), 0.0)
        sigma_y = max(sigma_y_mm / max(dy, 1e-9), 0.0)

        if sigma_x <= 0 and sigma_y <= 0:
            return arr

        blurred = gaussian_filter(arr, sigma=(sigma_y, sigma_x), mode='nearest')
        blend = float(settings.get('blend', 0.60))
        blend = np.clip(blend, 0.0, 1.0)

        out = (1.0 - blend) * arr + blend * blurred

        if settings.get('preserve_peak', True):
            src_max = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
            out_max = float(np.nanmax(out)) if np.isfinite(out).any() else 0.0
            if out_max > 0 and src_max > 0:
                out *= (src_max / out_max)

        min_alpha, max_alpha = Config.ATTENUATION_RANGE
        return np.clip(out, min_alpha, max_alpha)

    @staticmethod
    def _apply_display_realism(field):
        """仅用于显示:轻压背景、压弱尾迹、轻微保峰,让缺陷更像自然成像"""
        settings = getattr(Config, 'DISPLAY_REALISM', {})
        if not settings.get('enabled', False):
            return np.array(field, dtype=np.float64, copy=True)

        arr = np.array(field, dtype=np.float64, copy=True)
        if arr.size == 0 or not np.isfinite(arr).any():
            return arr

        min_alpha, max_alpha = Config.ATTENUATION_RANGE
        finite_vals = arr[np.isfinite(arr)]
        bg_p = float(settings.get('background_percentile', 0.0))
        bg_strength = float(settings.get('background_strength', 0.0))
        if bg_p > 0 and bg_strength > 0 and finite_vals.size > 0:
            bg = np.percentile(finite_vals, bg_p) * bg_strength
            arr = np.clip(arr - bg, min_alpha, max_alpha)

        peak = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
        if peak > 0:
            gamma = float(settings.get('tail_gamma', 1.0))
            norm = np.clip(arr / peak, 0.0, 1.0)
            if gamma != 1.0:
                norm = np.power(norm, gamma)
            boost = float(settings.get('peak_boost', 1.0))
            arr = np.clip(norm * peak * boost, min_alpha, max_alpha)

        return arr

    @staticmethod
    def _infer_data_key(data_filename):
        if not data_filename:
            return None
        base_name = os.path.basename(str(data_filename))
        for key in ['A4', 'A3', 'A2', 'A1']:
            if key in base_name:
                return key
        return None

    @staticmethod
    def _get_reference_defects(data_key):
        real_defects = Config.QUANTIFICATION_SETTINGS.get('REAL_DEFECTS', {})
        item = real_defects.get(data_key)
        if not item:
            return []
        if isinstance(item, dict) and item.get('type') == 'multi' and 'defects' in item:
            return item['defects']
        if isinstance(item, dict):
            return [item]
        return []

    @staticmethod
    def _overlay_reference_defects(ax, data_key):
        defects = Visualizer._get_reference_defects(data_key)
        for idx, defect in enumerate(defects, start=1):
            x_center = float(defect['x'])
            y_center = float(defect['y'])
            width = float(defect['width'])
            height = float(defect['height'])
            rect = Rectangle(
                (x_center - width / 2.0, y_center - height / 2.0),
                width,
                height,
                fill=False,
                edgecolor='red',
                linewidth=2.0,
                linestyle=(0, (5, 3)),
                zorder=6
            )
            ax.add_patch(rect)

    @staticmethod
    def _build_quant_text(quantified_data, data_key, language='zh'):
        if not quantified_data:
            return None

        real_defects = Config.QUANTIFICATION_SETTINGS.get('REAL_DEFECTS', {})
        item = real_defects.get(data_key, {})
        is_multi = isinstance(item, dict) and item.get('type') == 'multi'

        if not is_multi:
            lines = ['量化结果' if language == 'zh' else 'Quantification']
            eva = quantified_data.get('evaluation_误差评估', {})
            offset = eva.get('centroid_offset_mm_质心偏移', None)
            area_pct = eva.get('area_error_percent_面积误差率(%)', None)
            offset_label = '质心偏移' if language == 'zh' else 'Centroid Offset'
            area_label = '面积误差率' if language == 'zh' else 'Area Error'
            if offset is not None:
                lines.append(f'{offset_label}: {float(offset):.2f} mm')
            if area_pct is not None:
                lines.append(f'{area_label}: {float(area_pct):.2f}%')
            return "\n".join(lines)

        per_defects = quantified_data.get('per_defect_evaluation_逐缺陷误差', [])
        if per_defects:
            lines = ['量化结果(从左往右)' if language == 'zh' else 'Quantification (Left to Right)']
            offset_label = '偏移' if language == 'zh' else 'Offset'
            area_label = '面积误差率' if language == 'zh' else 'Area Error'
            defect_label = '缺陷' if language == 'zh' else 'Defect'
            for idx, item in enumerate(per_defects, start=1):
                offset = item.get('centroid_offset_mm_质心偏移', None)
                area_pct = item.get('area_error_percent_面积误差率(%)', None)
                if offset is None and area_pct is None:
                    continue
                offset_txt = f'{float(offset):.2f} mm' if offset is not None else '--'
                area_txt = f'{abs(float(area_pct)):.2f}%' if area_pct is not None else '--'
                lines.append(f'{defect_label}{idx}: {offset_label} {offset_txt}')
                lines.append(f'        {area_label} {area_txt}')
            return "\n".join(lines)

        return None

    @staticmethod
    def _plot_attenuation_field(
        ax,
        field,
        tx_pos,
        rx_pos,
        x_coords,
        y_coords,
        language='en',
        title_key='iterated_attenuation',
        data_key=None,
        quantified_data=None
    ):
        vmin, vmax = Config.ATTENUATION_RANGE
        cmap = 'viridis'

        display_field = Visualizer._smooth_display_field(field, x_coords, y_coords, title_key)
        display_field = Visualizer._apply_display_realism(display_field)

        mask = Visualizer._create_sensor_mask(x_coords, y_coords)
        masked_field = np.ma.masked_where(~mask, display_field)

        values = masked_field.compressed()

        if values.size > 0 and Config.DISPLAY_CLIP_PERCENTILES is not None:
            p_low, p_high = Config.DISPLAY_CLIP_PERCENTILES
            vmin_disp, vmax_disp = np.percentile(values, [p_low, p_high])

            if vmin_disp >= vmax_disp:
                vmin_disp, vmax_disp = vmin, vmax
        else:
            vmin_disp, vmax_disp = vmin, vmax

        norm = None
        if Config.DISPLAY_GAMMA is not None and Config.DISPLAY_GAMMA != 1.0:
            norm = PowerNorm(
                gamma=Config.DISPLAY_GAMMA,
                vmin=vmin_disp,
                vmax=vmax_disp
            )

        imshow_kwargs = dict(
            extent=[*Config.COORD_RANGE_X, *Config.COORD_RANGE_Y],
            origin='lower',
            cmap=cmap,
            interpolation=getattr(Config, 'IMAGE_INTERPOLATION', 'bilinear')
        )

        if norm is None:
            im = ax.imshow(masked_field, vmin=vmin_disp, vmax=vmax_disp, **imshow_kwargs)
        else:
            im = ax.imshow(masked_field, norm=norm, **imshow_kwargs)

        ax.set_title(
            Config.TITLES[title_key][language],
            fontproperties=Visualizer._get_font_prop(language, Config.FONT_SIZE_TITLE)
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(
            Config.COLORBAR_LABELS['attenuation'][language],
            fontproperties=Visualizer._get_font_prop(language, Config.FONT_SIZE_COLORBAR)
        )
        cbar.ax.tick_params(labelsize=Config.FONT_SIZE_TICK)

        Visualizer._plot_sensor_boundary(ax, None)
        if data_key:
            Visualizer._overlay_reference_defects(ax, data_key)
        Visualizer._setup_axes(ax, language)

        return im

    @staticmethod
    def plot_straight_rays(data, data_filename=None):
        plt.figure(figsize=Config.FIG_SIZE)

        tx_pos, rx_pos = Visualizer._get_sensor_positions(data)

        plt.scatter(tx_pos[:, 0], tx_pos[:, 1], c='r', s=40, marker='^', label='Tx', zorder=5)
        plt.scatter(rx_pos[:, 0], rx_pos[:, 1], c='b', s=20, marker='o', label='Rx', zorder=5)

        sample_size = min(Config.RAY_SAMPLE_SIZE, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)

        for _, row in data.iloc[sample_indices].iterrows():
            plt.plot(
                [row['Tx_X'], row['Rx_X']],
                [row['Tx_Y'], row['Rx_Y']],
                'b-',
                alpha=0.6,
                lw=1.2
            )

        Visualizer._setup_axes(plt.gca())
        plt.title(Config.TITLES['straight_rays']['en'])
        plt.legend()
        plt.tight_layout()

        image_filename = Visualizer.generate_image_path('straight_rays', data_filename)
        plt.savefig(image_filename, bbox_inches='tight', dpi=Config.PLOT_DPI)
        print(f"直射路径图已保存为 {image_filename}")
        plt.close()

    @staticmethod
    def plot_actual_rays(data, engine, data_filename=None):
        plt.figure(figsize=Config.FIG_SIZE)

        tx_pos, rx_pos = Visualizer._get_sensor_positions(data)

        plt.scatter(tx_pos[:, 0], tx_pos[:, 1], c='r', s=40, marker='^', label='Tx', zorder=5)
        plt.scatter(rx_pos[:, 0], rx_pos[:, 1], c='b', s=20, marker='o', label='Rx', zorder=5)

        sample_size = min(Config.RAY_SAMPLE_SIZE, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        plotted_pairs = set()

        for idx in sample_indices:
            row = data.iloc[idx]

            tx, ty, rx, ry = row['Tx_X'], row['Tx_Y'], row['Rx_X'], row['Rx_Y']

            key = (
                round(float(tx), 3),
                round(float(ty), 3),
                round(float(rx), 3),
                round(float(ry), 3)
            )
            key_rev = (key[2], key[3], key[0], key[1])

            if key in plotted_pairs or key_rev in plotted_pairs:
                continue

            plotted_pairs.add(key)

            path_x, path_y = engine.get_ray_path(idx)
            if path_x is not None and path_y is not None:
                plt.plot(path_x, path_y, 'b-', alpha=0.6, lw=1.2)

        Visualizer._setup_axes(plt.gca())
        plt.title(Config.TITLES['actual_rays']['en'])
        plt.legend()
        plt.tight_layout()

        image_filename = Visualizer.generate_image_path('actual_rays', data_filename)
        plt.savefig(image_filename, bbox_inches='tight', dpi=Config.PLOT_DPI)
        print(f"实际路径图已保存为 {image_filename}")
        plt.close()

    @staticmethod
    def plot_iterated_attenuation(engine, language='en', data_filename=None, quantified_data=None):
        fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

        tx_pos, rx_pos = Visualizer._get_sensor_positions(engine.data)
        x_coords = np.linspace(*Config.COORD_RANGE_X, Config.GRID_NX)
        y_coords = np.linspace(*Config.COORD_RANGE_Y, Config.GRID_NY)

        data_key = Visualizer._infer_data_key(data_filename)
        Visualizer._plot_attenuation_field(
            ax,
            engine.reconstructor.attenuation_field,
            tx_pos,
            rx_pos,
            x_coords,
            y_coords,
            language,
            'iterated_attenuation',
            data_key,
            quantified_data
        )

        plt.tight_layout()

        image_filename = Visualizer.generate_image_path(
            'iterated_attenuation',
            data_filename,
            language
        )
        plt.savefig(image_filename, bbox_inches='tight', dpi=Config.PLOT_DPI)
        print(f"迭代衰减图已保存为 {image_filename}")
        plt.close()

    @staticmethod
    def plot_spline_attenuation(engine, spline_field, x_new, y_new, language='en', data_filename=None, quantified_data=None):
        fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

        tx_pos, rx_pos = Visualizer._get_sensor_positions(engine.data)
        x_coords = np.linspace(*Config.COORD_RANGE_X, len(x_new))
        y_coords = np.linspace(*Config.COORD_RANGE_Y, len(y_new))

        data_key = Visualizer._infer_data_key(data_filename)
        Visualizer._plot_attenuation_field(
            ax,
            spline_field,
            tx_pos,
            rx_pos,
            x_coords,
            y_coords,
            language,
            'spline_attenuation',
            data_key,
            quantified_data
        )

        plt.tight_layout()

        image_filename = Visualizer.generate_image_path(
            'spline_attenuation',
            data_filename,
            language
        )
        plt.savefig(image_filename, bbox_inches='tight', dpi=Config.PLOT_DPI)
        print(f"插值衰减图已保存为 {image_filename}")
        plt.close()

    @staticmethod
    def plot_results(data, engine, spline_field, x_new, y_new, data_filename=None, quantified_data=None):
        if Config.GENERATE_IMAGES['straight_rays']:
            Visualizer.plot_straight_rays(data, data_filename)

        if Config.GENERATE_IMAGES['actual_rays']:
            Visualizer.plot_actual_rays(data, engine, data_filename)

        if Config.GENERATE_IMAGES['iterated_attenuation_en']:
            Visualizer.plot_iterated_attenuation(engine, 'en', data_filename, quantified_data)

        if Config.GENERATE_IMAGES['iterated_attenuation_zh']:
            Visualizer.plot_iterated_attenuation(engine, 'zh', data_filename, quantified_data)

        if Config.GENERATE_IMAGES['spline_attenuation_en']:
            Visualizer.plot_spline_attenuation(engine, spline_field, x_new, y_new, 'en', data_filename, quantified_data)

        if Config.GENERATE_IMAGES['spline_attenuation_zh']:
            Visualizer.plot_spline_attenuation(engine, spline_field, x_new, y_new, 'zh', data_filename, quantified_data)

        if Config.GENERATE_IMAGES.get('combined_zh', False):
            Visualizer._plot_combined_attenuation(engine, spline_field, x_new, y_new, 'zh', data_filename)

        if Config.GENERATE_IMAGES.get('combined_en', False):
            Visualizer._plot_combined_attenuation(engine, spline_field, x_new, y_new, 'en', data_filename)

    @staticmethod
    def _plot_combined_attenuation(engine, spline_field, x_new, y_new, language, data_filename):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIG_SIZE_COMBINED)

        tx_pos, rx_pos = Visualizer._get_sensor_positions(engine.data)

        x_coords = np.linspace(*Config.COORD_RANGE_X, Config.GRID_NX)
        y_coords = np.linspace(*Config.COORD_RANGE_Y, Config.GRID_NY)

        Visualizer._plot_attenuation_field(
            ax1,
            engine.reconstructor.attenuation_field,
            tx_pos,
            rx_pos,
            x_coords,
            y_coords,
            language,
            'iterated_attenuation'
        )

        x_new_coords = np.linspace(*Config.COORD_RANGE_X, len(x_new))
        y_new_coords = np.linspace(*Config.COORD_RANGE_Y, len(y_new))

        Visualizer._plot_attenuation_field(
            ax2,
            spline_field,
            tx_pos,
            rx_pos,
            x_new_coords,
            y_new_coords,
            language,
            'spline_attenuation'
        )

        plt.tight_layout()

        image_filename = Visualizer.generate_image_path(
            f'combined_{language}',
            data_filename,
            language
        )
        plt.savefig(image_filename, bbox_inches='tight', dpi=Config.PLOT_DPI)
        print(f"衰减场组合图 ({language}) 已保存为 {image_filename}")
        plt.close()
