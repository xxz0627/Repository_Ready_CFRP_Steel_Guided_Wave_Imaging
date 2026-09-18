"""图像重建模块"""
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
try:
    from skimage.restoration import denoise_tv_chambolle
except Exception:
    denoise_tv_chambolle = None
from tqdm import tqdm
from config import Config


class Reconstructor:
    """重建器：基于能量衰减模型，使用SIRT算法重建衰减系数场"""

    def __init__(self, grid_manager):
        """初始化重建器"""
        self.grid_manager = grid_manager
        self.attenuation_field = None

    def initialize_attenuation_field(self):
        """初始化衰减系数场"""
        initial_alpha = Config.INITIAL_ATTENUATION
        
        print(f"初始化衰减场，初始值: {initial_alpha:.4f}")
        
        self.attenuation_field = np.full(
            (Config.GRID_NY, Config.GRID_NX),
            initial_alpha,
            dtype=np.float64
        )
        return self.attenuation_field.copy()

    def reconstruct(self, data, ray_paths, method=None):
        """执行重建算法入口"""
        if method is None:
            method = getattr(Config, 'DEFAULT_RECONSTRUCTION_METHOD', 'SIRT_TV')
        
        method = str(method).upper()
        if method in ('SIRT_TV', 'TV_SIRT', 'TV-SIRT'):
            if denoise_tv_chambolle is None:
                raise ImportError('当前环境缺少 scikit-image，无法运行 TV-SIRT；普通 SIRT 不需要该依赖。')
            print("使用全变分正则化SIRT (TV-SIRT) 算法重建衰减场...")
            return self._sirt_tv_reconstruct(data, ray_paths)
        elif method == 'SIRT':
            print("使用普通SIRT算法重建衰减场（不含TV正则）...")
            return self._sirt_reconstruct(data, ray_paths)
        else:
            raise ValueError(f'未知重建方法: {method}. 可选: SIRT, SIRT_TV')

    def _postprocess_attenuation_field(self):
        """重建后背景扣除，降低低幅雾状伪影，同时保留缺陷峰值。"""
        if not getattr(Config, 'POST_RECON_BACKGROUND_SUPPRESSION', False):
            return

        field = np.asarray(self.attenuation_field, dtype=np.float64)
        finite_values = field[np.isfinite(field)]
        if finite_values.size == 0:
            return

        p = float(getattr(Config, 'BACKGROUND_SUPPRESSION_PERCENTILE', 60))
        strength = float(getattr(Config, 'BACKGROUND_SUPPRESSION_STRENGTH', 1.0))

        background = np.percentile(finite_values, p) * strength
        min_alpha, max_alpha = Config.ATTENUATION_RANGE

        self.attenuation_field = np.clip(field - background, min_alpha, max_alpha)
        print(f"  - 背景扣除: percentile={p:.1f}, background={background:.4f}")

    def _sirt_tv_reconstruct(self, data, ray_paths):
        """执行带有全变分(TV)正则化的SIRT迭代"""
        A = self._build_ray_matrix(data, ray_paths)

        row_sums = A.sum(axis=1).A.ravel()
        row_weights = diags(1.0 / (row_sums + Config.SMALL_VALUE), dtype=np.float64)
        
        col_sums = A.sum(axis=0).A.ravel()
        with np.errstate(divide='ignore'):
            col_weights_data = 1.0 / col_sums
        col_weights_data[col_sums < 1e-6] = 0.0
        col_weights = diags(col_weights_data, dtype=np.float64)

        min_alpha, max_alpha = Config.ATTENUATION_RANGE

        t_prev = 1.0
        x_prev = self.attenuation_field.copy()
        use_fista = getattr(Config, 'USE_FISTA_ACCELERATION', False)

        observed_integral = data['Travel_Time_us'].values

        for iter_num in tqdm(range(Config.SIRT_ITERATIONS)):
            current_attenuation = self.attenuation_field.ravel()
            projected_integral = A.dot(current_attenuation)
            
            residual = observed_integral - projected_integral
            
            weighted_residual = row_weights.dot(residual)
            grad_update = col_weights.dot(
                A.T.dot(weighted_residual)
            ).reshape(Config.GRID_NY, Config.GRID_NX)
            
            learning_rate = Config.RECONSTRUCTION_LEARNING_RATE_INIT * (
                1.0 - iter_num / Config.SIRT_ITERATIONS
            )
            
            x_new = self.attenuation_field + learning_rate * grad_update
            x_new = np.clip(x_new, min_alpha, max_alpha)

            if hasattr(Config, 'TV_WEIGHT') and Config.TV_WEIGHT > 0:
                norm_range = max_alpha - min_alpha
                if norm_range > 0:
                    x_norm = (x_new - min_alpha) / norm_range
                    x_norm = denoise_tv_chambolle(
                        x_norm,
                        weight=Config.TV_WEIGHT,
                        max_num_iter=20
                    )
                    x_new = x_norm * norm_range + min_alpha

            if use_fista:
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
                momentum = (t_prev - 1.0) / t_new
                self.attenuation_field = x_new + momentum * (x_new - x_prev)
                t_prev = t_new
            else:
                self.attenuation_field = x_new
            
            self.attenuation_field = np.clip(self.attenuation_field, min_alpha, max_alpha)
            x_prev = x_new.copy()

        if Config.RECONSTRUCTION_GAUSSIAN_SIGMA > 0:
            self.attenuation_field = gaussian_filter(
                self.attenuation_field,
                sigma=Config.RECONSTRUCTION_GAUSSIAN_SIGMA
            )

        self._postprocess_attenuation_field()
        
        print(f"\n重建完成诊断:")
        print(
            f"  - 衰减场 α [1/m] 统计: "
            f"min={self.attenuation_field.min():.4f}, "
            f"max={self.attenuation_field.max():.4f}, "
            f"mean={self.attenuation_field.mean():.4f}"
        )
        print(f"  - 95百分位: {np.percentile(self.attenuation_field, 95):.4f}, 99百分位: {np.percentile(self.attenuation_field, 99):.4f}")
        print(f"  - 观测残差 b_i 统计: min={observed_integral.min():.4f}, max={observed_integral.max():.4f}, median={np.median(observed_integral):.4f}")
        print(f"  - 非零观测路径数量: {np.sum(observed_integral > 0)}")
        
        return self.attenuation_field

    def _sirt_reconstruct(self, data, ray_paths):
        """执行传统的SIRT迭代，重建衰减系数场"""
        A = self._build_ray_matrix(data, ray_paths)

        row_sums = A.sum(axis=1).A.ravel()
        row_weights = diags(1.0 / (row_sums + Config.SMALL_VALUE), dtype=np.float64)
        
        col_sums = A.sum(axis=0).A.ravel()
        with np.errstate(divide='ignore'):
            col_weights_data = 1.0 / col_sums
        col_weights_data[col_sums < 1e-6] = 0.0
        col_weights = diags(col_weights_data, dtype=np.float64)

        min_alpha, max_alpha = Config.ATTENUATION_RANGE
        observed_integral = data['Travel_Time_us'].values

        for iter_num in tqdm(range(Config.SIRT_ITERATIONS)):
            current_attenuation = self.attenuation_field.ravel()
            projected_integral = A.dot(current_attenuation)
            
            residual = observed_integral - projected_integral
            
            weighted_residual = row_weights.dot(residual)
            update = col_weights.dot(
                A.T.dot(weighted_residual)
            ).reshape(Config.GRID_NY, Config.GRID_NX)
            
            learning_rate = Config.RECONSTRUCTION_LEARNING_RATE_INIT * (
                1.0 - iter_num / Config.SIRT_ITERATIONS
            )

            self.attenuation_field += learning_rate * update
            
            if Config.ITERATIVE_SMOOTHING_SIGMA > 0:
                self.attenuation_field = gaussian_filter(
                    self.attenuation_field,
                    sigma=Config.ITERATIVE_SMOOTHING_SIGMA
                )
            
            self.attenuation_field = np.clip(self.attenuation_field, min_alpha, max_alpha)

        if Config.RECONSTRUCTION_GAUSSIAN_SIGMA > 0:
            self.attenuation_field = gaussian_filter(
                self.attenuation_field,
                sigma=Config.RECONSTRUCTION_GAUSSIAN_SIGMA
            )

        self._postprocess_attenuation_field()
        
        return self.attenuation_field

    def _build_ray_matrix(self, data, ray_paths):
        """构建射线矩阵"""
        A = lil_matrix((len(data), Config.GRID_NX * Config.GRID_NY), dtype=np.float64)

        for i, (_, row) in tqdm(enumerate(data.iterrows()), total=len(data)):
            if i in ray_paths:
                path_x, path_y = ray_paths[i]
            else:
                path_x = np.linspace(
                    row['Tx_X'],
                    row['Rx_X'],
                    Config.RAY_TRACING_STEPS,
                    dtype=np.float64
                )
                path_y = np.linspace(
                    row['Tx_Y'],
                    row['Rx_Y'],
                    Config.RAY_TRACING_STEPS,
                    dtype=np.float64
                )

            grid_lengths = self._calculate_path_length_in_grid(path_x, path_y)

            for j in range(Config.GRID_NY):
                for k in range(Config.GRID_NX):
                    index = j * Config.GRID_NX + k
                    A[i, index] = grid_lengths[j, k]

        return A.tocsr()

    def _calculate_path_length_in_grid(self, path_x, path_y):
        """计算射线在每个网格中的长度（中点落格法）"""
        gx = self.grid_manager.grid_x
        gy = self.grid_manager.grid_y

        cell_w = float(np.mean(np.diff(gx)))
        cell_h = float(np.mean(np.diff(gy)))
        min_cell = min(cell_w, cell_h)

        grid_lengths_mm = np.zeros((Config.GRID_NY, Config.GRID_NX), dtype=np.float64)

        for i in range(len(path_x) - 1):
            x1, y1 = float(path_x[i]), float(path_y[i])
            x2, y2 = float(path_x[i + 1]), float(path_y[i + 1])

            seg_len = np.hypot(x2 - x1, y2 - y1)
            n = max(1, int(np.ceil(seg_len / (0.5 * min_cell))))
            
            xs = np.linspace(x1, x2, n + 1)
            ys = np.linspace(y1, y2, n + 1)

            for k in range(n):
                xm = 0.5 * (xs[k] + xs[k + 1])
                ym = 0.5 * (ys[k] + ys[k + 1])
                ds = float(np.hypot(xs[k + 1] - xs[k], ys[k + 1] - ys[k]))

                ii, jj = self.grid_manager.get_grid_cell_index(xm, ym)
                if 0 <= ii < Config.GRID_NX and 0 <= jj < Config.GRID_NY:
                    grid_lengths_mm[jj, ii] += ds

        return grid_lengths_mm * 1e-3

    def advanced_interpolation(self):
        """高级空间插值。默认使用线性插值，减少三次样条在A2小缺陷附近的雾状扩散。"""
        scale = int(getattr(Config, 'INTERPOLATION_SCALE', 4))
        scale = max(scale, 1)

        x_new = np.linspace(*Config.COORD_RANGE_X, Config.GRID_NX * scale, dtype=np.float64)
        y_new = np.linspace(*Config.COORD_RANGE_Y, Config.GRID_NY * scale, dtype=np.float64)

        method = getattr(Config, 'INTERPOLATION_METHOD', 'linear')

        if method == 'spline':
            spline = RectBivariateSpline(
                self.grid_manager.grid_y,
                self.grid_manager.grid_x,
                self.attenuation_field,
                kx=3,
                ky=1
            )
            attenuation_interp = spline(y_new, x_new)
        else:
            interpolator = RegularGridInterpolator(
                (self.grid_manager.grid_y, self.grid_manager.grid_x),
                self.attenuation_field,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            yy, xx = np.meshgrid(y_new, x_new, indexing='ij')
            pts = np.column_stack([yy.ravel(), xx.ravel()])
            attenuation_interp = interpolator(pts).reshape(len(y_new), len(x_new))

        tv_weight = float(getattr(Config, 'INTERPOLATION_TV_WEIGHT', 0.0))
        min_alpha, max_alpha = Config.ATTENUATION_RANGE

        if tv_weight > 0 and max_alpha > min_alpha:
            x_norm = (attenuation_interp - min_alpha) / (max_alpha - min_alpha)
            x_norm = denoise_tv_chambolle(
                x_norm,
                weight=tv_weight,
                max_num_iter=8
            )
            attenuation_interp = x_norm * (max_alpha - min_alpha) + min_alpha

        attenuation_interp = np.clip(attenuation_interp, min_alpha, max_alpha)
        return attenuation_interp, x_new, y_new