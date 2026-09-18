"""射线追踪模块：基于费马原理的最小走时路径优化"""
import numpy as np
from scipy.interpolate import interp1d
from config import Config


class RayTracer:
    """射线追踪器：基于费马原理计算最优射线路径"""

    def __init__(self, grid_manager):
        """初始化射线追踪器
        
        Args:
            grid_manager: 网格管理器，提供速度场插值功能
        """
        self.grid_manager = grid_manager

    def trace_ray(self, tx, ty, rx, ry, velocity_field, energy=None, energy_gradient=None, is_neighbor_path=False):
        """追踪单条射线路径
        
        Args:
            tx, ty: 发射器坐标(mm)
            rx, ry: 接收器坐标(mm)
            velocity_field: 速度场(归一化值)
            energy: 能量差异值(可选)
            energy_gradient: 能量梯度(可选)
            is_neighbor_path: 是否为相邻传感器路径
            
        Returns:
            path_x, path_y: 射线路径的x和y坐标数组
        """
        num_points = Config.RAY_TRACING_STEPS
        path_x = np.linspace(tx, rx, num_points, dtype=np.float64)
        path_y = np.linspace(ty, ry, num_points, dtype=np.float64)

        if energy is None:
            path_x, path_y = self._optimize_path(path_x, path_y, velocity_field)
        else:
            path_x, path_y = self._bend_path(path_x, path_y, energy, energy_gradient, is_neighbor_path)

        return path_x, path_y

    def _optimize_path(self, path_x, path_y, velocity_field):
        """基于费马原理优化路径，使走时最小化
        
        使用梯度下降法迭代优化路径点位置，
        使射线的总走时达到最小（费马原理）
        
        Args:
            path_x, path_y: 初始直线路径坐标
            velocity_field: 速度场
            
        Returns:
            优化后的路径坐标
        """
        max_iterations = Config.PATH_OPTIMIZATION_MAX_ITER
        learning_rate = Config.PATH_OPTIMIZATION_LR
        
        for iteration in range(max_iterations):
            travel_time = self._calculate_travel_time(path_x, path_y, velocity_field)
            gradients_x, gradients_y = self._calculate_path_gradients(path_x, path_y, velocity_field)
            path_x[1:-1] -= learning_rate * gradients_x[1:-1]
            path_y[1:-1] -= learning_rate * gradients_y[1:-1]

        return path_x, path_y

    def _bend_path(self, path_x, path_y, energy, energy_gradient, is_neighbor_path):
        """根据能量差异弯曲射线路径
        
        当有能量数据时，根据能量差异调整路径弯曲程度
        低能量区域（缺陷）会导致射线绕行
        
        Args:
            path_x, path_y: 初始路径坐标
            energy: 能量值
            energy_gradient: 能量梯度
            is_neighbor_path: 是否为相邻传感器路径
            
        Returns:
            弯曲后的路径坐标
        """
        if not is_neighbor_path:
            curvature_factor = abs(energy) * Config.RAY_CURVATURE_FACTOR
            direction = 1 if (energy_gradient or energy) > 0 else -1

            mid_index = len(path_x) // 2
            tx, ty = path_x[0], path_y[0]
            rx, ry = path_x[-1], path_y[-1]

            dx = rx - tx
            dy = ry - ty
            norm = np.sqrt(dx ** 2 + dy ** 2) + Config.SMALL_VALUE
            dir_x = dx / norm
            dir_y = dy / norm

            perp_x = -dir_y * direction
            perp_y = dir_x * direction

            if energy_gradient:
                gradient_factor = max(0.5, min(2.0, 
                    abs(energy_gradient) / (abs(energy) + Config.SMALL_VALUE)))
            else:
                gradient_factor = Config.GRADIENT_FACTOR_DEFAULT
            
            offset = curvature_factor * gradient_factor * Config.OFFSET_SCALE_FACTOR

            path_x[mid_index] += perp_x * offset
            path_y[mid_index] += perp_y * offset

            try:
                new_path_x = np.array([path_x[0], path_x[mid_index], path_x[-1]])
                new_path_y = np.array([path_y[0], path_y[mid_index], path_y[-1]])

                if len(np.unique(new_path_x)) >= 2 and len(np.unique(new_path_y)) >= 2:
                    t = np.linspace(0, 1, len(path_x))
                    fx = interp1d([0, 0.5, 1], new_path_x, kind='quadratic')
                    fy = interp1d([0, 0.5, 1], new_path_y, kind='quadratic')
                    path_x = fx(t)
                    path_y = fy(t)
            except Exception:
                pass

        return path_x, path_y

    def _calculate_travel_time(self, path_x, path_y, velocity_field):
        """计算射线的总走时
        
        将路径离散为微段，累加每个微段的走时
        走时 = 路径长度 / 速度
        
        Args:
            path_x, path_y: 路径坐标
            velocity_field: 速度场
            
        Returns:
            总走时(微秒)
        """
        travel_time = 0.0
        
        for i in range(len(path_x) - 1):
            x1, y1 = path_x[i], path_y[i]
            x2, y2 = path_x[i + 1], path_y[i + 1]

            ds = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1e-3

            v1 = self.grid_manager.get_velocity_at_point(x1, y1, velocity_field)
            v2 = self.grid_manager.get_velocity_at_point(x2, y2, velocity_field)
            
            v_avg = (v1 + v2) / 2.0

            travel_time += (ds / v_avg) * 1e6

        return travel_time

    def _calculate_path_gradients(self, path_x, path_y, velocity_field):
        """计算路径上各点的走时梯度
        
        基于变分原理推导的梯度公式，用于梯度下降优化
        梯度方向指向走时增加最快的方向
        
        Args:
            path_x, path_y: 路径坐标
            velocity_field: 速度场
            
        Returns:
            gradients_x, gradients_y: x和y方向的梯度
        """
        num_points = len(path_x)
        gradients_x = np.zeros(num_points, dtype=np.float64)
        gradients_y = np.zeros(num_points, dtype=np.float64)

        for i in range(1, num_points - 1):
            x_prev, y_prev = path_x[i - 1], path_y[i - 1]
            x_curr, y_curr = path_x[i], path_y[i]
            x_next, y_next = path_x[i + 1], path_y[i + 1]

            len_prev = np.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2) + Config.SMALL_VALUE
            len_next = np.sqrt((x_next - x_curr) ** 2 + (y_next - y_curr) ** 2) + Config.SMALL_VALUE

            dir_prev_x = (x_curr - x_prev) / len_prev
            dir_prev_y = (y_curr - y_prev) / len_prev
            dir_next_x = (x_next - x_curr) / len_next
            dir_next_y = (y_next - y_curr) / len_next

            v_curr = self.grid_manager.get_velocity_at_point(x_curr, y_curr, velocity_field)
            
            delta = Config.RAY_DIFFERENTIAL_STEP
            v_curr_x = (self.grid_manager.get_velocity_at_point(x_curr + delta, y_curr, velocity_field) - v_curr) / delta
            v_curr_y = (self.grid_manager.get_velocity_at_point(x_curr, y_curr + delta, velocity_field) - v_curr) / delta

            term1_x = dir_next_x / v_curr
            term1_y = dir_next_y / v_curr
            
            term2_x = dir_prev_x / v_curr
            term2_y = dir_prev_y / v_curr
            
            term3_x = (v_curr_x / (v_curr ** 2)) * (len_prev + len_next) / 2
            term3_y = (v_curr_y / (v_curr ** 2)) * (len_prev + len_next) / 2

            gradients_x[i] = term1_x - term2_x + term3_x
            gradients_y[i] = term1_y - term2_y + term3_y

        return gradients_x, gradients_y
