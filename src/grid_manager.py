"""网格管理模块"""
import numpy as np
from config import Config


class GridManager:
    """网格管理器：创建计算网格并提供插值功能"""
    
    def __init__(self):
        self.grid_x = np.linspace(Config.COORD_RANGE_X[0], Config.COORD_RANGE_X[1], Config.GRID_NX, dtype=np.float64)
        self.grid_y = np.linspace(Config.COORD_RANGE_Y[0], Config.COORD_RANGE_Y[1], Config.GRID_NY, dtype=np.float64)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)

    def get_grid_cell_index(self, x, y):
        """获取坐标对应的网格索引"""
        i = np.clip(np.searchsorted(self.grid_x, x) - 1, 0, Config.GRID_NX - 1)
        j = np.clip(np.searchsorted(self.grid_y, y) - 1, 0, Config.GRID_NY - 1)
        return i, j

    def get_velocity_at_point(self, x, y, velocity_field):
        """获取某点的声速（双线性插值）"""
        i, j = self.get_grid_cell_index(x, y)
        i = min(max(i, 0), Config.GRID_NX - 2)
        j = min(max(j, 0), Config.GRID_NY - 2)

        # 获取网格单元四个角点的坐标和速度值
        x1, x2 = self.grid_x[i], self.grid_x[i + 1]
        y1, y2 = self.grid_y[j], self.grid_y[j + 1]
        v11 = velocity_field[j, i]  # 左下角
        v12 = velocity_field[j, i + 1]  # 右下角
        v21 = velocity_field[j + 1, i]  # 左上角
        v22 = velocity_field[j + 1, i + 1]  # 右上角

        # 计算归一化偏移量
        dx = (x - x1) / (x2 - x1)
        dy = (y - y1) / (y2 - y1)

        # 双线性插值
        return (1 - dx) * (1 - dy) * v11 + dx * (1 - dy) * v12 + (1 - dx) * dy * v21 + dx * dy * v22
