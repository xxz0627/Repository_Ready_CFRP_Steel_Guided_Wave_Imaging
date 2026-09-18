"""主程序模块"""
import os
import time
import traceback
import numpy as np
from tqdm import tqdm

from config import Config
from data_loader import DataLoader
from grid_manager import GridManager
from ray_tracer import RayTracer
from reconstructor import Reconstructor
from sensor_manager import SensorManager
from visualizer import Visualizer
from quantifier import quantify_defects_from_data, quantify_healthy_from_data


class TomographyEngine:
    """层析成像引擎：协调各模块完成成像流程"""

    def __init__(self, data, grid_manager, ray_tracer, reconstructor, sensor_manager, reconstruction_method=None):
        """初始化成像引擎"""
        self.data = data
        self.grid_manager = grid_manager
        self.ray_tracer = ray_tracer
        self.reconstructor = reconstructor
        self.sensor_manager = sensor_manager
        self.ray_paths = {}
        self.reconstruction_method = reconstruction_method

    def reconstruct(self):
        """执行完整的重建流程"""
        # 步骤1：射线追踪（在当前模型中，我们使用直线近似）
        print("使用直线近似射线路径...")
        # self._trace_all_rays() # 如果需要弯曲射线追踪，可以启用此部分

        # 步骤2：迭代重建衰减场
        print("开始迭代重建衰减场...")
        start_time = time.time()
        self.reconstructor.initialize_attenuation_field()
        attenuation_field = self.reconstructor.reconstruct(self.data, self.ray_paths, method=self.reconstruction_method)
        print(f"迭代重建完成，耗时 {time.time() - start_time:.2f} 秒")

        # 步骤3：高级空间插值
        print("开始基于物理模型的高级空间插值...")
        start_time = time.time()
        attenuation_interp, x_new, y_new = self.reconstructor.advanced_interpolation()
        print(f"插值完成，耗时 {time.time() - start_time:.2f} 秒")

        return attenuation_interp, x_new, y_new

    def get_ray_path(self, index):
        """获取指定索引的射线路径"""
        return self.ray_paths.get(index)


def process_energy_data(data_filename, state, healthy_filename=None, reconstruction_method=None, output_suffix=None):
    """处理能量数据并进行成像

    reconstruction_method 可选 'SIRT' 或 'SIRT_TV'。运行时会设置
    Config.CURRENT_OUTPUT_METHOD，使不同方法的成像图/量化结果进入独立文件夹。
    """
    previous_output_method = getattr(Config, 'CURRENT_OUTPUT_METHOD', None)
    method_for_output = Config.normalize_method_name(reconstruction_method or Config.DEFAULT_RECONSTRUCTION_METHOD)
    Config.CURRENT_OUTPUT_METHOD = method_for_output
    try:
        Visualizer.initialize()

        # 加载数据
        print(f"正在加载{state}状态数据...")
        if Config.ENERGY_BASELINE_MODE == "paired_healthy_damage" and healthy_filename:
            data = DataLoader.load_paired_data(healthy_filename, data_filename)
        else:
            data = DataLoader.load_and_validate(data_filename)
            
        if len(data) == 0:
            raise ValueError("加载的数据为空")

        # 初始化各模块
        grid_manager = GridManager()
        ray_tracer = RayTracer(grid_manager)
        reconstructor = Reconstructor(grid_manager)
        sensor_manager = SensorManager(data)

        # 打印诊断信息
        print("-" * 30)
        print(f"诊断信息:")
        print(f"总路径数/有效路径数: {len(data)}")
        print(f"网格大小: {Config.GRID_NX} × {Config.GRID_NY}")
        print("-" * 30)

        # 创建成像引擎
        engine = TomographyEngine(data, grid_manager, ray_tracer, reconstructor, sensor_manager, reconstruction_method=reconstruction_method)

        # 执行重建
        print(f"开始{state}状态层析成像重建...")
        spline_field, x_new, y_new = engine.reconstruct()

        # 生成文件名基础
        output_filename_base = os.path.splitext(os.path.basename(data_filename))[0]
        if output_suffix:
            output_filename_base = f"{output_filename_base}_{output_suffix}"

        # 如果启用了量化分析，则先执行量化
        quantified_data = None
        if Config.QUANTIFICATION.get('enabled', False):
            print(f"开始对 {state} 状态结果进行量化分析...")
            data_key = None
            for key in ['A4', 'A3', 'A2', 'A1']:
                if key in data_filename:
                    data_key = key
                    break

            if data_key:
                quantified_data = quantify_defects_from_data(
                    attenuation_field=engine.reconstructor.attenuation_field,
                    x_coords=engine.grid_manager.grid_x,
                    y_coords=engine.grid_manager.grid_y,
                    state=state,
                    base_name=output_filename_base,
                    data_key=data_key
                )
            else:
                quantified_data = quantify_healthy_from_data(
                    attenuation_field=engine.reconstructor.attenuation_field,
                    x_coords=engine.grid_manager.grid_x,
                    y_coords=engine.grid_manager.grid_y,
                    state=state,
                    base_name=output_filename_base
                )

        # 生成可视化结果（在量化之后，可以绘制mask叠加图）
        print(f"生成{state}状态可视化结果...")
        Visualizer.plot_results(data, engine, spline_field, x_new, y_new, output_filename_base, quantified_data)

        print(f"{state}状态处理完成！")

    except Exception as e:
        print(f"{state}状态处理错误: {str(e)}")
        traceback.print_exc()
    finally:
        Config.CURRENT_OUTPUT_METHOD = previous_output_method


def main():
    """主函数"""
    try:
        base_path = Config.BASE_PATH
        non_damage_data = os.path.join(base_path, Config.NON_DAMAGE_DATA_FILE)

        states_to_process = []
        if Config.PROCESS_STATE == 'both' or Config.PROCESS_STATE == 'damage':
            damage_data = os.path.join(base_path, Config.DAMAGE_DATA_FILE)
            states_to_process.append(('损伤', damage_data, non_damage_data))

        if Config.PROCESS_STATE == 'both' or Config.PROCESS_STATE == 'non_damage':
            states_to_process.append(('无损', non_damage_data, None))

        if not states_to_process:
            print(f"错误: 无效的 PROCESS_STATE 值: {Config.PROCESS_STATE}")
            return
            
        # 尝试读取 metadata.json 打印信息
        metadata_path = os.path.join(base_path, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print("\n--- 读取到 metadata.json ---")
                    for k, v in metadata.items():
                        print(f"{k}: {v}")
                    print("----------------------------\n")
            except Exception as e:
                print(f"读取 metadata.json 失败: {e}")
                
        for state, file_path, healthy_path in states_to_process:
            process_energy_data(file_path, state, healthy_path)

        print("状态处理完成！")

    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
