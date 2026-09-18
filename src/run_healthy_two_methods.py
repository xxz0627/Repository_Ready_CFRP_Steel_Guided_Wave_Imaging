"""用 healthy.xlsx 无损数据分别生成普通 SIRT 和 TV-SIRT 成像图。

运行方式：
    python run_healthy_two_methods.py

输出位置：
    能量成像/图片生成/插值衰减图中文版/无损/
    能量成像/图片生成/插值衰减图英文版/无损/
文件名会带 healthy_SIRT 或 healthy_SIRT_TV 后缀，便于论文对比。
"""
import os

from config import Config
from main import process_energy_data


def main():
    base_path = Config.BASE_PATH
    healthy_file = os.path.join(base_path, Config.NON_DAMAGE_DATA_FILE)

    methods = getattr(Config, 'HEALTHY_COMPARE_METHODS', ['SIRT', 'SIRT_TV'])
    for method in methods:
        print('\n' + '=' * 80)
        print(f'开始 healthy 无损试件成像：{method}')
        print('=' * 80)
        process_energy_data(
            data_filename=healthy_file,
            state='无损',
            healthy_filename=None,
            reconstruction_method=method,
            output_suffix=method
        )


if __name__ == '__main__':
    main()
