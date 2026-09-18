"""分别用普通 SIRT 与 TV-SIRT 生成 healthy 和所有 damage_A* 的成像图/量化结果。

输出会自动按方法分开：
    图片生成/SIRT_不含TV/...
    图片生成/TV-SIRT_含TV/...
    量化结果/SIRT_不含TV/...
    量化结果/TV-SIRT_含TV/...

运行方式：
    python run_two_methods_all_data.py
"""
import os
from config import Config
from main import process_energy_data


def main():
    base_path = Config.BASE_PATH
    healthy_file = os.path.join(base_path, Config.NON_DAMAGE_DATA_FILE)
    damage_files = [
        os.path.join(base_path, f"damage_A{i}.xlsx")
        for i in range(1, 5)
    ]

    methods = getattr(Config, 'HEALTHY_COMPARE_METHODS', ['SIRT', 'SIRT_TV'])
    for method in methods:
        print('\n' + '=' * 80)
        print(f'开始方法对比输出：{method}')
        print('=' * 80)

        process_energy_data(
            data_filename=healthy_file,
            state='无损',
            healthy_filename=None,
            reconstruction_method=method,
            output_suffix=method
        )

        for damage_file in damage_files:
            if not os.path.exists(damage_file):
                print(f'跳过不存在的数据文件: {damage_file}')
                continue
            process_energy_data(
                data_filename=damage_file,
                state='损伤',
                healthy_filename=healthy_file,
                reconstruction_method=method,
                output_suffix=method
            )


if __name__ == '__main__':
    main()
