"""数据加载模块"""
import numpy as np
import pandas as pd
import time
from config import Config


class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_and_validate(file_path):
        """加载并验证能量数据"""
        print(f"正在从 {file_path} 加载数据...")
        start_time = time.time()
        df = pd.read_excel(file_path)
        print(f"数据加载完成，耗时 {time.time() - start_time:.2f} 秒")
        print(f"Excel文件中的列名: {df.columns.tolist()}")

        df = DataLoader._rename_columns(df)
        df = DataLoader._process_energy_data(df)
        DataLoader._validate_columns(df)
        df = DataLoader._filter_by_range(df)
        df = DataLoader._convert_types(df)
        
        print(f"坐标范围过滤后的数据量: {len(df)}")
        return df

    @staticmethod
    def _rename_columns(df):
        """重命名列"""
        for old_name, new_name in Config.COLUMN_MAPPING.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        return df

    @staticmethod
    def _denoise_attenuation_integral(values):
        """对路径衰减积分值进行稳健去噪，抑制由能量微小波动产生的雾状背景。"""
        settings = getattr(Config, 'PATH_ATTENUATION_DENOISE', {'enabled': False})
        values = np.asarray(values, dtype=np.float64).copy()

        if not settings.get('enabled', False):
            return values, 0.0

        positive = values[values > 0]
        if positive.size == 0:
            return values, 0.0

        method = settings.get('method', 'mad')
        if method == 'fixed':
            thr = float(settings.get('fixed_threshold', 0.0))
        else:
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            sigma = 1.4826 * mad if mad > 0 else 0.0
            thr = med + float(settings.get('mad_k', 1.0)) * sigma

        thr = max(float(settings.get('min_threshold', 0.0)), thr)
        max_thr = settings.get('max_threshold', None)
        if max_thr is not None:
            thr = min(float(max_thr), thr)

        mode = settings.get('mode', 'hard')
        if mode == 'soft':
            denoised = np.maximum(values - thr, 0.0)
        else:
            denoised = values.copy()
            denoised[denoised < thr] = 0.0

        return denoised, thr

    @staticmethod
    def load_paired_data(healthy_file, damage_file):
        """加载配对的无损和损伤数据"""
        print(f"正在从 {healthy_file} 和 {damage_file} 加载配对数据...")
        start_time = time.time()
        healthy_df = pd.read_excel(healthy_file)
        damage_df = pd.read_excel(damage_file)
        print(f"数据加载完成，耗时 {time.time() - start_time:.2f} 秒")
        
        healthy_df = DataLoader._rename_columns(healthy_df)
        damage_df = DataLoader._rename_columns(damage_df)
        
        DataLoader._validate_columns(healthy_df)
        DataLoader._validate_columns(damage_df)
        
        h_energy_col = 'Energy_mean' if 'Energy_mean' in healthy_df.columns else 'Energy'
        d_energy_col = 'Energy_mean' if 'Energy_mean' in damage_df.columns else 'Energy'
        
        if h_energy_col not in healthy_df.columns or d_energy_col not in damage_df.columns:
            raise ValueError("缺少能量列 (Energy_mean 或 Energy)")
            
        df = pd.merge(
            healthy_df,
            damage_df,
            on=['Tx_X', 'Tx_Y', 'Rx_X', 'Rx_Y'],
            suffixes=('_h', '_d')
        )
        
        E_h = df[h_energy_col + '_h'].values.copy()
        E_d = df[d_energy_col + '_d'].values.copy()
        
        E_h = np.maximum(E_h, Config.ENERGY_MIN_VALUE)
        E_d = np.maximum(E_d, Config.ENERGY_MIN_VALUE)
        
        b_i_raw = np.log(E_h / E_d)
        b_i_raw = np.maximum(b_i_raw, 0.0)

        b_i, denoise_thr = DataLoader._denoise_attenuation_integral(b_i_raw)
        
        df['Travel_Time_us'] = b_i
        df['Time_Difference_us'] = b_i
        df['Raw_Attenuation_Integral'] = b_i_raw
        df['Denoise_Threshold'] = denoise_thr
        
        print(f"原始衰减积分值 b_i 范围: [{b_i_raw.min():.4f}, {b_i_raw.max():.4f}]")
        print(f"去噪阈值: {denoise_thr:.4f}; 去噪后范围: [{b_i.min():.4f}, {b_i.max():.4f}]")
        print(f"去噪后平均值: {b_i.mean():.4f}, 中位数: {np.median(b_i):.4f}")
        
        df = DataLoader._filter_by_range(df)
        df = DataLoader._convert_types(df)
        
        print(f"有效路径数: {len(df)}, 非零衰减路径数: {np.sum(df['Travel_Time_us'] > 0)}")
        return df

    @staticmethod
    def _process_energy_data(df):
        """处理能量数据：根据指数衰减模型计算衰减积分值"""
        if 'Energy' not in df.columns:
            raise ValueError("能量数据中缺少 'Energy' 列")

        energy_values = df['Energy'].values.copy()
        energy_values = np.maximum(energy_values, Config.ENERGY_MIN_VALUE)

        e_in = getattr(Config, 'ENERGY_MAX_VALUE', 0.000735)
        
        attenuation_integral_raw = np.log(e_in / energy_values)
        attenuation_integral_raw = np.maximum(attenuation_integral_raw, 0.0)

        attenuation_integral, denoise_thr = DataLoader._denoise_attenuation_integral(
            attenuation_integral_raw
        )
        
        df['Travel_Time_us'] = attenuation_integral
        df['Raw_Attenuation_Integral'] = attenuation_integral_raw
        df['Denoise_Threshold'] = denoise_thr
        
        print(f"能量范围: [{energy_values.min():.6e}, {energy_values.max():.6e}]")
        print(f"基准能量 (E_in): {e_in:.6e}")
        print(f"原始衰减积分值范围: [{attenuation_integral_raw.min():.4f}, {attenuation_integral_raw.max():.4f}]")
        print(f"去噪阈值: {denoise_thr:.4f}; 去噪后范围: [{attenuation_integral.min():.4f}, {attenuation_integral.max():.4f}]")
        
        df['Time_Difference_us'] = df['Travel_Time_us']
        return df

    @staticmethod
    def _validate_columns(df):
        """验证必需列是否存在"""
        required_cols = ['Tx_X', 'Tx_Y', 'Rx_X', 'Rx_Y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"数据中缺少必要的列: {col}")

    @staticmethod
    def _filter_by_range(df):
        """按坐标范围过滤数据"""
        mask = (
            (df['Tx_X'] >= Config.COORD_RANGE_X[0]) &
            (df['Tx_X'] <= Config.COORD_RANGE_X[1]) &
            (df['Tx_Y'] >= Config.COORD_RANGE_Y[0]) &
            (df['Tx_Y'] <= Config.COORD_RANGE_Y[1]) &
            (df['Rx_X'] >= Config.COORD_RANGE_X[0]) &
            (df['Rx_X'] <= Config.COORD_RANGE_X[1]) &
            (df['Rx_Y'] >= Config.COORD_RANGE_Y[0]) &
            (df['Rx_Y'] <= Config.COORD_RANGE_Y[1])
        )
        return df[mask].copy()

    @staticmethod
    def _convert_types(df):
        """类型转换"""
        for col in ['Tx_X', 'Tx_Y', 'Rx_X', 'Rx_Y', 'Travel_Time_us']:
            df[col] = df[col].astype(np.float64)
        return df