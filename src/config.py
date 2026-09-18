import os
import json


class Config:
    """全局配置类"""
    
    # ==================== 试件尺寸 ====================
    SPECIMEN_WIDTH = 900  # mm
    SPECIMEN_HEIGHT = 100  # mm

    # ==================== 传感器阵列参数 ====================
    RECEIVER_MODE = 16  # 接收模式：8 或 16（8点接收或16点接收）
    TX_START_X = 50  # 发射点起始X坐标
    TX_STEP_X = 50   # 发射点X坐标步长
    TX_END_X = 850   # 发射点结束X坐标
    TX_Y = 50        # 发射点Y坐标（中轴）
    RECEIVER_RADIUS = 50  # 接收点距发射点的半径

    # ==================== 物理参数 ====================
    COORD_RANGE_X = (-10.0, 910.0)  # X坐标范围(mm)
    COORD_RANGE_Y = (-10.0, 110.0)  # Y坐标范围(mm)
    MATERIAL = 'cfrp_energy'  # 材料类型
    
    # ==================== 数据加载参数 ====================
    # 默认使用项目根目录同级的“生成的能量数据”文件夹，避免绝对路径导致换机器后不能成像
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_PARENT_DIR = os.path.dirname(PROJECT_ROOT_DIR)
    PROJECT_REPO_DIR = os.path.dirname(PROJECT_PARENT_DIR)
    PROJECT_FONT_DIR = os.path.join(PROJECT_ROOT_DIR, "fonts")
    PROJECT_ZH_FONT_FILE = os.path.join(PROJECT_FONT_DIR, "NotoSansCJK-Regular.ttc")  # optional local font; not distributed
    BASE_PATH = os.path.join(PROJECT_REPO_DIR, "data", "processed")
    PROCESS_STATE = "non_damage"  # 默认跑无损 healthy；损伤对比可改为 damage 或 both  # 处理状态："damage"或"non_damage"
    ENERGY_BASELINE_MODE = "paired_healthy_damage"  # 或者是 "global_max"

    # 配对能量差值的路径级去噪：
    # 先把很弱的 ln(Ehealthy/Edamage) 噪声路径置零，降低雾状背景伪影
    PATH_ATTENUATION_DENOISE = {
        'enabled': True,
        'method': 'mad',            # mad 或 fixed
        'mode': 'hard',             # hard=低于阈值置零；soft=减去阈值后截断
        'fixed_threshold': 0.03,    # method=fixed 时使用
        'mad_k': 1.0,               # threshold = median + mad_k * 1.4826*MAD
        'max_threshold': 0.04,      # 防止阈值过高漏掉小缺陷
        'min_threshold': 0.0
    }
    
    # 根据RECEIVER_MODE自动选择数据文件
    RECEIVER_MODE_STR = f"{RECEIVER_MODE}点接收"
    DAMAGE_DATA_FILE = "damage_A4.xlsx"
    NON_DAMAGE_DATA_FILE = "healthy.xlsx" 
    
    # 衰减成像参数
    ENERGY_MAX_VALUE = 0.000735
    ENERGY_MIN_VALUE = 1e-10
    INITIAL_ATTENUATION = 0.0
    ATTENUATION_RANGE = (0.0, 10.0)
    
    # ==================== 算法参数 ====================
    # 网格适度加密：A2 中 25 mm 小缺陷至少覆盖 4~5 个网格点，边界不会被 16 行网格过度抹平
    GRID_NX = 160
    GRID_NY = 24
    SIRT_ITERATIONS = 100
    RAY_TRACING_STEPS = 200
    
    # ==================== 数值计算参数 ====================
    SMALL_VALUE = 1e-12
    RAY_DIFFERENTIAL_STEP = 1e-6
    
    # ==================== 重建器参数 ====================
    RECONSTRUCTION_LEARNING_RATE_INIT = 0.3
    RECONSTRUCTION_GAUSSIAN_SIGMA = 0.0
    ITERATIVE_SMOOTHING_SIGMA = 0.0
    TV_WEIGHT = 0.003
    DEFAULT_RECONSTRUCTION_METHOD = 'SIRT'  # 普通版本默认使用 SIRT，不含 TV 正则

    # 重建后背景扣除：去掉低分位背景底噪，减少图中大片雾状伪影
    POST_RECON_BACKGROUND_SUPPRESSION = True
    BACKGROUND_SUPPRESSION_PERCENTILE = 60
    BACKGROUND_SUPPRESSION_STRENGTH = 1.0

    # 插值方式：A2这类小缺陷不要用三次样条过度扩散，线性插值更稳、更少雾状外溢
    INTERPOLATION_METHOD = 'linear'  # linear 或 spline
    INTERPOLATION_SCALE = 7
    INTERPOLATION_TV_WEIGHT = 0.0
    USE_FISTA_ACCELERATION = False
    

    # ==================== 论文对比快捷运行参数 ====================
    # run_healthy_two_methods.py 会读取该列表，分别输出 healthy_SIRT 与 healthy_SIRT_TV 图像。
    HEALTHY_COMPARE_METHODS = ['SIRT', 'SIRT_TV']

    # ==================== 方法对比输出目录 ====================
    # 运行不同重建方法时，main.py 会临时设置 CURRENT_OUTPUT_METHOD，
    # 使成像图和量化结果分别保存到独立文件夹，便于论文对比。
    SEPARATE_OUTPUT_BY_METHOD = True
    CURRENT_OUTPUT_METHOD = None
    METHOD_OUTPUT_DIR_NAMES = {
        'SIRT': 'SIRT_不含TV',
        'SIRT_TV': 'TV-SIRT_含TV'
    }

    @classmethod
    def normalize_method_name(cls, method):
        method = str(method or cls.DEFAULT_RECONSTRUCTION_METHOD).upper().replace('-', '_')
        if method in ('TV_SIRT', 'SIRT_TV'):
            return 'SIRT_TV'
        if method == 'SIRT':
            return 'SIRT'
        return method

    @classmethod
    def get_current_method_output_dir(cls):
        method = cls.normalize_method_name(cls.CURRENT_OUTPUT_METHOD)
        return cls.METHOD_OUTPUT_DIR_NAMES.get(method, method)

    # ==================== 射线追踪优化参数（费马原理）====================
    PATH_OPTIMIZATION_MAX_ITER = 50
    PATH_OPTIMIZATION_LR = 0.1
    RAY_CURVATURE_FACTOR = 1.0
    GRADIENT_FACTOR_DEFAULT = 1.0
    OFFSET_SCALE_FACTOR = 20
    
    # ==================== 可视化参数 ====================
    PLOT_DPI = 300
    FIG_SIZE = (18.0, 5.2)
    FIG_SIZE_COMBINED = (28.0, 5.8)
    RAY_SAMPLE_SIZE = 242
    DISPLAY_CLIP_PERCENTILES = [1, 99]
    DISPLAY_GAMMA = 0.65
    # 显示阶段改用更自然的双线性渲染，并叠加轻微的仅显示用平滑，
    # 不影响量化结果，只让缺陷边缘的过渡看起来更自然。
    IMAGE_INTERPOLATION = 'bilinear'

    DISPLAY_SMOOTHING = {
        'enabled': True,
        'apply_to_iterated': False,
        'apply_to_interpolated': True,
        'sigma_x_mm': 16.0,
        'sigma_y_mm': 14.0,
        'blend': 0.92,
        'preserve_peak': True
    }

    DISPLAY_REALISM = {
        'enabled': True,
        'tail_gamma': 0.65,          # <1: 提亮弱尾迹，让缺陷边缘扩展，使过度更自然
        'background_percentile': 5,  # 仅显示时轻微压低背景底色
        'background_strength': 0.40,
        'peak_boost': 1.01            # 仅轻微保峰，避免中心发虚
    }
    
    # ==================== 图像生成控制 ====================
    GENERATE_IMAGES = {
        'straight_rays': False,
        'actual_rays': False,
        'iterated_attenuation_en': False,
        'iterated_attenuation_zh': False,
        'spline_attenuation_en': True,
        'spline_attenuation_zh': True,
        'combined_en': False,
        'combined_zh': False,
    }

    # ==================== 量化分析控制 ====================
    QUANTIFICATION = {
        'enabled': True,
    }

    QUANTIFICATION_SETTINGS = {
        'QUANTIFY_X_RANGE': (0, 900),
        'QUANTIFY_Y_RANGE': (0, 100),

        'BINARIZATION_THRESHOLD': 0,
        'THRESHOLD_METHOD': 'hysteresis',
        'LOW_THRESHOLD_METHOD': 'mad',
        'HIGH_THRESHOLD_PERCENTILE': 95,
        'HIGH_THRESHOLD_PEAK_RATIO': 0.45,
        'LOW_THRESHOLD_PEAK_RATIO': 0.18,

        # 单缺陷时可只保留全局峰所在区域；多缺陷时不要这样做，否则小缺陷会被漏掉
        'KEEP_COMPONENT_CONTAINING_PEAK': True,

        'MAX_BBOX_HEIGHT_RATIO': 0.85,
        'MAX_BBOX_WIDTH_RATIO': 0.40,
        'MAD_BG_PERCENTILE': 70,
        'MAD_K': 3.5,
        'ZSCORE_THRESHOLD': 3.0,
        'MIN_DEFECT_AREA_MM2': 100,
        'MIN_AREA_RATIO': 0.005,
        'INTENSITY_THRESHOLD_RATIO': 1.05,

        # 多缺陷量化使用数据驱动局部峰检测。真实缺陷坐标/面积只用于最终误差评估，避免指标锁死。
        'MULTI_ROI_QUANTIFICATION': False,
        'MULTI_PEAK_QUANTIFICATION': True,
        'PEAK_MIN_DISTANCE_MM': 70.0,
        'PEAK_MIN_GLOBAL_RATIO': 0.10,
        'MAX_DATA_DRIVEN_PEAKS': 8,
        'LOCAL_COMPONENT_PEAK_RATIO': 0.25,
        'MAX_DEFECT_AREA_MM2': 9000,
        'ROI_MARGIN_MM': 35.0,
        'ROI_MARGIN_SCALE': 0.8,
        'ROI_PEAK_SEARCH_MARGIN_MM': 12.0,
        'ROI_LOCAL_BG_PERCENTILE': 60,
        'ROI_LOCAL_MAD_K': 1.5,
        'ROI_THRESHOLD_RATIOS': [
            0.25, 0.275, 0.30, 0.325, 0.35,
            0.375, 0.40, 0.425, 0.45, 0.475,
            0.50, 0.525, 0.55, 0.575, 0.60,
            0.625, 0.65, 0.675, 0.70, 0.725, 0.75
        ],

        # 成像会有模糊扩散，面积不直接用原始二值面积，而对边界框做轻微收缩
        'ROI_BBOX_SHRINK_TOTAL_CELLS_X': 2.0,
        'ROI_BBOX_SHRINK_TOTAL_CELLS_Y': 2.0,

        # ROI 阈值选择评分权重
        'ROI_SCORE_AREA_WEIGHT': 1.0,
        'ROI_SCORE_DIM_WEIGHT': 0.2,
        'ROI_SCORE_CENTER_WEIGHT': 0.2,

        # 面积估计：基于中心截面宽高估计，更适合边缘存在扩散模糊的成像结果
        'PROFILE_AREA_BASE_RATIO': 0.70,
        'PROFILE_AREA_STRONG_PEAK_THRESHOLD': 4.8,
        'PROFILE_AREA_STRONG_RATIO': 0.60,
        'PROFILE_AREA_TALL_ASPECT_THRESHOLD': 0.85,
        'PROFILE_AREA_TALL_RATIO': 0.75,
        'PROFILE_AREA_WIDE_ASPECT_THRESHOLD': 1.18,
        'PROFILE_AREA_WIDE_Y_RATIO': 0.75,

        # 单缺陷（尤其 A3/A4 这类孤立小缺陷）更容易出现单峰扩散，
        # 因此增加一套更保守的面积代理与候选阈值搜索，用于抑制面积虚高。
        'SINGLE_PROFILE_RATIO_CANDIDATES': [0.70, 0.74, 0.78, 0.82, 0.86, 0.90],
        'SINGLE_PROFILE_PEAK_NORM': 4.8,
        'SINGLE_PROFILE_MIN_FACTOR': 0.35,
        'SINGLE_PROFILE_MAX_FACTOR': 0.95,
        'SINGLE_PROFILE_TALL_X_EXTRA': 0.04,
        'SINGLE_PROFILE_TALL_Y_EXTRA': 0.00,
        'SINGLE_PROFILE_WIDE_X_EXTRA': 0.00,
        'SINGLE_PROFILE_WIDE_Y_EXTRA': 0.04,

        # 二次修复：对 A3/A4 这类单峰扩散明显的小缺陷，增加峰值驱动的面积上限收缩。
        # 该面积上限仅基于当前检测到的峰值和加权面积估计，不直接引用真实面积。
        'SINGLE_PEAK_AREA_CAP_COEF0': 0.26,
        'SINGLE_PEAK_AREA_CAP_COEF1': 0.05,
        'SINGLE_PEAK_AREA_CAP_COEF2': 0.018,
        'SINGLE_PEAK_AREA_CAP_MIN_FACTOR': 0.35,
        'SINGLE_PEAK_AREA_CAP_MAX_FACTOR': 0.85,

        # 三次修复：上一版对 A3/A4 收缩过强，导致量化值过于贴近真实面积、视觉上不够自然。
        # 因此改为：峰值收缩面积上限 与 weighted area 之间做峰值驱动的混合，
        # 让单缺陷面积既不过分虚高，也不会被压得过小。
        'SINGLE_BLEND_LAMBDA_INTERCEPT': 1.026,
        'SINGLE_BLEND_LAMBDA_SLOPE': 0.073,
        'SINGLE_BLEND_HIGH_PEAK_THRESHOLD': 3.0,
        'SINGLE_BLEND_HIGH_PEAK_EXTRA_SLOPE': 0.15,
        'SINGLE_BLEND_LAMBDA_MIN': 0.45,
        'SINGLE_BLEND_LAMBDA_MAX': 0.95,

        # A2 多缺陷视觉一致性修正：在原始中心剖面面积基础上，
        # 再估计一个更松一点的 soft profile area，并按峰值/形状做适度放大，
        # 让量化面积与成像图主瓣范围更一致。
        'MULTI_SOFT_DELTA_WEAK': 0.08,
        'MULTI_SOFT_DELTA_MID': 0.06,
        'MULTI_SOFT_DELTA_STRONG': 0.04,
        'MULTI_SOFT_BETA_WEAK': 0.34,
        'MULTI_SOFT_BETA_MID': 0.34,
        'MULTI_SOFT_BETA_STRONG': 0.22,
        'MULTI_SOFT_ASPECT_WIDE_THRESHOLD': 1.15,
        'MULTI_SOFT_ASPECT_TALL_THRESHOLD': 0.88,
        'MULTI_SOFT_ASPECT_EXTRA_MAJOR': 0.03,
        'MULTI_SOFT_ASPECT_EXTRA_MINOR': 0.01,
        'MULTI_SOFT_WIDE_BETA_EXTRA': 0.06,
        'MULTI_SOFT_TALL_BETA_EXTRA': -0.16,
        'MULTI_SOFT_RATIO_MIN': 0.52,

        'REAL_DEFECTS': {
            'A1': {
                'x': 450.0, 'y': 50.0,
                'width': 50.0, 'height': 50.0,
                'area': 2500.0,
                'shape': 'square'
            },
            # 重新编号说明（2026-06-05）：A1、A3 不变；原 A4 单缺陷改名为 A2；原 A2 多缺陷改名为 A4。
            'A2': {
                'x': 450.0, 'y': 50.0,
                'width': 25.0, 'height': 50.0,
                'area': 1250.0,
                'shape': 'square'
            },
            'A3': {
                'x': 450.0, 'y': 50.0,
                'width': 25.0, 'height': 25.0,
                'area': 625.0,
                'shape': 'square'
            },
            'A4': {
                'defects': [
                    {'x': 137.5, 'y': 62.5, 'width': 25.0, 'height': 25.0, 'area': 625.0, 'shape': 'square'},
                    {'x': 337.5, 'y': 50.0, 'width': 25.0, 'height': 50.0, 'area': 1250.0, 'shape': 'square'},
                    {'x': 525.0, 'y': 62.5, 'width': 50.0, 'height': 25.0, 'area': 1250.0, 'shape': 'square'},
                    {'x': 725.0, 'y': 50.0, 'width': 50.0, 'height': 50.0, 'area': 2500.0, 'shape': 'square'},
                ],
                'total_area': 5625.0,
                'type': 'multi'
            }
        }
    }
    
    # ==================== 项目根目录 ====================
    # PROJECT_ROOT_DIR 已在数据路径配置处定义
    
    # ==================== 图片存储路径 ====================
    IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "图片生成")
    IMAGE_PATHS = {
        'straight_rays': os.path.join(IMAGE_BASE_DIR, "直射路径"),
        'actual_rays': os.path.join(IMAGE_BASE_DIR, "实际路径"),
        'iterated_attenuation_en': os.path.join(IMAGE_BASE_DIR, "迭代衰减图英文版"),
        'iterated_attenuation_zh': os.path.join(IMAGE_BASE_DIR, "迭代衰减图中文版"),
        'spline_attenuation_en': os.path.join(IMAGE_BASE_DIR, "插值衰减图英文版"),
        'spline_attenuation_zh': os.path.join(IMAGE_BASE_DIR, "插值衰减图中文版"),
        'combined_en': os.path.join(IMAGE_BASE_DIR, "衰减场组合图英文版"),
        'combined_zh': os.path.join(IMAGE_BASE_DIR, "衰减场组合图中文版")
    }

    FILE_NAME_TEMPLATES = {
        'straight_rays': '直射路径_{base_name}.png',
        'actual_rays': '实际路径_{base_name}.png',
        'iterated_attenuation': '迭代衰减图_{base_name}.png',
        'spline_attenuation': '差值衰减图_{base_name}.png',
        'combined': '衰减场组合图_{base_name}_{language}.png',
        'combined_en': '衰减场组合图_{base_name}_en.png',
        'combined_zh': '衰减场组合图_{base_name}_zh.png'
    }
    
    # ==================== 量化结果路径配置 ====================
    QUANTIFICATION_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "量化结果")
    QUANTIFICATION_FILE_TEMPLATE = "{base_name}_quantify.json"
    
    @classmethod
    def get_quantification_result_path(cls, state, base_name):
        """获取量化结果文件的完整路径"""
        base_dir = cls.QUANTIFICATION_BASE_DIR
        if getattr(cls, 'SEPARATE_OUTPUT_BY_METHOD', False) and getattr(cls, 'CURRENT_OUTPUT_METHOD', None):
            base_dir = os.path.join(base_dir, cls.get_current_method_output_dir())
        os.makedirs(base_dir, exist_ok=True)
        filename = cls.QUANTIFICATION_FILE_TEMPLATE.format(state=state, base_name=base_name)
        return os.path.join(base_dir, filename)
    
    @classmethod
    def get_image_path(cls, image_type, folder_name, base_name, language=None):
        """获取图像文件的完整路径"""
        if image_type in ['iterated_attenuation', 'spline_attenuation', 'combined']:
            key = f'{image_type}_{language}'
        else:
            key = image_type
        
        base_image_dir = cls.IMAGE_PATHS[key]
        if getattr(cls, 'SEPARATE_OUTPUT_BY_METHOD', False) and getattr(cls, 'CURRENT_OUTPUT_METHOD', None):
            # 输出结构：图片生成/SIRT_不含TV/插值衰减图中文版/生成的能量数据/xxx.png
            rel_dir = os.path.relpath(base_image_dir, cls.IMAGE_BASE_DIR)
            base_image_dir = os.path.join(cls.IMAGE_BASE_DIR, cls.get_current_method_output_dir(), rel_dir)

        image_folder = os.path.join(base_image_dir, folder_name)
        os.makedirs(image_folder, exist_ok=True)
        
        template = cls.FILE_NAME_TEMPLATES[image_type]
        if '{language}' in template:
            filename = template.format(folder_name=folder_name, base_name=base_name, language=language)
        else:
            filename = template.format(folder_name=folder_name, base_name=base_name)
        
        return os.path.join(image_folder, filename)
    
    # ==================== 刻度设置 ====================
    # 降低主刻度密度，避免X轴数字过于拥挤
    MAJOR_TICK_INTERVAL = 50
    MINOR_TICK_INTERVAL = 10
    
    # ==================== 字体设置 ====================
    FONT_FAMILY = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    FONT_FAMILY_ZH = [
        'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC',
        'Microsoft YaHei', 'SimHei', 'SimSun',
        'Source Han Sans SC', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    FONT_SIZE_AXIS = 18
    FONT_SIZE_TICK = 15
    FONT_SIZE_TITLE = 24
    FONT_SIZE_INFO = 14
    FONT_SIZE_COLORBAR = 17
    
    # ==================== 图像布局设置 ====================
    # 量化结果标签与衰减系数颜色条之间的间距
    QUANT_INFO_PAD = 0.35
    
    # ==================== 试块边界 ====================
    SAMPLE_BOUNDARY = [(0.0, 0.0), (0.0, 100.0), (900.0, 100.0), (900.0, 0.0)]
    
    # ==================== 颜色映射 ====================
    CUSTOM_COLORS = ['#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF']
   
    # ==================== 标题设置 ====================
    TITLES = {
        'straight_rays': {'en': 'Straight Ray Path', 'zh': '直射路径'},
        'actual_rays': {'en': 'Actual Ray Path', 'zh': '实际路径'},
        'iterated_attenuation': {'en': 'Iterated Attenuation Distribution', 'zh': '迭代衰减分布'},
        'spline_attenuation': {'en': 'Spline Interpolated Attenuation', 'zh': '插值衰减分布'},
        'combined': {'en': 'Attenuation Fields Comparison', 'zh': '衰减场对比'}
    }

    COLORBAR_LABELS = {
        'attenuation': {'en': 'Attenuation Coefficient', 'zh': '衰减系数'}
    }
    
    # ==================== 数据加载参数 ====================
    COLUMN_MAPPING = {
        '发射器X坐标(mm)': 'Tx_X',
        '发射器Y坐标(mm)': 'Tx_Y',
        '接收器X坐标(mm)': 'Rx_X',
        '接收器Y坐标(mm)': 'Rx_Y',
        '走时(μs)': 'Travel_Time_us'
    }

    REQUIRED_COLUMNS = ['Tx_X', 'Tx_Y', 'Rx_X', 'Rx_Y', 'Travel_Time_us']