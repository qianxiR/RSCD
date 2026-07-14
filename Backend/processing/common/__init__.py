"""
RSCD Backend 公共处理层

集中存放 single_image / single_raster / batch_image / batch_raster 四个入口
共用的辅助函数，消除历史复制粘贴。所有函数均保持原行为契约。

入参: 见各子模块
方法: 提供模型缓存、变换构建、滑窗/权重、内存估算、可视化、IO 六类工具
出参: 工具函数与全局缓存（MODEL_CACHE）
"""
from .model_cache import MODEL_CACHE, load_model, clear_model_cache
from .geo_probe import has_georeference
from .transforms_helper import get_transform
from .sliding_window import (
    create_sliding_windows,
    create_weight_map,
    determine_optimal_patch_size,
    determine_optimal_stride,
)
from .memory import estimate_memory_usage, adjust_batch_size
from .visualization import visualize_results
from .io_utils import preprocess_image, ensure_output_dir

__all__ = [
    "MODEL_CACHE",
    "load_model",
    "clear_model_cache",
    "has_georeference",
    "get_transform",
    "create_sliding_windows",
    "create_weight_map",
    "determine_optimal_patch_size",
    "determine_optimal_stride",
    "estimate_memory_usage",
    "adjust_batch_size",
    "visualize_results",
    "preprocess_image",
    "ensure_output_dir",
]
