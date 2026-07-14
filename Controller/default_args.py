"""
推理默认参数配置

集中维护 change_detection_model._create_args 的参数默认值，便于配置管理与调整。

入参: build_default_args(before_path, after_path, output_path, model_path)
方法: 用 DEFAULT_INFERENCE_ARGS 字典构造 argparse.Namespace
出参: argparse.Namespace，含全部模型/滑窗/栅格/矢量所需字段
"""
import argparse

import torch


# 推理默认参数字典：被 build_default_args 用于构造 Namespace
# 做什么: 集中维护 4 个检测模式共用的参数默认值
# 为什么: 历史 _create_args 中 50+ 行硬编码赋值难以维护与查阅，外置后可一目了然
DEFAULT_INFERENCE_ARGS = {
    # 模型与设备
    "checkpoint": "checkpoint/best_model.pth",  # 由 build_default_args 用 model_path 覆盖；best_model 是裸 state_dict
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "pretrained": None,
    "model_arch": "siam_unet",

    # 图像/滑窗
    "patch_size": 256,
    "in_height": 256,
    "in_width": 256,
    "stride_ratio": 0.5,
    "batch_size": 16,
    "overlap_weights": True,
    "block_size": 512,
    "raw_output": False,
    "auto_memory": False,
    "max_patch_size": 1024,
    "min_patch_size": 256,
    "auto_patch_divisor": 16,
    "warp_projection": None,

    # 模型结构
    "num_class": 1,
    "num_perception_frame": 1,
    "dataset": "CD",

    # 输出/保存
    "binary_mask": True,
    "save_binary_mask": True,
    "save_result": True,
    "save_visualization": True,
    "max_images": 0,
    "quiet": True,

    # 批量文件配对
    "file_ext": ".png,.jpg,.jpeg,.tif,.tiff",

    # 栅格处理
    "ignore_geo": False,
    "band_indices": "1,2,3",
    "warp_projection": None,

    # 矢量导出
    "export_shapefile": True,
    "export_geojson": True,
    "min_polygon_area": 100.0,
    "simplify_tolerance": 0.5,
    "attribute_change_type": True,
    "calculate_area": True,

    # 矢量合并
    "merge_vectors": True,
    "merged_file_name": "merged_changes",
}


def build_default_args(before_path, after_path, output_path, model_path=None):
    """
    构造推理用 argparse.Namespace

    入参:
        before_path/after_path: 前后时相文件/目录路径
        output_path: 输出路径
        model_path: 模型权重路径，None 时用 DEFAULT_INFERENCE_ARGS['checkpoint']
    方法: 复制默认参数字典 → 覆盖路径字段 → 构造 Namespace
    出参: argparse.Namespace

    做什么: 统一的推理参数构造入口
    为什么: 取代 detection_service._create_args 中 50+ 行硬编码赋值
    """
    args = argparse.Namespace()
    for key, value in DEFAULT_INFERENCE_ARGS.items():
        setattr(args, key, value)

    # 路径字段（每次调用必须覆盖）
    args.before_path = before_path
    args.after_path = after_path
    args.output_path = output_path
    args.checkpoint = model_path if model_path else DEFAULT_INFERENCE_ARGS["checkpoint"]

    return args
