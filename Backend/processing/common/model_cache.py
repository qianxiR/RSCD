"""
模型加载与缓存

入参: load_model(args) — args 需含 checkpoint/device/in_height/in_width/num_perception_frame 等字段
方法: 以 (checkpoint, device) 为键缓存 Trainer 实例，命中时按需重建感知帧
出参: Trainer 实例（eval 模式）
"""
import argparse

import torch

from Backend.network.encoder import Trainer

# 全局模型缓存字典：键为 "{checkpoint}_{device}"，值为已加载权重的 Trainer 实例
# 做什么: 避免同一权重在批量/连续调用时反复磁盘 IO 与权重加载
# 为什么: X3D 权重约 50MB，反复 torch.load 会显著拖慢批量推理
MODEL_CACHE = {}


def load_model(args):
    """
    加载模型和权重，使用缓存机制避免重复加载

    入参:
        args: argparse.Namespace，需包含 checkpoint/device/in_height/in_width/
              num_perception_frame 等字段；in_height/in_width<=0 时回退到 256
    方法: 构建缓存键 → 命中则校验感知帧尺寸并按需重建 → 未命中则构造 Trainer、
          容错加载权重（尺寸不匹配的参数被跳过）、置 eval、写入缓存
    出参: Trainer 实例（已 .eval()）

    做什么: 统一的模型加载入口，含权重尺寸不匹配容错与感知帧动态重建
    为什么: 取自 batch_image.py 的最完善版本，单图版缺少这两段容错会导致
            不同 patch_size 下推理失败
    """
    # 构建缓存键 - 使用检查点路径和设备信息
    cache_key = f"{args.checkpoint}_{args.device}"

    # 确保模型输入尺寸有效
    if args.in_height <= 0 or args.in_width <= 0:
        args.in_height = 256
        args.in_width = 256

    # 检查缓存中是否已存在模型
    if cache_key in MODEL_CACHE:
        # 从缓存获取模型
        model = MODEL_CACHE[cache_key]

        # 检查模型参数是否与当前需求匹配
        current_perception_frame_size = (args.num_perception_frame, args.in_height, args.in_width)
        if hasattr(model.encoder, 'perception_frames'):
            try:
                model_perception_frame_size = (
                    model.encoder.perception_frames.shape[2],
                    model.encoder.perception_frames.shape[3],
                    model.encoder.perception_frames.shape[4]
                )

                # 如果尺寸不匹配，则重新创建感知帧
                if current_perception_frame_size != model_perception_frame_size:
                    # 创建新的感知帧参数
                    new_perception_frames = torch.randn(
                        1, 3, args.num_perception_frame, args.in_height, args.in_width,
                        requires_grad=True,
                        device=args.device
                    )
                    # 替换模型中的感知帧
                    model.encoder.perception_frames = torch.nn.Parameter(new_perception_frames)
            except Exception:
                pass

        return model

    # 创建模型 - 确保此时 args.in_height 和 args.in_width 是有效的
    model = Trainer(args).to(args.device)

    # 加载权重（兼容两种格式：带 'state_dict' 包裹的 checkpoint / 裸 state_dict）
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    # best_model.pth 是裸 state_dict（OrderedDict），checkpoint.pth.tar 是 dict 含 'state_dict' 键
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    # 尝试加载模型参数，但忽略感知帧参数，因为尺寸可能不匹配
    try:
        # 常规加载尝试（strict=True，形状必须完全匹配）
        model.load_state_dict(state_dict)
    except RuntimeError:
        # 大多数错误是由尺寸不匹配导致的（如不同 patch_size 的 perception_frames）
        # 获取模型状态字典和检查点状态字典
        model_state_dict = model.state_dict()

        # 创建新的状态字典，仅包含形状匹配的参数
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                new_state_dict[k] = v
            # 形状不匹配的参数被跳过（感知帧等）

        # 加载匹配的参数
        model.load_state_dict(new_state_dict, strict=False)

    # 设置模型为评估模式
    model.eval()

    # 将模型保存到缓存
    MODEL_CACHE[cache_key] = model

    return model


def clear_model_cache():
    """
    清除模型缓存并释放显存

    入参: 无
    方法: 清空 MODEL_CACHE 字典，若 CUDA 可用则调用 empty_cache
    出参: 无

    做什么: 在长时间运行进程的合适时机（如模式切换）释放模型显存
    为什么: 不同检测模式可能用不同缓存键，旧模型不释放会持续占用显存
    """
    MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
