"""
通用 IO 工具

入参: preprocess_image(image_path) / ensure_output_dir(output_path)
方法: 图像读取（灰度/多波段统一到 3 通道）/ 输出目录确保存在
出参: numpy 图像数组 / bool
"""
import os

import numpy as np
from skimage import io


def preprocess_image(image_path):
    """
    读取和预处理图像

    入参:
        image_path: 图像文件路径
    方法: skimage 读取 → 灰度图复制为 3 通道 → 多波段截取前 3 通道
    出参: numpy.ndarray，形状 (h, w, 3)，uint8

    做什么: 将任意通道数的输入统一为模型可接受的 3 通道 RGB
    为什么: X3D 模型输入通道固定为 3，灰度/多波段需在推理前归一
    """
    try:
        img = io.imread(image_path)
        if len(img.shape) == 2:
            # 灰度图复制为三通道
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] > 3:
            # 处理多波段图像，仅取前 3 个波段
            img = img[:, :, :3]
        return img
    except Exception as e:
        raise IOError(f"无法读取图像: {image_path}, 错误: {str(e)}")


def ensure_output_dir(output_path):
    """
    确保输出路径所在的目录存在

    入参:
        output_path: 文件路径或目录路径
    方法: 取父目录（若为文件）或自身（若为目录），不存在则创建
    出参: bool，已存在或创建成功返回 True

    做什么: 集中处理"输出目录可能不存在"的常见前置条件
    为什么: 4 个入口文件的 process_and_save 都有相同逻辑，统一避免重复
    """
    if os.path.isdir(output_path) or output_path.endswith(os.sep):
        target_dir = output_path
    else:
        target_dir = os.path.dirname(output_path)

    if not target_dir:
        return True  # 当前目录

    os.makedirs(target_dir, exist_ok=True)
    return True
