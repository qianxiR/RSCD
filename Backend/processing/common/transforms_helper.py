"""
数据变换构建

入参: get_transform(args) — args 需含 dataset/in_height/in_width 字段
方法: 用临时 Namespace 调 BCDTransforms.get_transform_pipelines，避免污染外部 args
出参: 验证/推理用的 val_transform 可调用对象
"""
import argparse

from Backend.data.transforms import BCDTransforms


def get_transform(args):
    """
    获取验证/推理用的数据变换

    入参:
        args: argparse.Namespace，需包含 dataset/in_height/in_width 字段；
              in_height/in_width<=0 时回退到 256
    方法: 构造临时 Namespace（仅含 dataset 与尺寸），调用 BCDTransforms 取 val_transform
    出参: val_transform（Compose 对象）

    做什么: 构建仅用于推理的变换管道，不修改调用方的 args
    为什么: 取自 batch_image.py 的无副作用版本；single_image.py 旧实现会直接
            覆盖 args.in_height/width，污染后续逻辑，统一改用局部对象隔离
    """
    # 用临时参数对象隔离 BCDTransforms 可能对 args 的依赖，避免污染外部 args
    temp_args = argparse.Namespace()
    temp_args.dataset = args.dataset
    temp_args.in_height = args.in_height if args.in_height > 0 else 256
    temp_args.in_width = args.in_width if args.in_width > 0 else 256

    _, val_transform = BCDTransforms.get_transform_pipelines(temp_args)
    return val_transform
