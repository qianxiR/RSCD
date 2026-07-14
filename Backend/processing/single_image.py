#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 单影像变化检测推理（统一入口）
#
# 自动探测地理坐标：有坐标 → 保留地理参考的滑窗推理，输出 GeoTIFF 掩膜 + 四联图 + 矢量；
# 无坐标 → block+patch 双层滑窗推理，输出 PNG 掩膜 + 四联图，跳过矢量。
# 依赖 GDAL / rasterio / geopandas（仅坐标路径需要），GDAL 不可用时退化为无坐标路径。

"""
单影像变化检测入口

入参: process_and_save(args) — args 需含 before_path/after_path/output_path/checkpoint/device 等
方法: geo 探测 → 选择坐标/无坐标路径 → 模型懒加载 → 滑窗推理 → 掩膜/可视化/矢量落盘
出参: Dict[str, Any] — status/message/output_path/quad_view_path/vector_files/processing_time
"""
import os
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# 公共工具统一从 common 子包导入
from Backend.processing.common import (
    load_model,
    clear_model_cache,
    get_transform,
    create_weight_map,
    has_georeference,
    visualize_results,
)
from Backend.processing.common.io_utils import ensure_output_dir


# GDAL 与栅格/矢量工具仅在 GDAL 可用时导入，不可用时走无坐标路径
try:
    from osgeo import gdal
    gdal.UseExceptions()
    from Backend.processing.common import (
        create_sliding_windows,
        determine_optimal_patch_size,
        determine_optimal_stride,
        adjust_batch_size,
    )
    from Backend.processing.raster import (
        read_geotiff,
        save_geotiff_result,
        mask_to_lines,
        export_vector,
    )
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False


def get_args(before_path=None, after_path=None, output_path=None, model_path=None, **kwargs):
    """
    构建单影像推理参数命名空间

    入参:
        before_path/after_path/output_path: 路径，None 时由命令行解析
        model_path: 模型权重路径，将被映射为 args.checkpoint
        **kwargs: 任意覆盖参数（如 patch_size/batch_size/device）
    方法: 构建 argparse → 无命令行参数时用入参填充 → 强制 in_height/width = patch_size（无坐标路径用）
    出参: argparse.Namespace

    做什么: 单影像模式的参数入口，同时兼容进程内调用与 CLI 调用
    为什么: 无坐标路径要求输入尺寸严格等于 patch_size，故默认对齐；坐标路径会自行重设 patch_size
    """
    parser = argparse.ArgumentParser(description='影像变化检测')

    parser.add_argument('--before_path', type=str, default=before_path, help='前时相影像路径')
    parser.add_argument('--after_path', type=str, default=after_path, help='后时相影像路径')
    parser.add_argument('--output_path', type=str, default=output_path, help='输出路径')

    parser.add_argument('--model_path', type=str, default=model_path, help='模型路径')
    parser.add_argument('--model_arch', type=str, default='siam_unet', help='模型架构')
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='设备 (例如 cuda:0 或 cpu)')

    parser.add_argument('--patch_size', type=int, default=256, help='图像块大小')
    parser.add_argument('--block_size', type=int, default=512, help='处理块大小（无坐标路径用）')
    parser.add_argument('--stride_ratio', type=float, default=0.5, help='滑动步长比例')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--overlap_weights', action='store_true', help='使用加权融合处理重叠区域')

    parser.add_argument('--raw_output', action='store_true', help='输出原始预测 (浮点数) 而不是二值化结果')
    parser.add_argument('--save_binary_mask', action='store_true', default=True, help='保存二值掩码')
    parser.add_argument('--save_visualization', action='store_true', default=True, help='保存四连图可视化结果')

    parser.add_argument('--in_height', type=int, default=256)
    parser.add_argument('--in_width', type=int, default=256)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--num_perception_frame', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='CD')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--auto_memory', action='store_true', default=False)
    parser.add_argument('--quiet', action='store_true', default=True)

    # 坐标路径专属参数（无坐标路径不使用）
    parser.add_argument('--max_patch_size', type=int, default=1024)
    parser.add_argument('--min_patch_size', type=int, default=256)
    parser.add_argument('--auto_patch_divisor', type=int, default=16)
    parser.add_argument('--ignore_geo', action='store_true', default=False)
    parser.add_argument('--band_indices', type=str, default='1,2,3')
    parser.add_argument('--warp_projection', type=str, default=None)

    # 矢量导出参数（仅坐标路径有效）
    parser.add_argument('--export_shapefile', action='store_true', default=True)
    parser.add_argument('--export_geojson', action='store_true', default=True)
    parser.add_argument('--min_polygon_area', type=float, default=100.0)
    parser.add_argument('--simplify_tolerance', type=float, default=0.5)
    parser.add_argument('--attribute_change_type', action='store_true', default=True)
    parser.add_argument('--calculate_area', action='store_true', default=True)

    import sys as _sys
    if len(_sys.argv) <= 1 and (before_path or after_path or output_path or model_path or kwargs):
        args = parser.parse_args([])
        if before_path: args.before_path = before_path
        if after_path: args.after_path = after_path
        if output_path: args.output_path = output_path
        if model_path: args.checkpoint = model_path
        for key, value in kwargs.items():
            if hasattr(args, key):
                setattr(args, key, value)
    else:
        args = parser.parse_args()

    # 无坐标路径要求输入尺寸等于 patch_size
    if not hasattr(args, 'in_height') or args.in_height != args.patch_size:
        args.in_height = args.patch_size
    if not hasattr(args, 'in_width') or args.in_width != args.patch_size:
        args.in_width = args.patch_size

    return args


# ==================== 无坐标路径：block + patch 双层滑窗 ====================

def process_large_image(before_path, after_path, model, transform, args):
    """
    处理无坐标影像（block + patch 双层滑窗）

    入参:
        before_path/after_path: 前后时相影像路径
        model: 已加载的 Trainer 实例
        transform: 推理用变换管道
        args: 需含 block_size/patch_size/stride_ratio/batch_size/device/overlap_weights/raw_output
    方法: 读图并对齐尺寸 → 按 block_size 切块 → 块内按 patch_size 滑窗 → 批量推理 →
          高斯权重融合重叠区 → 二值化（除非 raw_output）
    出参: tuple(pred_mask, pre_img, post_img)

    做什么: 无坐标路径特有的双层滑窗推理
    为什么: block 层用于控制单次内存峰值，patch 层用于模型推理
    """
    before_path = before_path.replace('\\', '/')
    after_path = after_path.replace('\\', '/')

    pre_img = cv2.imread(before_path)
    post_img = cv2.imread(after_path)

    if pre_img is None or post_img is None:
        # OpenCV 读取失败时尝试 GDAL（部分 TIFF cv2 无法读但 GDAL 可读，读后转 uint8）
        pre_img = _gdal_read_as_uint8(before_path) if GDAL_AVAILABLE else pre_img
        post_img = _gdal_read_as_uint8(after_path) if GDAL_AVAILABLE else post_img
        if pre_img is None or post_img is None:
            raise ValueError(f"无法读取影像: {before_path} 或 {after_path}")

    if pre_img.shape != post_img.shape:
        min_height = min(pre_img.shape[0], post_img.shape[0])
        min_width = min(pre_img.shape[1], post_img.shape[1])
        pre_img = cv2.resize(pre_img, (min_width, min_height))
        post_img = cv2.resize(post_img, (min_width, min_height))

    height, width = pre_img.shape[:2]
    block_size = args.block_size
    patch_size = args.patch_size
    stride = int(patch_size * args.stride_ratio)
    if stride == 0:
        stride = patch_size

    pred_mask_accum = np.zeros((height, width), dtype=np.float32)
    weight_accum = np.zeros((height, width), dtype=np.float32)

    weight_map = create_weight_map(patch_size, stride) if args.overlap_weights \
        else np.ones((patch_size, patch_size), dtype=np.float32)

    num_blocks_y = math.ceil(height / block_size)
    num_blocks_x = math.ceil(width / block_size)

    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            start_y = by * block_size
            start_x = bx * block_size
            end_y = min(start_y + block_size, height)
            end_x = min(start_x + block_size, width)

            pre_block = pre_img[start_y:end_y, start_x:end_x]
            post_block = post_img[start_y:end_y, start_x:end_x]
            block_h, block_w = pre_block.shape[:2]

            current_patch_size_h = min(patch_size, block_h)
            current_patch_size_w = min(patch_size, block_w)
            current_stride_h = min(stride, current_patch_size_h)
            current_stride_w = min(stride, current_patch_size_w)
            if current_stride_h == 0: current_stride_h = 1
            if current_stride_w == 0: current_stride_w = 1

            patch_coords = []
            for py in range(0, block_h - current_patch_size_h + 1, current_stride_h):
                actual_py = py if py + current_patch_size_h <= block_h else block_h - current_patch_size_h
                for px in range(0, block_w - current_patch_size_w + 1, current_stride_w):
                    actual_px = px if px + current_patch_size_w <= block_w else block_w - current_patch_size_w
                    patch_coords.append((actual_px, actual_py, actual_px + current_patch_size_w, actual_py + current_patch_size_h))
            patch_coords = sorted(list(set(patch_coords)))

            for i in range(0, len(patch_coords), args.batch_size):
                batch_coords = patch_coords[i:i + args.batch_size]
                batch_pre = []
                batch_post = []

                for x1, y1, x2, y2 in batch_coords:
                    patch_pre = pre_block[y1:y2, x1:x2]
                    patch_post = post_block[y1:y2, x1:x2]

                    if patch_pre is not None and patch_post is not None and patch_pre.size > 0 and patch_post.size > 0:
                        try:
                            patch_pre = _ensure_rgb(patch_pre)
                            patch_post = _ensure_rgb(patch_post)

                            image = np.concatenate([patch_pre, patch_post], axis=2)
                            mask = np.zeros(patch_pre.shape[:2], dtype=np.float32)
                            image_t, _ = transform(image, mask)
                            batch_pre.append(image_t[0:3])
                            batch_post.append(image_t[3:6])
                        except Exception:
                            continue

                if not batch_pre or not batch_post:
                    continue

                batch_pre_t = torch.stack(batch_pre).to(args.device)
                batch_post_t = torch.stack(batch_post).to(args.device)

                with torch.no_grad():
                    outputs = model.update_bcd(batch_pre_t, batch_post_t)

                preds = outputs.squeeze(1).cpu().numpy()

                for j, (x1, y1, x2, y2) in enumerate(batch_coords):
                    pred_patch = preds[j]
                    global_y1, global_y2 = start_y + y1, start_y + y2
                    global_x1, global_x2 = start_x + x1, start_x + x2

                    current_weight_map = cv2.resize(weight_map, (pred_patch.shape[1], pred_patch.shape[0])) \
                        if pred_patch.shape != weight_map.shape[:2] else weight_map

                    pred_mask_accum[global_y1:global_y2, global_x1:global_x2] += pred_patch * current_weight_map
                    weight_accum[global_y1:global_y2, global_x1:global_x2] += current_weight_map

                del batch_pre_t, batch_post_t, outputs, preds
                if torch.cuda.is_available() and str(args.device) != 'cpu':
                    torch.cuda.empty_cache()

    pred_mask = np.divide(pred_mask_accum, weight_accum,
                          out=np.zeros_like(pred_mask_accum), where=weight_accum != 0)

    if not args.raw_output:
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    return pred_mask, pre_img, post_img


# ==================== 坐标路径：保留地理参考的滑窗 ====================

def process_large_raster(pre_img_path, post_img_path, model, transform, args):
    """
    使用滑窗处理带坐标影像（保留地理参考）

    入参:
        pre_img_path/post_img_path: 影像文件路径
        model/transform: 已加载的模型与变换
        args: 含 patch_size/stride_ratio/band_indices/ignore_geo/in_height/in_width 等
    方法: 读 GeoTIFF（保留 geo）→ 对齐尺寸 → pad 到 256 倍数 → 滑窗推理 → 权重融合
    出参: tuple(result_map, pre_img, post_img, geo_transform, projection, pre_ds)

    做什么: 坐标路径的滑窗推理，地理信息读取、256 倍数 padding、geo/projection 透传
    为什么: vs 无坐标的 block+patch，这里多了地理信息与 padding，避免边界 patch 尺寸不匹配
    """
    geo_transform = None
    projection = None
    pre_ds = None

    if args.in_height <= 0:
        args.in_height = 256
    if args.in_width <= 0:
        args.in_width = 256

    original_in_height = args.in_height
    original_in_width = args.in_width

    try:
        if not args.ignore_geo:
            band_indices = [int(i) for i in args.band_indices.split(',')] if args.band_indices else None
            pre_img, pre_geo_transform, pre_projection, pre_ds = read_geotiff(pre_img_path, band_indices)
            post_img, post_geo_transform, post_projection, _ = read_geotiff(post_img_path, band_indices)

            geo_transform = pre_geo_transform
            projection = pre_projection

            # 前后地理参考不一致时静默使用前时相
            if len(pre_geo_transform) != len(post_geo_transform) or \
               any(abs(a - b) > 1e-5 for a, b in zip(pre_geo_transform, post_geo_transform)) or \
               pre_projection != post_projection:
                geo_transform = pre_geo_transform
                projection = pre_projection
        else:
            pre_img = _gdal_read_as_uint8(pre_img_path)
            post_img = _gdal_read_as_uint8(post_img_path)

        if pre_img.shape[:2] != post_img.shape[:2]:
            post_img = cv2.resize(post_img, (pre_img.shape[1], pre_img.shape[0]))

        h, w = pre_img.shape[:2]

        MODEL_SIZE = 256
        pad_h = (MODEL_SIZE - h % MODEL_SIZE) % MODEL_SIZE
        pad_w = (MODEL_SIZE - w % MODEL_SIZE) % MODEL_SIZE
        if pad_h > 0 or pad_w > 0:
            pre_img = np.pad(pre_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            post_img = np.pad(post_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = pre_img.shape[:2]

        patch_size = determine_optimal_patch_size((h, w), args)
        stride = determine_optimal_stride(patch_size, args)
        args.patch_size = patch_size
        transform = get_transform(args)
        batch_size = adjust_batch_size(patch_size, args)
        windows = create_sliding_windows((h, w), patch_size, stride)

        if args.raw_output or args.overlap_weights:
            result_map = np.zeros((h, w), dtype=np.float32)
        else:
            result_map = np.zeros((h, w), dtype=np.uint8)

        weight_map = None
        count_map = None
        if args.overlap_weights:
            weight_map = create_weight_map(patch_size, stride)
            count_map = np.zeros((h, w), dtype=np.float32)

        num_batches = int(np.ceil(len(windows) / batch_size))

        model.eval()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_windows = windows[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                batch_pre = []
                batch_post = []
                batch_windows_valid = []

                for x1, y1, x2, y2 in batch_windows:
                    if x2 - x1 == patch_size and y2 - y1 == patch_size:
                        pre_patch = pre_img[y1:y2, x1:x2]
                        post_patch = post_img[y1:y2, x1:x2]

                        image = np.concatenate([pre_patch, post_patch], axis=2)
                        mask = np.zeros((patch_size, patch_size), dtype=np.float32)
                        image_t, _ = transform(image, mask)

                        pre_t = image_t[0:3].unsqueeze(0)
                        post_t = image_t[3:6].unsqueeze(0)

                        if pre_t.shape[2] != args.in_height or pre_t.shape[3] != args.in_width:
                            pre_t = F.interpolate(pre_t, size=(args.in_height, args.in_width),
                                                  mode='bilinear', align_corners=False)
                            post_t = F.interpolate(post_t, size=(args.in_height, args.in_width),
                                                   mode='bilinear', align_corners=False)

                        batch_pre.append(pre_t)
                        batch_post.append(post_t)
                        batch_windows_valid.append((x1, y1, x2, y2))

                if batch_pre:
                    batch_pre = torch.cat(batch_pre, dim=0).to(args.device)
                    batch_post = torch.cat(batch_post, dim=0).to(args.device)

                    outputs = model.update_bcd(batch_pre, batch_post)

                    if args.raw_output:
                        preds = outputs.squeeze(1).cpu().numpy()
                    else:
                        preds = (outputs > 0.5).float().squeeze(1).cpu().numpy()

                    for idx, (x1, y1, x2, y2) in enumerate(batch_windows_valid):
                        if idx < len(preds):
                            if args.overlap_weights:
                                result_map[y1:y2, x1:x2] += preds[idx] * weight_map
                                count_map[y1:y2, x1:x2] += weight_map
                            else:
                                result_map[y1:y2, x1:x2] = preds[idx] if args.raw_output else preds[idx] * 255

                    del batch_pre, batch_post, outputs, preds
                    if torch.cuda.is_available() and str(args.device) != 'cpu':
                        torch.cuda.empty_cache()

        if args.overlap_weights:
            count_map = np.maximum(count_map, 1e-6)
            result_map = result_map / count_map
            if not args.raw_output:
                result_map = (result_map > 0.5).astype(np.uint8) * 255

        return result_map, pre_img, post_img, geo_transform, projection, pre_ds

    finally:
        args.in_height = original_in_height
        args.in_width = original_in_width


# ==================== 工具函数 ====================

def _ensure_rgb(img):
    """
    将单通道或 4 通道图像统一为 3 通道 RGB

    入参: img — numpy 图像数组（HxW 或 HxWxC）
    方法: 灰度→RGB / 4通道截前3通道 / 已是3通道转 RGB 色序
    出参: numpy.ndarray (H, W, 3)
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        return img[:, :, :3]
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _gdal_read_as_uint8(raster_path):
    """
    GDAL 读取栅格为 3 通道 uint8（无坐标路径读 TIFF 时的回退）

    入参: raster_path — 栅格路径
    方法: 复用 read_geotiff 取 uint8 图像，丢弃 geo/projection
    出参: np.uint8 (h,w,3) 或 None（GDAL 不可用或打开失败）
    """
    if not GDAL_AVAILABLE:
        return None
    try:
        image, _, _, _ = read_geotiff(raster_path)
        return image
    except Exception:
        return None


# ==================== 统一入口 ====================

def process_and_save(args):
    """
    对外暴露的单影像处理入口（自动探测坐标）

    入参: args — Namespace，需含 before_path/after_path/output_path/checkpoint/device 等
    方法: 路径规范化 → geo 探测 → 选择坐标/无坐标路径 → 模型懒加载 → 推理 → 落盘
    出参: Dict[str, Any] — status/message/output_path/quad_view_path/vector_files/processing_time

    做什么: 编排整个单影像推理流程，自动决定是否保留坐标系
    为什么: 取消图像/影像二元划分后，是否输出 GeoTIFF+矢量由影像自身属性决定
    """
    start_time = time.time()

    if 'cuda' in str(args.device):
        args.device = torch.device(args.device)
    else:
        args.device = torch.device('cpu')

    args.before_path = args.before_path.replace('\\', '/')
    args.after_path = args.after_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')

    if not ensure_output_dir(args.output_path):
        return {"status": "error", "message": "无法创建输出目录"}

    if not os.path.exists(args.before_path):
        return {"status": "error", "message": f"前时相影像不存在: {args.before_path}"}
    if not os.path.exists(args.after_path):
        return {"status": "error", "message": f"后时相影像不存在: {args.after_path}"}

    if not os.path.splitext(args.output_path)[1]:
        args.output_path = args.output_path + ".png"

    # 坐标自动探测：有坐标且 GDAL 可用 → 坐标路径
    has_geo = GDAL_AVAILABLE and has_georeference(args.before_path)

    model_load_start = time.time()
    try:
        model = load_model(args)
        model_load_time = time.time() - model_load_start
    except Exception as e:
        return {"status": "error", "message": f"模型加载失败: {str(e)}"}

    transform = get_transform(args)

    process_start = time.time()
    try:
        if has_geo:
            result = _run_geo_path(args, model, transform)
        else:
            result = _run_plain_path(args, model, transform)
        process_time = time.time() - process_start
    except Exception as e:
        return {"status": "error", "message": f"影像处理失败: {str(e)}"}

    total_time = time.time() - start_time
    result["processing_time"] = {
        "total": total_time,
        "model_load": model_load_time,
        "process": process_time,
    }
    return result


def _run_plain_path(args, model, transform):
    """
    无坐标路径：block+patch 推理，PNG 掩码 + 四联图，无矢量

    入参: args/model/transform
    方法: process_large_image 推理 → _save_mask(PNG) → _save_visualization
    出参: 结果字典（status/message/output_path/quad_view_path）
    """
    pred_mask, pre_img, post_img = process_large_image(
        args.before_path, args.after_path, model, transform, args
    )

    mask_save_success = _save_mask(pred_mask, args)
    quad_view_path, visualize_success = _save_visualization(pre_img, post_img, pred_mask, args)

    final_status = "success" if mask_save_success and visualize_success else "error"
    final_message = "影像模型处理完成"
    if not mask_save_success:
        final_message += " (保存掩码失败)"
    if not visualize_success and args.save_visualization:
        final_message += " (保存可视化失败)"

    return {
        "status": final_status,
        "message": final_message,
        "output_path": args.output_path if mask_save_success else None,
        "quad_view_path": quad_view_path if visualize_success and args.save_visualization else None,
        "vector_files": [],
    }


def _run_geo_path(args, model, transform):
    """
    坐标路径：保留 geo 推理，GeoTIFF 掩码 + 四联图 + 矢量

    入参: args/model/transform
    方法: process_large_raster 推理 → save_geotiff_result → visualize → mask_to_lines + export_vector
    出参: 结果字典（含 vector_files）
    """
    pred_mask, pre_img, post_img, geo_transform, projection, src_ds = process_large_raster(
        args.before_path, args.after_path, model, transform, args
    )

    output_dir = os.path.dirname(args.output_path) or "."
    base_name = os.path.splitext(os.path.basename(args.output_path))[0]
    if not base_name:
        base_name = os.path.splitext(os.path.basename(args.before_path))[0] + "_result"

    results = {
        "status": "processing",
        "message": "",
        "output_path": None,
        "quad_view_path": None,
        "vector_files": [],
    }

    # GeoTIFF 掩膜
    mask_output_path = os.path.join(output_dir, f"{base_name}_mask.tif")
    if args.save_binary_mask:
        save_geotiff_result(pred_mask, mask_output_path, geo_transform, projection, src_ds)
        results["output_path"] = mask_output_path

    # 四联图
    if args.save_visualization:
        viz_output_path = os.path.join(output_dir, f"{base_name}_quadview.png")
        try:
            vis_pre, _, _, _ = read_geotiff(args.before_path)
            vis_post, _, _, _ = read_geotiff(args.after_path)
            visualize_results(vis_pre, vis_post, pred_mask, viz_output_path)
            results["quad_view_path"] = viz_output_path
        except Exception:
            pass

    # 矢量导出（仅当有有效投影）
    vector_files = []
    if projection and projection.strip() and (args.export_shapefile or args.export_geojson):
        vector_dir = os.path.join(output_dir, "vectors")
        os.makedirs(vector_dir, exist_ok=True)
        try:
            polygons, areas = mask_to_lines(
                pred_mask,
                min_area=args.min_polygon_area,
                simplify=True,
                simplify_tolerance=args.simplify_tolerance
            )
            if polygons:
                attributes = None
                if args.attribute_change_type or args.calculate_area:
                    attributes = []
                    for area in areas:
                        attr = {}
                        if args.attribute_change_type:
                            attr['change_type'] = 'change'
                        if args.calculate_area:
                            attr['area_m2'] = area
                        attributes.append(attr)

                if args.export_shapefile:
                    shp_path = os.path.join(vector_dir, f"{base_name}_changes.shp")
                    export_vector(polygons, areas, shp_path, geo_transform, projection,
                                  attributes, export_format='shp')
                    vector_files.append(shp_path)
                if args.export_geojson:
                    geojson_path = os.path.join(vector_dir, f"{base_name}_changes.geojson")
                    export_vector(polygons, areas, geojson_path, geo_transform, projection,
                                  attributes, export_format='geojson')
                    vector_files.append(geojson_path)
        except Exception:
            pass

    results["vector_files"] = vector_files
    results["status"] = "success"
    results["message"] = "影像模型处理完成（保留坐标系）"
    return results


def _save_mask(pred_mask, args):
    """
    保存 PNG 二值掩膜

    入参: pred_mask — 掩膜数组 / args — 含 output_path/save_binary_mask/raw_output
    出参: bool
    """
    if not args.save_binary_mask or pred_mask is None:
        return True if not args.save_binary_mask else False

    try:
        from PIL import Image
        save_data = (pred_mask * 255).astype(np.uint8) if args.raw_output else pred_mask
        Image.fromarray(save_data).save(args.output_path)
        return True
    except Exception:
        try:
            return bool(cv2.imwrite(args.output_path, pred_mask))
        except Exception:
            return False


def _save_visualization(pre_img, post_img, pred_mask, args):
    """
    保存四联图可视化

    入参: pre_img/post_img/pred_mask / args
    出参: (quad_view_path or None, success_flag)
    """
    if not args.save_visualization or pre_img is None or post_img is None or pred_mask is None:
        return None, True if not args.save_visualization else False

    quad_view_path = os.path.splitext(args.output_path)[0] + "_quadview.png"
    success = visualize_results(pre_img, post_img, pred_mask, quad_view_path,
                                is_raw_output=args.raw_output)
    return quad_view_path, success


def main():
    """命令行入口"""
    args = get_args()
    args.before_path = args.before_path.replace('\\', '/')
    args.after_path = args.after_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')
    process_and_save(args)


if __name__ == "__main__":
    main()
