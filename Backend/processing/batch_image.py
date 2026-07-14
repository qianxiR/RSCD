#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 批量影像变化检测推理（统一入口）
#
# 遍历前后时相目录中的同名影像对，按每对是否带坐标自动选择坐标/无坐标路径，
# 逐对调用 process_and_save 落盘。坐标路径产物含矢量，最后合并所有矢量。

"""
批量影像变化检测入口

入参: process_and_save(args) — args 需含 before_path/after_path/output_path/checkpoint/device 等
方法: 遍历目录配对 → 模型加载 → 逐对 geo 探测 + 推理 + 落盘 → 矢量合并
出参: Dict[str, Any] — status/message/output_path/result_dir/mask_dir/processing_time/details
"""
import os
import argparse
import time
import warnings
import glob as glob_module

import numpy as np
import torch

from Backend.processing.common import (
    load_model,
    has_georeference,
)


# 矢量合并依赖 GDAL 栈，不可用时跳过合并
try:
    import geopandas as gpd
    import pandas as pd
    from osgeo import gdal
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False


def get_args():
    """
    构建批量推理参数命名空间（CLI）

    入参: 无（命令行参数）
    方法: 仅 required 三个路径走 argparse，其余默认值在解析后手动挂载
    出参: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='批量影像变化检测推理')
    parser.add_argument('--before_path', type=str, required=True, help='前时相影像目录')
    parser.add_argument('--after_path', type=str, required=True, help='后时相影像目录')
    parser.add_argument('--output_path', type=str, default='results/batch_results', help='输出主目录')

    args = parser.parse_args()

    args.checkpoint = "checkpoint/best_model.pth"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 统一支持普通图像与栅格影像扩展名
    args.file_ext = '.png,.jpg,.jpeg,.tif,.tiff'
    args.in_height = 256
    args.in_width = 256
    args.num_perception_frame = 1
    args.num_class = 1
    args.save_result = True
    args.save_binary_mask = True
    args.save_visualization = True
    args.patch_size = 256
    args.block_size = 512
    args.stride_ratio = 0.5
    args.batch_size = 16
    args.overlap_weights = False
    args.raw_output = False
    args.dataset = 'LEVIR-CD'
    args.auto_memory = True
    args.max_images = 0
    args.quiet = True
    args.model_arch = 'siam_unet'
    args.pretrained = None
    args.max_patch_size = 1024
    args.min_patch_size = 256
    args.auto_patch_divisor = 16
    args.ignore_geo = False
    args.band_indices = '1,2,3'
    args.warp_projection = None
    args.export_shapefile = True
    args.export_geojson = True
    args.min_polygon_area = 100.0
    args.simplify_tolerance = 0.5
    args.attribute_change_type = True
    args.calculate_area = True
    args.merge_vectors = True
    args.merged_file_name = 'merged_changes'

    args.pre_dir = args.before_path
    args.post_dir = args.after_path
    args.output_dir = args.output_path

    return args


def get_image_pairs(args):
    """
    按文件名匹配前后时相目录中的影像对

    入参: args — 需含 before_path/after_path/file_ext
    方法: 遍历前时相目录 → 按同名规则在后时相目录查找匹配
    出参: List[Tuple(pre_file, post_file, filename_without_ext)]
    """
    pre_dir = os.path.normpath(args.before_path)
    post_dir = os.path.normpath(args.after_path)

    if not os.path.exists(pre_dir) or not os.path.exists(post_dir):
        raise ValueError(f"前后时相目录必须存在: {pre_dir}, {post_dir}")

    file_extensions = args.file_ext.split(',')

    pre_files = []
    for ext in file_extensions:
        pre_files.extend(glob_module.glob(os.path.normpath(os.path.join(pre_dir, f'*{ext}'))))
    pre_files = sorted(set(pre_files))

    image_pairs = []
    no_match_count = 0
    for pre_file in pre_files:
        filename = os.path.basename(pre_file)
        filename_without_ext, file_ext = os.path.splitext(filename)
        post_file = os.path.normpath(os.path.join(post_dir, f"{filename_without_ext}{file_ext}"))

        if os.path.exists(post_file):
            image_pairs.append((pre_file, post_file, filename_without_ext))
        else:
            no_match_count += 1

    if no_match_count > 0:
        print(f"共有 {no_match_count} 个前时相影像没有找到对应的后时相影像")

    return image_pairs


def merge_vector_files(input_dir, output_file, file_format='shp', pattern='*'):
    """
    合并多个矢量文件为一个

    入参:
        input_dir: 矢量文件所在目录
        output_file: 合并后文件名（不含扩展名）
        file_format: 'shp' 或 'geojson'
        pattern: 文件匹配模式
    方法: 读入所有匹配文件 → pd.concat 合并 → 写到 input_dir/merged/ 子目录
    出参: bool
    """
    if not GDAL_AVAILABLE or not os.path.exists(input_dir):
        return False

    merged_output_dir = os.path.join(input_dir, 'merged')
    os.makedirs(merged_output_dir, exist_ok=True)

    ext = '.shp' if file_format.lower() == 'shp' else '.geojson'
    final_output_path = os.path.join(merged_output_dir, f"{output_file}{ext}")

    vector_files = glob_module.glob(os.path.join(input_dir, pattern))
    if not vector_files:
        return False

    gdfs = []
    crs = None
    for file_path in vector_files:
        try:
            gdf = gpd.read_file(file_path)
            if crs is None and gdf.crs is not None:
                crs = gdf.crs
            gdf['source_file'] = os.path.basename(file_path)
            gdfs.append(gdf)
        except Exception:
            continue

    if not gdfs:
        return False

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    if crs is not None:
        merged_gdf.crs = crs

    try:
        if file_format.lower() == 'shp':
            merged_gdf.to_file(final_output_path)
        else:
            merged_gdf.to_file(final_output_path, driver='GeoJSON')
        return True
    except Exception:
        return False


def process_and_save(args):
    """
    对外暴露的批量处理入口（每对自动探测坐标）

    入参: args — Namespace，含 before_path/after_path/output_path/checkpoint/device 等
    方法: 路径规范化 → 输出目录创建 → 文件配对 → 模型加载 → 逐对调用 single_image 推理 → 汇总
    出参: Dict[str, Any] — status/message/output_path/result_dir/mask_dir/processing_time/details

    做什么: 编排批量推理，复用 single_image.process_and_save 处理单对
    为什么: 单对推理逻辑（坐标探测+分支+落盘）已在 single_image 统一实现，批处理只负责循环与汇总
    """
    from Backend.processing.single_image import process_and_save as process_single

    start_time_total = time.time()

    args.before_path = args.before_path.replace('\\', '/')
    args.after_path = args.after_path.replace('\\', '/')
    args.output_path = args.output_path.replace('\\', '/')

    if not os.path.isabs(args.output_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        args.output_path = os.path.normpath(os.path.join(project_root, args.output_path))

    main_output_dir = args.output_path
    os.makedirs(main_output_dir, exist_ok=True)
    result_dir = os.path.join(main_output_dir, 'result')
    mask_dir = os.path.join(main_output_dir, 'masks')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    args.pre_dir = args.before_path
    args.post_dir = args.after_path
    args.output_dir = args.output_path

    image_pairs = get_image_pairs(args)
    if len(image_pairs) == 0:
        return _empty_result(args, "未找到匹配的影像对，请确认目录结构和文件扩展名是否正确")

    if args.max_images > 0 and args.max_images < len(image_pairs):
        image_pairs = image_pairs[:args.max_images]

    if torch.cuda.is_available() and 'cuda' in str(args.device):
        args.device = torch.device(str(args.device))
    else:
        args.device = torch.device('cpu')

    start_time_model = time.time()
    try:
        load_model(args)
        model_load_time = time.time() - start_time_model
    except Exception as e:
        return {
            "status": "failed",
            "message": f"模型加载失败: {str(e)}",
            "output_path": args.output_path,
            "processing_time": {"total": time.time() - start_time_total,
                                "model_load": time.time() - start_time_model},
            "details": []
        }

    total_process_time = 0
    success_count = 0
    failed_count = 0
    all_vector_files = []
    details = []

    for pre_img_path, post_img_path, filename in image_pairs:
        item_detail = _process_one_pair(
            pre_img_path, post_img_path, filename, args, process_single,
            result_dir, mask_dir
        )
        details.append(item_detail)

        if item_detail["status"] == "success":
            success_count += 1
            total_process_time += item_detail["processing_time"]
            all_vector_files.extend(item_detail.get("vector_files", []))
        else:
            failed_count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 合并矢量
    merged_vector_files = []
    if getattr(args, 'merge_vectors', False) and all_vector_files and GDAL_AVAILABLE:
        vector_output_dir = os.path.join(args.output_path, 'vectors')
        for fmt in ('shp', 'geojson'):
            merge_vector_files(vector_output_dir, args.merged_file_name, fmt, f'*.{fmt}')

    total_elapsed = time.time() - start_time_total

    return {
        "status": "success" if success_count > 0 else "failed",
        "message": f"批量处理完成! 成功: {success_count}, 失败: {failed_count}, 总数: {len(image_pairs)}",
        "output_path": args.output_path,
        "result_dir": result_dir,
        "mask_dir": mask_dir,
        "vector_files": all_vector_files,
        "merged_vector_files": merged_vector_files,
        "processing_time": {
            "total": round(total_elapsed, 2),
            "model_load": round(model_load_time, 2),
            "processing": round(total_process_time, 2),
            "avg_per_image": round(total_process_time / max(success_count, 1), 2)
        },
        "details": details
    }


def _process_one_pair(pre_img_path, post_img_path, filename, args, process_single, result_dir, mask_dir):
    """
    处理单对影像：构造单对 args → 调 single_image.process_and_save → 汇总产物路径

    入参: 路径对/文件名/批量 args/process_single 函数/输出目录
    方法: 派生单对 output_path（无坐标→masks/png，有坐标→masks/tif，由 single_image 内部决定）
          → 调 process_single → 收集 output_path/quad_view/vector_files
    出参: Dict，单对处理结果
    """
    # 无坐标默认输出 PNG 到 masks；坐标路径由 single_image 内部生成 *_mask.tif
    single_output = os.path.join(mask_dir, f"{filename}_result")

    pair_args = argparse.Namespace(**vars(args))
    pair_args.before_path = pre_img_path
    pair_args.after_path = post_img_path
    pair_args.output_path = single_output

    start_time_process = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = process_single(pair_args)

        if result.get("status") not in ("success", "completed"):
            return _failed_detail(pre_img_path, post_img_path,
                                  result.get("message", "处理返回失败状态"), start_time_process)

        # 四联图移动到 result 目录统一管理
        result_path = None
        if result.get("quad_view_path") and os.path.exists(result["quad_view_path"]):
            target = os.path.join(result_dir, f"{filename}_quadview.png")
            try:
                import shutil
                shutil.move(result["quad_view_path"], target)
                result_path = target
            except Exception:
                result_path = result["quad_view_path"]

        total_item_time = time.time() - start_time_process

        return {
            "pre_img_path": pre_img_path,
            "post_img_path": post_img_path,
            "status": "success",
            "mask_path": result.get("output_path"),
            "result_path": result_path,
            "vector_files": result.get("vector_files", []),
            "processing_time": total_item_time
        }

    except Exception as e:
        return _failed_detail(pre_img_path, post_img_path, f"处理出错: {str(e)}", start_time_process)


def _empty_result(args, message):
    """构造空结果（无影像对时返回）"""
    return {
        "status": "failed",
        "message": message,
        "output_path": args.output_path,
        "processing_time": {"total": 0},
        "details": []
    }


def _failed_detail(pre_img_path, post_img_path, message, start_time):
    """构造失败明细"""
    return {
        "pre_img_path": pre_img_path,
        "post_img_path": post_img_path,
        "status": "failed",
        "message": message,
        "result_path": None,
        "mask_path": None,
        "vector_files": [],
        "processing_time": time.time() - start_time
    }


def main():
    """命令行入口"""
    args = get_args()
    args.output_dir = args.output_path
    process_and_save(args)


if __name__ == "__main__":
    main()
