#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练数据预处理脚本

将 CD-CD_20 数据集转换为 BCDDataset 期望的格式。

入参: 源目录 data/CD-CD_20（含 {train,val,test}/{t1,t2,t1_mask,t2_mask}）
方法:
  1. 由 t1_mask 与 t2_mask 的逐像素差异合成变化标签（变化=255, 背景=0）
  2. t1/t2 的 .tif 转为 .png（统一扩展名，避免 BCDDataset 按 label 文件名拼路径出错）
  3. 输出到 data/CD-CD_20_prepared/{split}/{t1,t2,label}/，文件名一致
出参: 无（产物写入磁盘）

做什么: 解决三类不兼容：
  (1) change_mask 全图非零（是语义分割图不是变化标签），故改用 |t1_mask - t2_mask| 合成真变化标签
  (2) 目录名是 t1_mask/t2_mask，需归一为 label
  (3) 扩展名 .tif vs .png 不一致，BCDDataset 用 label 文件名拼 t1/t2 路径会 FileNotFoundError
为什么: 训练必须有真实的二值变化标签，源数据的 change_mask 实测全像素非零（地物语义图）
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# 训练/验证/测试三个 split
SPLITS = ["train", "val", "test"]


def synthesize_change_mask(t1_mask_path: Path, t2_mask_path: Path, output_path: Path) -> dict:
    """
    由 t1_mask 与 t2_mask 的差异合成二值变化标签

    入参:
        t1_mask_path: 前时相地物语义 mask（RGB PNG，多类灰度值）
        t2_mask_path: 后时相地物语义 mask（同上）
        output_path: 输出二值变化标签路径（.png, mode='L'）
    方法: 两张 mask 转灰度 → 逐像素比较 → 不同则变化（255），相同则背景（0）
    出参: dict，含变化像素数/总像素数/变化比例

    做什么: 生成真正的"变化/不变化"二值标签
    为什么: 源数据的 change_mask 实测全像素非零（地物语义分割图，非变化标签），
            无法直接用于变化检测训练；两期地物 mask 的差异才是真实变化
    """
    t1_arr = np.array(Image.open(t1_mask_path).convert("L"))
    t2_arr = np.array(Image.open(t2_mask_path).convert("L"))

    # 逐像素比较：地物类别不同 = 变化
    changed = (t1_arr != t2_arr).astype(np.uint8) * 255
    Image.fromarray(changed, mode="L").save(output_path)

    total = changed.size
    changed_count = int(np.sum(changed > 0))
    return {"total": total, "changed": changed_count, "change_ratio": changed_count / total}


def convert_image_to_png(src_path: Path, output_path: Path) -> bool:
    """
    将任意格式图像转为 PNG（统一扩展名）

    入参:
        src_path: 源图像（.tif/.tiff/.png/...）
        output_path: 输出 PNG 路径
    方法: PIL 读取 → 转 RGB → 保存为 PNG
    出参: bool，成功 True

    做什么: 统一 t1/t2 为 .png 扩展名，与 label 文件名严格一致
    为什么: BCDDataset 用 label 目录的文件名拼 t1/t2 路径，扩展名不一致会找不到文件
    """
    img = Image.open(src_path).convert("RGB")
    img.save(output_path, format="PNG")
    return True


def process_split(src_root: Path, dst_root: Path, split: str) -> dict:
    """
    处理单个 split（train/val/test）

    入参:
        src_root: 源数据根（data/CD-CD_20）
        dst_root: 目标根（data/CD-CD_20_prepared）
        split: 'train'/'val'/'test'
    方法: 遍历 t1_mask 目录所有 .png → 与同名 t2_mask 合成变化标签到 label/ → 同名 t1/t2 转 PNG
    出参: dict，含处理统计（文件数、平均变化比例、跳过数）

    做什么: 把一个 split 的文件转换到 BCDDataset 兼容格式
    为什么: 保持 t1/t2/label 三目录同名同扩展名，BCDDataset 即可零改动加载
    """
    src_split_dir = src_root / split
    dst_split_dir = dst_root / split

    if not src_split_dir.exists():
        print(f"[跳过] split 目录不存在: {src_split_dir}")
        return {"processed": 0, "skipped": 0}

    # 以 t1_mask 目录的文件为基准（决定文件名）
    src_t1_mask_dir = src_split_dir / "t1_mask"
    if not src_t1_mask_dir.exists():
        print(f"[跳过] t1_mask 目录不存在: {src_t1_mask_dir}")
        return {"processed": 0, "skipped": 0}

    mask_files = sorted([f for f in src_t1_mask_dir.iterdir() if f.suffix.lower() == ".png"])
    print(f"[{split}] 发现 {len(mask_files)} 个 mask 文件")

    # 创建目标子目录
    for target_sub in ["t1", "t2", "label"]:
        (dst_split_dir / target_sub).mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    change_ratios = []

    for mask_file in mask_files:
        stem = mask_file.stem  # 如 train_00025

        # 在 t1/t2/t2_mask 目录查找同名文件
        src_t1 = _find_by_stem(src_split_dir / "t1", stem)
        src_t2 = _find_by_stem(src_split_dir / "t2", stem)
        src_t2_mask = _find_by_stem(src_split_dir / "t2_mask", stem)

        if src_t1 is None or src_t2 is None or src_t2_mask is None:
            missing = []
            if src_t1 is None: missing.append("t1")
            if src_t2 is None: missing.append("t2")
            if src_t2_mask is None: missing.append("t2_mask")
            print(f"  [警告] {stem}: 缺失 {','.join(missing)}，跳过")
            skipped += 1
            continue

        # 输出路径（统一 .png 扩展名）
        dst_label = dst_split_dir / "label" / f"{stem}.png"
        dst_t1 = dst_split_dir / "t1" / f"{stem}.png"
        dst_t2 = dst_split_dir / "t2" / f"{stem}.png"

        # 合成变化标签
        stats = synthesize_change_mask(mask_file, src_t2_mask, dst_label)
        change_ratios.append(stats["change_ratio"])

        # 转换 t1/t2
        convert_image_to_png(src_t1, dst_t1)
        convert_image_to_png(src_t2, dst_t2)

        processed += 1

    avg_ratio = sum(change_ratios) / len(change_ratios) if change_ratios else 0
    print(f"[{split}] 完成: 转换 {processed} 对，跳过 {skipped}，平均变化比例 {avg_ratio*100:.2f}%")
    return {"processed": processed, "skipped": skipped, "avg_change_ratio": avg_ratio}


def _find_by_stem(directory: Path, stem: str):
    """
    在目录中按文件名（不含扩展名）查找文件

    入参:
        directory: 搜索目录
        stem: 文件名（不含扩展名）
    方法: 遍历目录，返回第一个 stem 匹配的文件（支持 .tif/.tiff/.png 等）
    出参: Path 或 None
    """
    if not directory.exists():
        return None
    for f in directory.iterdir():
        if f.stem == stem and f.is_file():
            return f
    return None


def verify_dataset(dst_root: Path) -> bool:
    """
    验证预处理后数据集的一致性

    入参: dst_root — 预处理输出根目录
    方法: 遍历每个 split，确认 t1/t2/label 三目录文件名集合完全相同
    出参: bool，全部一致返回 True

    做什么: 数据完整性校验，确保 BCDDataset 能正确配对
    为什么: 训练时 t1/t2/label 错配会导致 label 与图像不对应，污染训练
    """
    print("\n=== 数据集一致性验证 ===")
    all_ok = True
    for split in SPLITS:
        split_dir = dst_root / split
        if not split_dir.exists():
            continue

        names = {}
        for sub in ["t1", "t2", "label"]:
            sub_dir = split_dir / sub
            if sub_dir.exists():
                names[sub] = set(f.name for f in sub_dir.iterdir() if f.is_file())
            else:
                names[sub] = set()

        t1_eq_t2 = names["t1"] == names["t2"]
        t1_eq_label = names["t1"] == names["label"]
        status = "✓" if (t1_eq_t2 and t1_eq_label) else "✗"
        print(f"{status} {split}: t1={len(names['t1'])}, t2={len(names['t2'])}, label={len(names['label'])}"
              f" | t1==t2: {t1_eq_t2}, t1==label: {t1_eq_label}")

        if not (t1_eq_t2 and t1_eq_label):
            all_ok = False
            # 显示差异
            only_in_t1 = names["t1"] - names["t2"]
            only_in_t2 = names["t2"] - names["t1"]
            if only_in_t1:
                print(f"    仅在 t1: {sorted(only_in_t1)[:3]}")
            if only_in_t2:
                print(f"    仅在 t2: {sorted(only_in_t2)[:3]}")

    return all_ok


def main():
    """
    命令行入口

    入参: --src / --dst 命令行参数
    方法: 遍历 3 个 split → process_split → verify_dataset
    出参: exit code 0=成功
    """
    parser = argparse.ArgumentParser(description="预处理 CD-CD_20 数据集为 BCDDataset 兼容格式")
    parser.add_argument("--src", type=str, default="data/CD-CD_20",
                        help="源数据根目录（含 train/val/test）")
    parser.add_argument("--dst", type=str, default="data/CD-CD_20_prepared",
                        help="输出目录（预处理后数据）")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"[错误] 源目录不存在: {src_root}")
        return 1

    print("=" * 60)
    print("CD-CD_20 训练数据预处理")
    print("=" * 60)
    print(f"源目录: {src_root}")
    print(f"目标目录: {dst_root}")
    print()

    # 清空目标目录（避免残留旧文件干扰）
    if dst_root.exists():
        print(f"清理旧的目标目录: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # 处理每个 split
    total_stats = {"processed": 0, "skipped": 0}
    for split in SPLITS:
        stats = process_split(src_root, dst_root, split)
        total_stats["processed"] += stats["processed"]
        total_stats["skipped"] += stats.get("skipped", 0)

    # 验证一致性
    ok = verify_dataset(dst_root)

    print("\n" + "=" * 60)
    print(f"预处理完成: 共转换 {total_stats['processed']} 对样本，跳过 {total_stats['skipped']} 对")
    print(f"验证结果: {'全部一致 ✓' if ok else '存在不一致 ✗'}")
    print(f"输出目录: {dst_root}")
    print("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
