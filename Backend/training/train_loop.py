#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练主循环

从零实现的 X3D 变化检测模型训练流程：数据加载 → 模型 → 优化器 → epoch 循环 → 验证 → checkpoint。

入参: train_model(data_root, save_dir, epochs, batch_size, lr, device, pretrained, progress_callback)
方法: 每个 epoch 训练一轮 + 验证一轮，通过 progress_callback 实时回传 loss/F1 等指标
出参: Dict[str, Any]，含 best_f1 / train_history / val_history
"""
import os
import time
import argparse
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from Backend.data.dataset import BCDDataset
from Backend.data.transforms import BCDTransforms
from Backend.network.encoder import Trainer
from Backend.network.network_utils import BCEDiceLoss, clip_gradient
from Backend.evaluation.metrics import ConfuseMatrixMeter, AverageMeter


def _build_args(data_root: str, epochs: int, batch_size: int, lr: float,
                pretrained: Optional[str]) -> argparse.Namespace:
    """
    构造训练用 Namespace

    入参: 训练超参
    方法: 组装 Trainer/BCDDataset/BCDTransforms 所需的全部字段
    出参: argparse.Namespace
    """
    args = argparse.Namespace()
    # 数据/模型尺寸
    args.in_height = 256
    args.in_width = 256
    args.dataset = "LEVIR-CD"
    # 模型结构
    args.num_class = 1
    args.num_perception_frame = 1
    args.pretrained = pretrained  # X3D 主干预训练权重路径，None 则随机初始化
    # 训练超参
    args.max_epochs = epochs
    args.lr = lr
    args.lr_mode = "step"
    args.step_loss = max(epochs // 3, 1)  # 每 1/3 总 epoch 衰减一次
    args.lr_factor = 0.5
    args.grad_clip = 5.0
    args.resume = None
    args.log_file = None
    # 归一化（与推理保持一致）
    args.normalize_mean = [0.5] * 6
    args.normalize_std = [0.5] * 6
    return args


def _save_checkpoint(model: Trainer, optimizer: torch.optim.Optimizer,
                     epoch: int, metrics: Dict[str, float], save_dir: str) -> None:
    """
    保存完整 checkpoint（含 optimizer，可续训）

    入参: 模型/优化器/epoch/指标/save_dir
    方法: torch.save 到 save_dir/checkpoint.pth.tar，字段对齐历史 checkpoint 约定
    出参: 无

    做什么: 保存可恢复训练的完整状态
    为什么: 训练中断后可从此文件恢复；字段结构与 checkpoint.pth.tar 一致以便推理加载
    """
    torch.save({
        "epoch": epoch,
        "arch": str(model),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_train": metrics.get("train_loss", 0),
        "loss_val": metrics.get("val_loss", 0),
        "F_train": metrics.get("train_f1", 0),
        "F_val": metrics.get("val_f1", 0),
        "lr": optimizer.param_groups[0]["lr"],
    }, os.path.join(save_dir, "checkpoint.pth.tar"))


def _save_best_model(model: Trainer, save_dir: str) -> None:
    """
    保存最佳模型（裸 state_dict，与推理默认加载的 best_model.pth 格式一致）

    入参: 模型/save_dir
    方法: torch.save(model.state_dict()) 到 save_dir/best_model.pth
    出参: 无

    做什么: 仅保存权重（不含 optimizer），供推理直接加载
    为什么: 推理只需 state_dict，裸字典体积更小（无 optimizer 状态约省 30%）
    """
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))


def _validate_one_epoch(model: Trainer, val_loader: DataLoader, device: str,
                        criterion, val_meter: ConfuseMatrixMeter) -> Dict[str, float]:
    """
    验证一个 epoch

    入参: 模型/验证加载器/设备/损失函数/混淆矩阵计数器
    方法: model.eval() → 遍历验证集 → 累加 loss 与混淆矩阵 → 返回指标
    出参: Dict，含 val_loss/val_f1/val_iou/val_kappa
    """
    model.eval()
    val_meter.clear()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for img, label in val_loader:
            # img: [B,6,H,W]，切分为 pre [B,3,H,W] 和 post [B,3,H,W]
            pre = img[:, 0:3].to(device)
            post = img[:, 3:6].to(device)
            label = label.float().to(device)

            pred = model.update_bcd(pre, post)  # [B,1,H,W] 已 sigmoid
            loss = criterion(pred, label)
            loss_meter.update(loss.item(), pre.size(0))

            # 二值化预测，更新混淆矩阵
            pred_bin = (pred > 0.5).cpu().numpy().squeeze().astype(int)
            gt = label.cpu().numpy().squeeze().astype(int)
            val_meter.update_cm(pred_bin, gt)

    scores = val_meter.get_scores()
    return {
        "val_loss": loss_meter.avg,
        "val_f1": scores.get("F1", 0),
        "val_iou": scores.get("IoU", 0),
        "val_kappa": scores.get("Kappa", 0),
    }


def train_model(
    data_root: str,
    save_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    pretrained: Optional[str] = "checkpoint/X3D_L.pyth",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    模型训练主入口

    入参:
        data_root: 数据根目录（需含 {train,val}/{t1,t2,label} 子目录）
        save_dir: checkpoint 保存目录（best_model.pth / checkpoint.pth.tar）
        epochs: 训练轮数
        batch_size: 批大小
        lr: 初始学习率
        device: 训练设备（cuda:0 / cpu）
        pretrained: X3D 主干预训练权重路径，None 则随机初始化
        progress_callback: 每个 epoch 调用一次，回传 {epoch, train_loss, val_f1, log_line, ...}
    方法: 构造数据/模型/优化器 → epoch 循环（训练+验证）→ 按最佳 F1 保存 best_model
    出参: Dict，含 best_f1 / final_epoch / train_history / val_history / save_dir
    """
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    # 1. 构造 args 与数据加载
    args = _build_args(data_root, epochs, batch_size, lr, pretrained)
    train_transform, val_transform = BCDTransforms.get_transform_pipelines(args)

    train_set = BCDDataset(data_root, "train", transform=train_transform)
    val_set = BCDDataset(data_root, "val", transform=val_transform)

    if len(train_set) == 0:
        raise ValueError(f"训练集为空，请检查 {data_root}/train/{{t1,t2,label}} 目录")
    if len(val_set) == 0:
        raise ValueError(f"验证集为空，请检查 {data_root}/val/{{t1,t2,label}} 目录")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # 2. 模型/优化器/损失/指标
    model = Trainer(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss
    val_meter = ConfuseMatrixMeter(n_class=2)

    if progress_callback:
        progress_callback({
            "phase": "init",
            "message": f"训练初始化完成: 训练集 {len(train_set)} 对, 验证集 {len(val_set)} 对, "
                       f"设备 {device}, batch_size {batch_size}, epochs {epochs}",
            "log_line": f"[初始化] 训练集 {len(train_set)} 对 | 验证集 {len(val_set)} 对 | 设备 {device}",
        })

    # 3. 训练循环
    best_f1 = 0.0
    train_history = []
    val_history = []

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        loss_meter = AverageMeter()

        for img, label in train_loader:
            # img: [B,6,H,W]，切分为 pre 和 post
            pre = img[:, 0:3].to(device)
            post = img[:, 3:6].to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            pred = model.update_bcd(pre, post)  # [B,1,H,W] 已 sigmoid
            loss = criterion(pred, label)
            loss.backward()
            clip_gradient(optimizer, args.grad_clip)
            optimizer.step()

            loss_meter.update(loss.item(), pre.size(0))

        # 4. 验证
        val_metrics = _validate_one_epoch(model, val_loader, device, criterion, val_meter)
        epoch_time = time.time() - epoch_start

        # 记录历史
        train_history.append({"epoch": epoch + 1, "loss": loss_meter.avg})
        val_history.append({"epoch": epoch + 1, **val_metrics})

        # 5. checkpoint 保存（每 epoch 覆盖 checkpoint.pth.tar）
        metrics_combined = {"train_loss": loss_meter.avg, **val_metrics}
        _save_checkpoint(model, optimizer, epoch + 1, metrics_combined, save_dir)

        # 6. 最佳模型保存
        if val_metrics["val_f1"] > best_f1:
            best_f1 = val_metrics["val_f1"]
            _save_best_model(model, save_dir)

        # 7. 进度回传（关键：通过闭包更新 training_tasks 字典）
        if progress_callback:
            log_line = (f"[Epoch {epoch+1}/{epochs}] loss={loss_meter.avg:.4f} | "
                        f"val_F1={val_metrics['val_f1']:.4f} | IoU={val_metrics['val_iou']:.4f} | "
                        f"Kappa={val_metrics['val_kappa']:.4f} | best_F1={best_f1:.4f} | "
                        f"耗时 {epoch_time:.1f}s")
            progress_callback({
                "phase": "epoch",
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "train_loss": loss_meter.avg,
                "val_f1": val_metrics["val_f1"],
                "val_iou": val_metrics["val_iou"],
                "best_f1": best_f1,
                "epoch_time": epoch_time,
                "log_line": log_line,
            })

    total_time = time.time() - start_time
    if progress_callback:
        progress_callback({
            "phase": "done",
            "log_line": f"[训练完成] 总耗时 {total_time:.1f}s | 最佳 F1={best_f1:.4f} | 模型已保存到 {save_dir}",
        })

    return {
        "status": "success",
        "best_f1": best_f1,
        "final_epoch": epochs,
        "train_history": train_history,
        "val_history": val_history,
        "save_dir": save_dir,
        "total_time": total_time,
    }
