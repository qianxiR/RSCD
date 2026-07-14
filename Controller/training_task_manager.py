"""
RSCD 控制层 - 训练任务调度

参照 task_manager.run_detection_task 的状态机模式，调度 Backend.training.train_model。
独立的 training_tasks 字典与 _training_lock 确保训练任务与检测任务解耦且单训练任务运行。

入参: run_training_task(task_id, data_root, save_dir, epochs, batch_size, lr, pretrained)
方法: pending → running → completed/failed，每 epoch 通过闭包更新 training_tasks 字典
出参: 无（状态写入 training_tasks 全局字典）
"""
import os
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

# 训练任务全局字典：task_id → 状态详情
# 与 detection_tasks 解耦，避免训练与检测的字段混淆
training_tasks: Dict[str, Dict[str, Any]] = {}

# 训练任务互斥锁：确保同一时刻只有一个训练任务运行（避免 GPU OOM）
# 训练是长任务且占用几乎全部显存，并发会导致 OOM 崩溃
_training_lock = threading.Lock()


def _update_progress(task_id: str, progress: Dict[str, Any]) -> None:
    """
    训练进度回调（线程安全地更新 training_tasks 字典）

    入参:
        task_id: 训练任务 ID
        progress: train_model 回传的进度字典（含 phase/epoch/train_loss/val_f1/log_line 等）
    方法: 按 phase 分类更新对应字段；log_line 追加到 logs 列表
    出参: 无

    做什么: 把 Backend 训练循环的进度实时反映到 training_tasks，供前端轮询读取
    为什么: 训练循环在工作线程运行，前端在主线程轮询，字典是两者的共享状态
    """
    task = training_tasks.get(task_id)
    if task is None:
        return

    phase = progress.get("phase")

    if phase == "init":
        task["status"] = "running"
        task["message"] = progress.get("message", "训练已启动")

    if "log_line" in progress:
        task["logs"].append(progress["log_line"])
        # 限制日志条数，避免内存无限增长（保留最近 500 条）
        if len(task["logs"]) > 500:
            task["logs"] = task["logs"][-500:]

    if phase == "epoch":
        task["current_epoch"] = progress.get("epoch", task["current_epoch"])
        task["train_loss"] = progress.get("train_loss")
        task["val_f1"] = progress.get("val_f1")
        task["val_iou"] = progress.get("val_iou")
        task["best_f1"] = progress.get("best_f1")
        task["message"] = (f"Epoch {progress.get('epoch')}/{progress.get('total_epochs')} | "
                           f"loss={progress.get('train_loss', 0):.4f} | "
                           f"F1={progress.get('val_f1', 0):.4f}")


def run_training_task(
    task_id: str,
    data_root: str,
    save_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    pretrained: Optional[str],
) -> None:
    """
    训练任务执行函数（由 BackgroundTasks 调度）

    入参:
        task_id: 任务唯一标识
        data_root: 训练数据根目录
        save_dir: 模型保存目录
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        pretrained: X3D 预训练权重路径
    方法: 获取 _training_lock → 调 train_model → 进度回调更新字典 → finally 设置终态
    出参: 无（结果写入 training_tasks）

    做什么: 训练任务的状态机编排（pending→running→completed/failed）
    为什么: 与 run_detection_task 对称，但用独立字典与锁，确保训练/检测互不干扰
    """
    # 延迟导入避免循环依赖（Backend.training 会导入 Backend.network 等）
    from Backend.training import train_model

    task = training_tasks.get(task_id)
    if task is None:
        return

    try:
        # 单训练任务锁：训练占用全部 GPU 资源，并发必 OOM
        if not _training_lock.acquire(blocking=False):
            task["status"] = "failed"
            task["message"] = "已有训练任务在运行，请等待完成后再启动新任务"
            task["error"] = "training_lock_busy"
            task["end_time"] = datetime.now().isoformat()
            return

        try:
            task["status"] = "running"
            task["message"] = "训练任务已启动"

            result = train_model(
                data_root=data_root,
                save_dir=save_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                pretrained=pretrained,
                progress_callback=lambda p: _update_progress(task_id, p),
            )

            # 训练完成
            task["status"] = "completed"
            task["message"] = (f"训练完成: 最佳 F1={result['best_f1']:.4f}, "
                               f"总耗时 {result['total_time']:.1f}s")
            task["best_f1"] = result["best_f1"]
            task["current_epoch"] = result["final_epoch"]

        finally:
            _training_lock.release()

    except Exception as e:
        task["status"] = "failed"
        task["message"] = f"训练失败: {str(e)}"
        task["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    finally:
        task["end_time"] = datetime.now().isoformat()
