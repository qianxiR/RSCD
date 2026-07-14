"""
RSCD Backend 训练模块

提供模型训练主循环，从零实现的 epoch 训练（含 loss/优化/指标/checkpoint 保存）。

入参: train_model(data_root, save_dir, epochs, batch_size, lr, ...)
方法: 构造 DataLoader → Trainer 模型 → Adam 优化器 → epoch 循环（训练+验证）→ checkpoint 保存
出参: Dict[str, Any]，含训练历史与最佳指标
"""
from .train_loop import train_model

__all__ = ["train_model"]
