#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型训练对话框

复用现有批量处理对话框的视觉风格（目录选择行 + QListWidget 日志 + 居中按钮），
提供训练目录选择与实时训练进度显示。

入参: TrainingDialog(navigation_functions, parent)
方法: 目录选择 → 默认配置启动训练 → 轮询进度 → 日志区实时显示
出参: 无（训练产物保存到指定目录，UI 显示进度）
"""
import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QWidget, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QObject, QThread, Signal, QMetaObject, QCoreApplication, Q_ARG, Slot
from PySide6.QtGui import QFont

from Frontend.theme import ThemeManager
from Frontend.views.common.qt_logging import ThreadSafeLogMixin
from Controller.local_service import start_training, get_training_progress, check_connection


class TrainingDialog(QDialog, ThreadSafeLogMixin):
    """
    模型训练对话框

    入参: navigation_functions（提供 is_dark_theme / theme_changed_signal）, parent
    方法: 初始化 UI → 连接信号 → 启动训练时创建 Worker 线程轮询进度
    出参: 无（通过日志区显示训练过程）
    """

    def __init__(self, navigation_functions, parent=None):
        """
        初始化训练对话框

        入参:
            navigation_functions: 导航函数对象，含 is_dark_theme / theme_changed_signal
            parent: 父窗口
        """
        super().__init__(parent)
        self.navigation_functions = navigation_functions
        self.is_dark_theme = navigation_functions.is_dark_theme

        # 训练参数状态
        self.data_root_dir = ""    # 训练数据根目录
        self.save_dir = ""         # 模型保存目录

        # 工作线程
        self.train_thread = None
        self.train_worker = None
        self.is_shutting_down = False

        # 初始化 UI
        self.init_ui()

        # 连接主题变化信号
        if hasattr(navigation_functions, "theme_changed_signal"):
            navigation_functions.theme_changed_signal.connect(self.on_theme_changed)

    def init_ui(self):
        """
        初始化用户界面

        入参: 无
        方法: 构建垂直布局：标题 → 目录选择行 × 2 → 状态行 → 日志区 → 按钮
        出参: 无

        做什么: 完全复用 batch_dialog 的视觉风格（QLabel+QLabel+QPushButton 目录选择行）
        为什么: 保持系统 UI 一致性，用户零学习成本
        """
        self.setWindowTitle("模型训练")
        self.resize(800, 600)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # 标题
        title_label = QLabel("模型训练")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei UI", 12, QFont.Bold))
        title_label.setStyleSheet(f"color: {ThemeManager.get_colors(self.is_dark_theme)['text']};")
        main_layout.addWidget(title_label)

        # 训练数据目录选择行
        data_layout = QHBoxLayout()
        data_label = QLabel("训练数据目录:")
        self.data_dir_label = QLabel("未选择")
        self.data_dir_button = QPushButton("浏览...")
        self.data_dir_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.data_dir_button.setFixedSize(80, 32)
        self.data_dir_button.setFont(QFont("Microsoft YaHei UI", 9))
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_dir_label, 1)
        data_layout.addWidget(self.data_dir_button)
        main_layout.addLayout(data_layout)

        # 模型保存目录选择行
        save_layout = QHBoxLayout()
        save_label = QLabel("模型保存目录:")
        self.save_dir_label = QLabel("未选择")
        self.save_dir_button = QPushButton("浏览...")
        self.save_dir_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.save_dir_button.setFixedSize(80, 32)
        self.save_dir_button.setFont(QFont("Microsoft YaHei UI", 9))
        save_layout.addWidget(save_label)
        save_layout.addWidget(self.save_dir_label, 1)
        save_layout.addWidget(self.save_dir_button)
        main_layout.addLayout(save_layout)

        # 提示信息
        tip_label = QLabel("注意: 训练数据需包含 train/val 子目录，各含 t1/t2/label 三类同名文件")
        tip_label.setStyleSheet(f"color: {ThemeManager.get_colors(self.is_dark_theme)['info_icon']};")
        tip_label.setWordWrap(True)
        main_layout.addWidget(tip_label)

        # 状态行（实时显示当前 epoch/loss/F1）
        self.status_label = QLabel("等待开始训练...")
        self.status_label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # 日志标签
        log_label = QLabel("训练日志")
        log_label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Bold))
        main_layout.addWidget(log_label)

        # 日志列表（复用 ThreadSafeLogMixin，需设置 log_widget）
        self.log_widget = QListWidget()
        self.log_widget.setStyleSheet(ThemeManager.get_list_widget_style(self.is_dark_theme))
        self.log_widget.setMinimumHeight(250)
        main_layout.addWidget(self.log_widget)

        # 按钮容器（透明背景，居中放置）
        button_container = QWidget()
        button_container.setStyleSheet(ThemeManager.get_transparent_container_style())
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)

        self.start_button = QPushButton("开始训练")
        self.start_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.start_button.setFixedSize(120, 36)
        self.start_button.setFont(QFont("Microsoft YaHei UI", 9, QFont.Bold))

        self.stop_button = QPushButton("停止")
        self.stop_button.setStyleSheet(ThemeManager.get_dialog_button_style(self.is_dark_theme))
        self.stop_button.setFixedSize(80, 36)
        self.stop_button.setFont(QFont("Microsoft YaHei UI", 9))
        self.stop_button.setEnabled(False)

        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        main_layout.addWidget(button_container)

        # 应用主题
        self._apply_theme()

        # 连接信号
        self.data_dir_button.clicked.connect(self.select_data_dir)
        self.save_dir_button.clicked.connect(self.select_save_dir)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)

    def _apply_theme(self):
        """应用主题样式到对话框背景"""
        colors = ThemeManager.get_colors(self.is_dark_theme)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {colors['background']};
                color: {colors['text']};
            }}
            QLabel {{
                color: {colors['text']};
                background-color: transparent;
            }}
        """)

    def on_theme_changed(self):
        """主题变化时刷新样式"""
        self.is_dark_theme = self.navigation_functions.is_dark_theme
        self._apply_theme()
        # 刷新按钮样式
        btn_style = ThemeManager.get_dialog_button_style(self.is_dark_theme)
        self.data_dir_button.setStyleSheet(btn_style)
        self.save_dir_button.setStyleSheet(btn_style)
        self.start_button.setStyleSheet(btn_style)
        self.stop_button.setStyleSheet(btn_style)
        self.log_widget.setStyleSheet(ThemeManager.get_list_widget_style(self.is_dark_theme))

    def select_data_dir(self):
        """选择训练数据根目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择训练数据目录", "")
        if dir_path:
            self.data_root_dir = dir_path
            self.data_dir_label.setText(os.path.basename(dir_path))
            self.data_dir_label.setToolTip(dir_path)

    def select_save_dir(self):
        """选择模型保存目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型保存目录", "")
        if dir_path:
            self.save_dir = dir_path
            self.save_dir_label.setText(os.path.basename(dir_path))
            self.save_dir_label.setToolTip(dir_path)

    def start_training(self):
        """
        启动训练任务

        入参: 无（从 UI 控件读取参数）
        方法: 校验输入 → 创建 Worker 线程 → 连接信号 → start
        出参: 无

        做什么: 提交训练任务到后端并在工作线程中轮询进度
        为什么: 训练长达数小时，必须放工作线程避免阻塞 UI
        """
        # 校验输入
        if not self.data_root_dir:
            QMessageBox.warning(self, "提示", "请先选择训练数据目录")
            return
        if not self.save_dir:
            QMessageBox.warning(self, "提示", "请先选择模型保存目录")
            return

        # 校验训练数据结构
        train_dir = os.path.join(self.data_root_dir, "train")
        val_dir = os.path.join(self.data_root_dir, "val")
        if not os.path.isdir(train_dir):
            QMessageBox.warning(self, "提示", f"训练数据缺少 train 子目录:\n{train_dir}")
            return
        if not os.path.isdir(val_dir):
            QMessageBox.warning(self, "提示", f"训练数据缺少 val 子目录:\n{val_dir}")
            return

        # 检查 API 连接
        if not check_connection():
            QMessageBox.warning(self, "提示", "无法连接 API 服务，请确认后端已启动")
            return

        # 清空日志
        self.log_widget.clear()
        self._add_log("正在提交训练任务...")

        # 禁用按钮
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # 创建工作线程
        self.train_thread = QThread()
        self.train_worker = TrainingWorker(
            data_root=self.data_root_dir,
            save_dir=self.save_dir,
        )
        self.train_worker.moveToThread(self.train_thread)

        # 连接信号
        self.train_worker.signals.log.connect(self._add_log)
        self.train_worker.signals.progress.connect(self._on_progress)
        self.train_worker.signals.finished.connect(self._on_finished)
        self.train_worker.signals.error.connect(self._on_error)
        self.train_thread.started.connect(self.train_worker.run)

        self.train_thread.start()

    def _on_progress(self, progress):
        """
        更新状态行（Worker 回传的进度信号触发）

        入参: progress dict，含 current_epoch/total_epochs/train_loss/val_f1/best_f1
        方法: 拼接状态文本，更新 status_label
        出参: 无
        """
        epoch = progress.get("current_epoch", 0)
        total = progress.get("total_epochs", 0)
        loss = progress.get("train_loss")
        f1 = progress.get("val_f1")
        best = progress.get("best_f1")

        parts = [f"Epoch {epoch}/{total}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if f1 is not None:
            parts.append(f"F1={f1:.4f}")
        if best is not None:
            parts.append(f"best={best:.4f}")

        self.status_label.setText(" | ".join(parts))

    def _on_finished(self, result):
        """
        训练完成回调

        入参: result dict（训练最终结果）
        方法: 日志显示完成信息 → 重新启用开始按钮
        出参: 无
        """
        self._add_log(f"训练完成: {result.get('message', '已完成')}")
        self.status_label.setText("训练完成")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._cleanup_worker()

        QMessageBox.information(self, "训练完成", f"训练已完成\n{result.get('message', '')}")

    def _on_error(self, error_msg):
        """
        训练错误回调

        入参: error_msg str
        方法: 日志显示错误 → 重新启用开始按钮
        出参: 无
        """
        self._add_log(f"[错误] {error_msg}")
        self.status_label.setText("训练失败")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._cleanup_worker()

    def stop_training(self):
        """停止训练（终止工作线程，后端训练任务继续运行直至完成或失败）"""
        self._add_log("[用户操作] 正在停止训练任务监控（后端训练将继续执行）...")
        self._cleanup_worker()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("已停止监控")

    def _cleanup_worker(self):
        """清理工作线程与 Worker 资源"""
        if self.train_worker is not None:
            try:
                self.train_worker.signals.log.disconnect(self._add_log)
            except RuntimeError:
                pass
            try:
                self.train_worker.deleteLater()
            except RuntimeError:
                pass
            self.train_worker = None

        if self.train_thread is not None:
            try:
                self.train_thread.quit()
                self.train_thread.wait(3000)
            except RuntimeError:
                pass
            try:
                self.train_thread.deleteLater()
            except RuntimeError:
                pass
            self.train_thread = None

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.is_shutting_down = True
        self._cleanup_worker()
        super().closeEvent(event)


class TrainingSignals(QObject):
    """训练进度信号定义"""
    log = Signal(str)           # 日志行
    progress = Signal(dict)     # 进度字典（epoch/loss/f1 等）
    finished = Signal(dict)     # 完成信号（最终结果）
    error = Signal(str)         # 错误信号


class TrainingWorker(QObject):
    """
    训练任务工作线程

    入参: data_root/save_dir
    方法: 调 start_training 提交任务 → 循环 get_training_progress 轮询 → 发射信号
    出参: 通过 signals 发射进度/日志/完成/错误
    """

    def __init__(self, data_root, save_dir):
        """
        初始化 Worker

        入参:
            data_root: 训练数据根目录
            save_dir: 模型保存目录
        方法: 保存目录参数，训练超参数由 start_training 的默认配置提供
        出参: 无
        """
        super().__init__()
        self.data_root = data_root
        self.save_dir = save_dir
        self.signals = TrainingSignals()

    def run(self):
        """
        工作线程主函数

        入参: 无（从 self 读参数）
        方法: start_training 提交 → get_training_progress 生成器逐次 yield → 发射信号
        出参: 无（通过 signals 通知 UI）

        做什么: 在后台线程提交训练并轮询进度，避免阻塞主 UI 线程
        为什么: QThread + moveToThread 是 Qt 标准的跨线程通信模式
        """
        try:
            # 提交训练任务
            result = start_training(
                data_root=self.data_root,
                save_dir=self.save_dir,
            )

            if result.get("status") != "pending":
                self.signals.error.emit(result.get("message", "提交训练任务失败"))
                return

            task_id = result.get("task_id")
            self.signals.log.emit(f"训练任务已提交，task_id: {task_id}")

            # 轮询进度（生成器模式，每次 yield 一个进度字典）
            for progress in get_training_progress(task_id, poll_interval=2):
                status = progress.get("status")

                # 发射进度信号（更新状态行）
                if status in ("running", "completed"):
                    self.signals.progress.emit({
                        "current_epoch": progress.get("current_epoch", 0),
                        "total_epochs": progress.get("total_epochs", 0),
                        "train_loss": progress.get("train_loss"),
                        "val_f1": progress.get("val_f1"),
                        "best_f1": progress.get("best_f1"),
                    })

                # 发射新增的日志行
                logs = progress.get("logs", [])
                # 通过比对长度只发射新增的日志（避免重复）
                if hasattr(self, "_last_log_count"):
                    new_logs = logs[self._last_log_count:]
                else:
                    new_logs = logs
                self._last_log_count = len(logs)

                for log_line in new_logs:
                    self.signals.log.emit(log_line)

                if status == "completed":
                    self.signals.finished.emit(progress)
                    return
                if status == "failed":
                    self.signals.error.emit(progress.get("error") or progress.get("message", "训练失败"))
                    return

        except Exception as e:
            self.signals.error.emit(f"工作线程异常: {str(e)}")


class TrainingModule:
    """
    训练功能模块入口（仿 BatchProcessing 的包装类）

    入参: navigation_functions
    方法: 懒加载 TrainingDialog 实例
    出参: 无（显示对话框）
    """

    def __init__(self, navigation_functions):
        """初始化模块，保存导航函数引用"""
        self.navigation_functions = navigation_functions
        self.dialog = None

    def show_training_dialog(self):
        """
        显示训练对话框

        入参: 无
        方法: 懒加载创建 TrainingDialog → exec
        出参: 无
        """
        if self.dialog is None:
            self.dialog = TrainingDialog(self.navigation_functions)
        else:
            # 复用已有对话框，刷新主题
            self.dialog.on_theme_changed()

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
