"""
RSCD Frontend 视图共享层

集中存放 batch_dialog / raster_batch 共用的线程池、跨线程日志、QSS、工具函数。
消除两个批量对话框（各 1700+ 行）中的重复代码。

入参: 见各子模块
方法: 提供线程池单例、Qt 跨线程日志 Mixin、主题 QSS 生成器、网格解析工具
出参: 函数 / Mixin 类 / 常量
"""
from .thread_pool import (
    CPU_COUNT,
    DEFAULT_THREAD_POOL_SIZE,
    get_thread_pool,
    cleanup_thread_pool,
)
from .qt_logging import ThreadSafeLogMixin
from .styles import tab_widget_qss, line_edit_qss, dialog_base_qss
from .utils import parse_grid_size

__all__ = [
    "CPU_COUNT",
    "DEFAULT_THREAD_POOL_SIZE",
    "get_thread_pool",
    "cleanup_thread_pool",
    "ThreadSafeLogMixin",
    "tab_widget_qss",
    "line_edit_qss",
    "dialog_base_qss",
    "parse_grid_size",
]
