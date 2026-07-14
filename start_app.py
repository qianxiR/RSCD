"""
RSCD 遥感影像变化检测系统 - 应用入口
启动 PySide6 桌面应用，初始化共享目录
"""
import os
import sys
from pathlib import Path
import traceback

import PySide6
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal, QThread, Slot


def _ensure_project_root_in_path():
    """
    确保项目根目录在 sys.path 中
    做什么: 将项目根目录添加到 Python 搜索路径
    为什么: 启动脚本可能在任意目录执行,需要确保能找到 Frontend/Controller/Backend/utils 包
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_in_path()

from Frontend.app import RemoteSensingApp


class Worker(QThread):
    """
    后台工作线程,用于执行初始化任务而不阻塞 UI

    入参:
        task_func: 待执行的无参函数
        task_name: 任务名称(用于日志)
    出参:
        通过信号发射执行状态和结果
    """
    update_status = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, task_func, task_name):
        super().__init__()
        self.task_func = task_func
        self.task_name = task_name

    def run(self):
        """
        执行后台任务
        做什么: 在子线程中调用 task_func 并通过信号通知主线程
        为什么: 避免目录清理等 IO 操作阻塞 Qt 主线程导致界面卡顿
        """
        try:
            self.update_status.emit(f"正在执行: {self.task_name}...")
            success = self.task_func()
            self.update_status.emit(f"{self.task_name} 完成.")
            self.finished.emit(success, self.task_name)
        except Exception as e:
            error_msg = f"{self.task_name} 执行出错: {str(e)}"
            print(f"[错误] {error_msg}")
            traceback.print_exc()
            self.update_status.emit(error_msg)
            self.finished.emit(False, self.task_name)


def setup_shared_directories():
    """
    创建并清理共享数据目录（data/t1, data/t2, data/output）

    入参: 无（路径从 utils.paths 读取统一配置）
    方法: 转调 utils.paths.ensure_shared_dirs
    出参: bool，成功返回 True

    做什么: 委托给统一的共享目录初始化入口
    为什么: 消除历史路径分歧，前端原用 t1/ 后端用 data/t1，现统一为 data/ 子目录
    """
    from utils.paths import ensure_shared_dirs, T1_DIR, T2_DIR, OUTPUT_DIR

    print(f"[信息] 共享目录: {T1_DIR} | {T2_DIR} | {OUTPUT_DIR}")
    return ensure_shared_dirs(clear=True)


def run_app():
    """
    启动应用主函数

    入参: 无
    方法: 配置 Qt 插件路径 → 创建 QApplication → 初始化主窗口 → 后台清理 → 启动事件循环
    出参: 无(调用 sys.exit)

    做什么: 完成所有初始化工作后进入 Qt 事件循环
    为什么: 分离初始化与 UI 显示,后台清理不阻塞首屏渲染
    """
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), "plugins")
    os.environ["QT_PLUGIN_PATH"] = plugin_path
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugin_path, "platforms")

    app = QApplication(sys.argv)

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("创建主窗口...")
    window = RemoteSensingApp()
    home_page = window.home_page

    print("准备后台任务...")
    cleanup_worker = Worker(setup_shared_directories, "清理缓存目录")

    cleanup_worker.update_status.connect(home_page.update_loading_message)
    cleanup_worker.finished.connect(home_page.handle_task_completion)

    print("启动后台任务...")
    cleanup_worker.start()

    print("显示主窗口...")
    window.show()

    print("启动 Qt 事件循环...")
    exit_code = app.exec()
    print("Qt 事件循环结束。")

    sys.exit(exit_code)


if __name__ == "__main__":
    run_app()
