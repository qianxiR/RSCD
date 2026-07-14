#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 项目路径管理模块 - 用于统一管理项目中的所有路径

import os
import sys
from pathlib import Path

def find_project_root(path=None):
    """
    从给定路径向上查找项目根目录
    通过查找.project_root标记文件来确定项目根目录

    Args:
        path: 起始查找路径，默认为当前文件所在目录

    Returns:
        Path: 项目根目录路径

    Raises:
        FileNotFoundError: 当无法找到.project_root标记文件时
    """
    if path is None:
        path = Path(os.path.dirname(os.path.abspath(__file__)))
    else:
        path = Path(path)

    if (path / ".project_root").exists():
        return path

    parent = path.parent
    if parent == path:
        raise FileNotFoundError("无法找到项目根目录，请确保.project_root文件存在")

    return find_project_root(parent)

# 项目根目录
PROJECT_ROOT = find_project_root()

# 重要目录路径 - 适配新目录结构
SCRIPTS_APP_DIR = PROJECT_ROOT / "Backend" / "processing"
API_DIR = PROJECT_ROOT / "Controller"
MODEL_DIR = PROJECT_ROOT / "Backend" / "network"
DATA_DIR = PROJECT_ROOT / "Backend" / "data"

# 模型路径 - 相对于项目根目录
# 推理默认用 best_model.pth（裸 state_dict，F1 最优权重）
# checkpoint.pth.tar 保留作为带 optimizer 状态的可续训权重备份
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoint" / "best_model.pth"

# 临时目录
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# 共享数据目录 - 前后端交换影像/结果的唯一约定路径
# 统一从 PROJECT_ROOT/data 派生，避免多处独立定义导致不一致
SHARED_DATA_DIR = PROJECT_ROOT / "data"
T1_DIR = SHARED_DATA_DIR / "t1"          # 前时相影像暂存目录
T2_DIR = SHARED_DATA_DIR / "t2"          # 后时相影像暂存目录
OUTPUT_DIR = SHARED_DATA_DIR / "output"  # 检测结果输出目录


def ensure_shared_dirs(clear: bool = True) -> bool:
    """
    创建并按需清空共享数据目录（t1/t2/output）

    入参:
        clear: 是否清空目录内已有文件，默认 True（前端每次启动前重置暂存区）
    方法: 遍历三个目录 → exist_ok 创建 → 可选地删除内部文件/子目录
    出参: bool，全部成功返回 True，任一异常返回 False

    做什么: 集中维护共享目录的初始化逻辑，替代各模块自行 makedirs 的副作用
    为什么: 消除 start_app.py / run_server.py / api_client.py 三处独立路径定义
    """
    import shutil

    target_dirs = [T1_DIR, T2_DIR, OUTPUT_DIR]
    try:
        for dir_path in target_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            if not clear:
                continue
            for item in dir_path.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except OSError as exc:
                    print(f"[警告] 清理 {item} 时出错: {exc}")
        return True
    except OSError as exc:
        print(f"[错误] 初始化共享目录时出错: {exc}")
        return False

def add_project_paths_to_sys_path():
    """将项目相关路径添加到sys.path"""
    for p in [PROJECT_ROOT, SCRIPTS_APP_DIR, PROJECT_ROOT.parent]:
        sys.path.insert(0, str(p))

    print(f"已添加项目路径到sys.path:\n{PROJECT_ROOT}\n{SCRIPTS_APP_DIR}\n{PROJECT_ROOT.parent}")

def setup_module_paths():
    """设置模块路径并返回一个上下文对象，包含所有重要路径

    Returns:
        dict: 包含所有重要路径的字典
    """
    add_project_paths_to_sys_path()

    # 将 Frontend/, Controller/, Backend/, utils/ 都加到 sys.path
    for subdir in ["Frontend", "Controller", "Backend", "utils"]:
        subdir_path = PROJECT_ROOT / subdir
        if subdir_path.exists():
            sys.path.insert(0, str(subdir_path))

    return {
        "project_root": PROJECT_ROOT,
        "scripts_app_dir": SCRIPTS_APP_DIR,
        "api_dir": API_DIR,
        "model_dir": MODEL_DIR,
        "data_dir": DATA_DIR,
        "default_model_path": DEFAULT_MODEL_PATH,
        "temp_dir": TEMP_DIR
    }

# 如果直接运行此模块，打印所有路径信息
if __name__ == "__main__":
    paths = setup_module_paths()

    print("项目路径信息:")
    for name, path in paths.items():
        print(f"{name}: {path}")
        print(f"路径存在: {Path(path).exists()}")

    print("\n系统路径:")
    for i, path in enumerate(sys.path[:5]):
        print(f"{i}: {path}")
