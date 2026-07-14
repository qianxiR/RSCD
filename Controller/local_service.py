"""
RSCD 控制层 - 本地检测服务（取代 api_client 的前端调用入口）

取消 FastAPI HTTP 后，前端通过本模块在进程内提交检测任务：
文件复制到共享目录 → 提交线程池执行 → 输出复制回用户目录并统一重命名。
保留 session_id 隔离与 display_image_path 返回，前端视图改动最小。

入参: submit_detection(before_path, after_path, output_path, batch=False)
方法: 复制输入到 data/t1,t2 → 创建 task → 线程池跑 run_detection_task → 复制重命名结果
出参: Dict[str, Any] — status/message/task_id/session_id/output_path/display_image_path
"""
import os
import uuid
import time
import shutil
import logging
import threading
from typing import Dict, Any, Optional, List

from utils.paths import T1_DIR, T2_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 复用 task_manager 的任务字典与执行函数
from Controller.task_manager import detection_tasks, run_detection_task
from Controller.detection_service import detection_model


# 后台线程池：单例，检测任务在此线程内同步执行，避免阻塞 Qt 主线程
# 做什么: 提供 submit 用的后台线程，前端 fire-and-forget 后轮询 get_task_status
# 为什么: Qt 主线程不能跑 torch 推理（会卡死 UI），用线程而非进程（本地调用无需跨进程序列化）
_executor_lock = threading.Lock()
_executor = None
# 训练独立线程池：训练是长任务，不能和检测共用单线程 executor（否则会阻塞检测）
_training_executor_lock = threading.Lock()
_training_executor = None


def _get_executor():
    """
    获取后台单线程执行器（懒加载，检测专用）

    入参: 无
    方法: 首次调用创建 ThreadPoolExecutor(max_workers=1)
    出参: ThreadPoolExecutor 实例

    做什么: 保证检测任务串行执行（GPU 推理不支持并发，且避免显存争用）
    为什么: 多任务并发会同时加载多份模型副本撑爆显存，串行更稳
    """
    global _executor
    with _executor_lock:
        if _executor is None:
            from concurrent.futures import ThreadPoolExecutor
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rscd-detect")
    return _executor


def _get_training_executor():
    """
    获取训练专用单线程执行器（懒加载）

    入参: 无
    出参: ThreadPoolExecutor(max_workers=1)

    做什么: 训练任务独立调度
    为什么: training_task_manager 内部已用 _training_lock 保证训练单任务运行，
            此处再独立 executor 是为了不与检测串行排队（训练数小时，会阻塞检测）
    """
    global _training_executor
    with _training_executor_lock:
        if _training_executor is None:
            from concurrent.futures import ThreadPoolExecutor
            _training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rscd-train")
    return _training_executor


def _generate_session_id() -> str:
    """
    生成唯一会话 ID

    入参: 无
    出参: 'YYYYMMDDHHMMSS_xxxxxxxx' 格式字符串
    """
    return f"{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _copy_to_shared(source_path: str, target_dir: str, session_id: str) -> str:
    """
    将文件复制到共享数据目录

    入参: source_path/target_dir/session_id
    方法: 加 session_id 前缀避免多任务冲突 → shutil.copy2
    出参: 复制后的目标文件路径

    做什么: 把用户选择的文件落到 Backend 读取的统一目录
    为什么: Backend 路径规范化基于 data/t1,data/t2，复制后路径稳定
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"源文件未找到: {source_path}")

    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.basename(source_path)
    target_path = os.path.join(target_dir, f"{session_id}_{filename}")

    shutil.copy2(source_path, target_path)
    return target_path


def _copy_directory_to_shared(source_dir: str, target_dir: str, session_id: str) -> str:
    """
    将目录复制到共享数据目录

    入参: source_dir/target_dir/session_id
    方法: 加 session_id 前缀避免多任务冲突 → shutil.copytree
    出参: 复制后的目标目录路径
    """
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"源路径不是目录: {source_dir}")

    dir_name = os.path.basename(source_dir.rstrip('/\\')) or "input"
    target_path = os.path.join(target_dir, f"{session_id}_{dir_name}")

    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.copytree(source_dir, target_path)
    return target_path


def _prepare_output_directory(user_output_path: str, session_id: str) -> str:
    """
    在共享输出目录创建会话专属子目录

    入参: user_output_path/session_id
    出参: 共享输出子目录路径
    """
    dir_name = os.path.basename(user_output_path.rstrip('/\\')) or f"output_{session_id}"
    target_path = os.path.join(str(OUTPUT_DIR), f"{session_id}_{dir_name}")
    os.makedirs(target_path, exist_ok=True)
    return target_path


def submit_detection(before_path: str, after_path: str, output_path: str,
                     batch: bool = False) -> str:
    """
    提交检测任务（fire-and-forget，返回 task_id 供轮询）

    入参:
        before_path: 用户选择的文件/目录
        after_path: 用户选择的文件/目录
        output_path: 用户选择的输出目录
        batch: 是否批量
    方法: 校验输入 → 复制到共享目录 → 创建 task → 提交线程池
    出参: task_id 字符串

    做什么: 前端检测入口，提交后立即返回，结果通过 get_task_status 查询
    为什么: 与原 api_client.detect_changes 的轮询模型对齐，前端改动最小
    """
    session_id = _generate_session_id()
    task_id = f"task_{session_id}"

    # 校验并复制输入
    if batch:
        if not os.path.isdir(before_path) or not os.path.isdir(after_path):
            raise ValueError("批量模式要求前后时相均为目录")
        shared_before = _copy_directory_to_shared(before_path, str(T1_DIR), session_id)
        shared_after = _copy_directory_to_shared(after_path, str(T2_DIR), session_id)
    else:
        if not os.path.isfile(before_path) or not os.path.isfile(after_path):
            raise ValueError("单文件模式要求前后时相均为文件")
        shared_before = _copy_to_shared(before_path, str(T1_DIR), session_id)
        shared_after = _copy_to_shared(after_path, str(T2_DIR), session_id)

    shared_output = _prepare_output_directory(output_path, session_id)

    detection_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "message": "变化检测任务已创建，等待执行",
        "before_path": shared_before,
        "after_path": shared_after,
        "output_path": shared_output,
        "session_id": session_id,
        "user_output_path": output_path,
        "batch": batch,
        "start_time": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "end_time": None,
        "result": None,
    }

    _get_executor().submit(
        run_detection_task,
        task_id, shared_before, shared_after, shared_output, batch, detection_model
    )

    return task_id


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    查询任务状态（前端轮询用）

    入参: task_id
    方法: 从 detection_tasks 读取；若终态则触发结果复制重命名并填充 display_image_path
    出参: 任务状态字典；终态任务额外含 session_id/output_path/display_image_path

    做什么: 前端轮询入口，终态时完成共享目录→用户目录的复制与重命名
    为什么: 复制重命名只在终态首次查询时做一次，用 _finalized 标记幂等
    """
    task = detection_tasks.get(task_id)
    if task is None:
        return {"status": "error", "message": f"任务不存在: {task_id}"}

    # 非终态直接返回当前状态
    if task["status"] not in ("completed", "failed"):
        return {
            "status": task["status"],
            "message": task["message"],
            "task_id": task_id,
        }

    # 终态且未做结果复制 → 复制并重命名（幂等）
    if not task.get("_finalized"):
        _finalize_task_output(task)
        task["_finalized"] = True

    return {
        "status": task["status"],
        "message": task["message"],
        "task_id": task_id,
        "session_id": task.get("session_id"),
        "output_path": task.get("user_output_path"),
        "display_image_path": task.get("display_image_path"),
        "vector_files": task.get("display_vector_files", []),
    }


def _finalize_task_output(task: Dict[str, Any]):
    """
    将共享目录结果复制回用户目录并统一重命名

    入参: task — 终态任务字典
    方法: copytree 共享输出→用户目录 → 按 session_id 找掩码/四联图重命名为统一名 →
          清理旧文件 → 回填 display_image_path/display_vector_files

    做什么: 终态后处理，使前端能在用户输出目录按固定命名找到显示文件
    为什么: 单文件与批量、坐标与无坐标产物命名不同，统一重命名便于前端展示
    """
    if task["status"] == "failed":
        return

    shared_output = task["output_path"]
    user_output = task["user_output_path"]
    session_id = task["session_id"]

    if not shared_output or not os.path.isdir(shared_output):
        task["display_image_path"] = None
        return

    try:
        os.makedirs(user_output, exist_ok=True)
        shutil.copytree(shared_output, user_output, dirs_exist_ok=True)
    except Exception as e:
        logging.error(f"[{session_id}] 复制结果到用户目录失败: {e}")
        task["display_image_path"] = None
        return

    # 找到当前 session 的掩码文件作为显示主图（按 session_id 前缀过滤）
    display_image_path = _find_display_image(user_output, session_id)
    task["display_image_path"] = display_image_path

    # 收集用户目录下的矢量文件供前端展示
    vector_files = _collect_vector_files(user_output)
    task["display_vector_files"] = vector_files

    # 清理旧任务的掩码文件（非当前 session）
    _cleanup_old_masks(user_output, session_id)


def _find_display_image(user_output: str, session_id: str) -> Optional[str]:
    """
    在用户目录查找当前任务的掩码主图（优先 tif，其次 png）

    入参: user_output/session_id
    方法: 遍历目录 → 过滤含 session_id 的 *_mask.tif/*_result.png → 返回首个匹配
    出参: 文件路径或 None
    """
    if not os.path.isdir(user_output):
        return None

    candidates = []
    for item in os.listdir(user_output):
        if session_id not in item or not os.path.isfile(os.path.join(user_output, item)):
            continue
        lower = item.lower()
        if lower.endswith("_mask.tif") or lower.endswith("_result.png"):
            candidates.append(os.path.join(user_output, item))

    return candidates[0] if candidates else None


def _collect_vector_files(user_output: str) -> List[str]:
    """
    收集用户目录（含 vectors/ 子目录）下的矢量文件

    入参: user_output
    出参: 矢量文件路径列表（含 shp 辅助文件）
    """
    if not os.path.isdir(user_output):
        return []

    vector_files = []
    vectors_dir = os.path.join(user_output, "vectors")
    search_dirs = [vectors_dir, user_output] if os.path.isdir(vectors_dir) else [user_output]

    for search_dir in search_dirs:
        for item in os.listdir(search_dir):
            lower = item.lower()
            full_path = os.path.join(search_dir, item)
            if not os.path.isfile(full_path):
                continue
            if lower.endswith((".shp", ".geojson")):
                vector_files.append(full_path)
                if lower.endswith(".shp"):
                    base = os.path.splitext(item)[0]
                    for ext in (".dbf", ".shx", ".prj", ".cpg", ".qpj"):
                        aux = os.path.join(search_dir, base + ext)
                        if os.path.exists(aux):
                            vector_files.append(aux)
    return vector_files


def _cleanup_old_masks(user_output: str, current_session_id: str):
    """
    清理旧任务的掩码文件，仅保留当前 session

    入参: user_output/current_session_id
    方法: glob *_mask.* 与 *_result.* → 非当前 session 的删除
    """
    if not os.path.isdir(user_output):
        return

    import glob as glob_module
    for pattern in ("*_mask.*", "*_result.*"):
        for file_path in glob_module.glob(os.path.join(user_output, pattern)):
            filename = os.path.basename(file_path)
            if current_session_id not in filename:
                try:
                    os.remove(file_path)
                except Exception:
                    pass


def check_connection() -> bool:
    """
    检测服务可用性（本地化后恒为 True）

    入参: 无
    出参: True（保留函数签名以兼容前端既有调用）

    做什么: 取代原 HTTP /health 探活
    为什么: 取消 FastAPI 后不再有远程服务，检测能力随应用启动即就绪
    """
    return True


# ==================== 训练任务（封装 training_task_manager） ====================


def start_training(data_root: str, save_dir: str, epochs: int = 10,
                   batch_size: int = 4, lr: float = 1e-4,
                   pretrained: Optional[str] = None) -> Dict[str, Any]:
    """
    提交模型训练任务（取代 api_client.start_training 的 HTTP 提交）

    入参:
        data_root: 训练数据根目录（含 train/val 子目录）
        save_dir: 模型保存目录
        epochs/batch_size/lr/pretrained: 训练超参
    方法: 校验数据目录 → 写入 training_tasks → 提交线程池 → 返回 pending + task_id
    出参: Dict，含 task_id / status=pending / message

    做什么: 前端训练入口，本地直接调度 Backend.training
    为什么: 取消 FastAPI 后训练与检测一样在进程内执行，复用 training_task_manager 的状态机
    """
    from Controller.training_task_manager import training_tasks, run_training_task

    if not os.path.isdir(data_root):
        return {"status": "error", "message": f"训练数据目录不存在: {data_root}"}
    if not os.path.isdir(os.path.join(data_root, "train")):
        return {"status": "error", "message": f"训练数据缺少 train 子目录: {data_root}/train"}
    if not os.path.isdir(os.path.join(data_root, "val")):
        return {"status": "error", "message": f"训练数据缺少 val 子目录: {data_root}/val"}

    os.makedirs(save_dir, exist_ok=True)

    task_id = f"train_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    training_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "current_epoch": 0,
        "total_epochs": epochs,
        "train_loss": None,
        "val_f1": None,
        "val_iou": None,
        "best_f1": None,
        "logs": [],
        "save_dir": save_dir,
        "message": "训练任务已创建，等待执行",
        "start_time": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "end_time": None,
        "error": None,
    }

    _get_training_executor().submit(
        run_training_task,
        task_id, data_root, save_dir, epochs, batch_size, lr, pretrained,
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": f"训练任务已创建: {epochs} epochs, batch_size={batch_size}, lr={lr}",
    }


def get_training_progress(task_id: str, poll_interval: float = 2.0,
                          max_wait_time: float = 7200):
    """
    生成器：持续轮询训练进度直到任务结束

    入参: task_id/poll_interval/max_wait_time
    方法: 循环读取 training_tasks 字典 → 仅在日志增量或状态变化时 yield → 终态停止
    出参: generator，每次 yield 当前进度 dict（含 status/current_epoch/logs 等）

    做什么: 取代 api_client.get_training_progress 的 HTTP 轮询，改读本地字典
    为什么: 进程内轮询无需 HTTP，但保留生成器接口与增量 yield 语义，前端零改动
    """
    from Controller.training_task_manager import training_tasks

    start_wait = time.time()
    last_log_count = 0

    while time.time() - start_wait < max_wait_time:
        progress = training_tasks.get(task_id)
        if progress is None:
            yield {"status": "error", "message": f"训练任务不存在: {task_id}"}
            return

        status = progress.get("status")
        current_logs = progress.get("logs", [])

        if len(current_logs) > last_log_count or status in ("completed", "failed"):
            last_log_count = len(current_logs)
            yield progress

        if status in ("completed", "failed"):
            return

        time.sleep(poll_interval)

    yield {"status": "timeout", "message": f"训练任务超时（{max_wait_time}s）"}


def wait_for_completion(task_id: str, poll_interval: float = 0.5,
                        max_wait_time: float = 3600) -> Dict[str, Any]:
    """
    阻塞等待任务完成（同步便捷接口）

    入参: task_id/poll_interval/max_wait_time
    方法: 循环 get_task_status 直至终态或超时
    出参: 终态任务状态字典

    做什么: 提供给需要同步语义的调用方（如批处理 worker）
    为什么: 进程内调用轮询间隔可大幅缩短（原 HTTP 轮询 15s，本地 0.5s）
    """
    start = time.time()
    while time.time() - start < max_wait_time:
        result = get_task_status(task_id)
        if result["status"] in ("completed", "failed", "error"):
            return result
        time.sleep(poll_interval)

    return {"status": "error", "message": f"等待任务 {task_id} 超时"}
