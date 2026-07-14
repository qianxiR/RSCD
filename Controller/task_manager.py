"""
RSCD 控制层 - 任务管理器（本地同步执行）

取消 FastAPI 后，任务在提交线程内同步执行；状态字典保留供前端轮询。
状态机：pending → running → completed/failed。

入参: run_detection_task(task_id, before_path, after_path, output_path, batch)
方法: 文件存在性检查 → 置 running → 调 detection_model.run_detection → 置 completed/failed
出参: 无（通过 detection_tasks 字典维护状态）
"""
import os
import logging
from typing import Dict, Any
from datetime import datetime

# 全局任务存储字典，key 为 task_id，value 为任务状态详情
detection_tasks: Dict[str, Dict[str, Any]] = {}


def run_detection_task(task_id, before_path, after_path, output_path, batch=False, detection_model=None):
    """
    同步执行变化检测任务

    入参:
        task_id: 任务唯一标识，用于在 detection_tasks 中索引
        before_path: 前时相文件/目录路径
        after_path: 后时相文件/目录路径
        output_path: 输出路径
        batch: 是否批量处理
        detection_model: ChangeDetectionModel 实例
    方法: 校验输入存在 → 置 running → 调用模型推理 → 置 completed/failed，收集产物路径

    做什么: 在提交线程内同步执行检测并更新任务状态
    为什么: 取消异步 background_tasks 后，由 local_service 的线程池调用本函数；
            同步执行使调用方可在结果字典中直接读到终态，无需轮询
    """
    if task_id not in detection_tasks:
        return

    task = detection_tasks[task_id]

    # 1. 校验输入存在性（批量校验目录，单文件校验文件）
    before_exists = os.path.exists(before_path)
    after_exists = os.path.exists(after_path)
    logging.info(
        f"任务 {task_id} 输入检查: Before='{before_path}' (存在:{before_exists}), "
        f"After='{after_path}' (存在:{after_exists})"
    )

    if not before_exists or not after_exists:
        error_msg = (
            f"任务 {task_id} 失败：输入未找到。"
            f"Before ('{before_path}'): {before_exists}, "
            f"After ('{after_path}'): {after_exists}"
        )
        logging.error(error_msg)
        task["status"] = "failed"
        task["message"] = error_msg
        task["end_time"] = datetime.now().isoformat()
        return

    # 2. 置 running
    task["status"] = "running"
    task["message"] = "输入已确认，正在执行变化检测..."

    model_result_data = {}

    # 3. 调用模型推理
    try:
        model_result_data = detection_model.run_detection(
            before_path=before_path,
            after_path=after_path,
            output_path=output_path,
            batch=batch,
        )

        # Backend 单影像返回 success；批量返回 success/failed
        backend_status = model_result_data.get("status")
        if backend_status not in ("success", "completed"):
            error_msg = model_result_data.get('message', f"模型处理失败")
            raise Exception(error_msg)

        # 收集所有产物路径（主输出、四联图、矢量及 shp 辅助文件）
        all_output_files = []
        for key in ('output_path', 'quad_view_path', 'result_dir', 'mask_dir'):
            path = model_result_data.get(key)
            if path:
                all_output_files.append(path)
        if model_result_data.get('vector_files'):
            all_output_files.extend(model_result_data['vector_files'])
            for vector_file in model_result_data['vector_files']:
                if vector_file.endswith('.shp'):
                    base_dir = os.path.dirname(vector_file)
                    base_name = os.path.splitext(os.path.basename(vector_file))[0]
                    for ext in ['.dbf', '.shx', '.prj', '.cpg', '.qpj']:
                        aux_file = os.path.join(base_dir, f"{base_name}{ext}")
                        if os.path.exists(aux_file):
                            all_output_files.append(aux_file)

        task["status"] = "completed"
        task["message"] = model_result_data.get("message", "变化检测任务完成")
        task["result"] = {
            **model_result_data,
            "output_files": all_output_files
        }
        task["output_path"] = output_path

    except Exception as e:
        error_msg = f"任务失败: {str(e)}"
        if task_id in detection_tasks:
            task["status"] = "failed"
            task["message"] = error_msg
            if "error" not in model_result_data:
                model_result_data["error"] = str(e)
            task["result"] = model_result_data
    finally:
        if task_id in detection_tasks and task["status"] not in ("pending", "running"):
            task["end_time"] = datetime.now().isoformat()
