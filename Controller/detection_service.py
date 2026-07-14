"""
RSCD 控制层 - 变化检测模型服务（本地进程内调用）

取消 FastAPI HTTP 层后，由本模块在 Controller 进程内直接调用 Backend 推理逻辑。
单/批量统一为单一流程，是否保留坐标系由 Backend 处理脚本内部依据 geo 探测自动决定。

入参: run_detection(before_path, after_path, output_path, batch=False)
方法: 懒加载 Backend 处理函数 → 按批量标志分发 → 返回推理结果字典
出参: Dict[str, Any] — 包含 status, output_path, quad_view_path, vector_files 等
"""
import os

# 解决 OpenMP 冲突：允许多个 OpenMP 库共存，避免运行时崩溃
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


# 默认模型权重路径（项目根/checkpoint/best_model.pth）
default_model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoint", "best_model.pth"
)


class ChangeDetectionModel:
    """
    变化检测模型服务（本地调用）

    取消图像/影像二元划分后，仅保留单文件与批量两个入口，统一由 Backend 自动探测坐标。
    """

    def __init__(self):
        """
        初始化模型服务

        入参: 无
        方法: 设置共享数据目录路径（项目相对）
        出参: 无

        做什么: 预置目录路径供路径规范化使用
        为什么: Backend 脚本接到的 output_path 可能是文件也可能是目录，需按需规范化
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.t1_dir = os.path.join(base_dir, "data", "t1")
        self.t2_dir = os.path.join(base_dir, "data", "t2")
        self.output_dir = os.path.join(base_dir, "data", "output")

    def run_detection(self, before_path, after_path, output_path, batch=False):
        """
        执行变化检测（本地直接调用 Backend，不走 HTTP）

        入参:
            before_path: 前时相文件/目录路径
            after_path: 后时相文件/目录路径
            output_path: 输出路径（单文件为文件/目录，批量为目录）
            batch: 是否批量处理（目录对目录）
        方法: 懒加载对应 Backend 处理函数 → 构造推理 args → 调 process_and_save
        出参: Dict[str, Any]，含 status/output_path/quad_view_path/vector_files 等

        做什么: Controller→Backend 的统一本地调用入口
        为什么: 取消 FastAPI 后，所有检测请求在此直接进入 Backend 进程内推理，
                无网络/轮询开销；坐标保留与否交由 Backend 自动探测
        """
        before_path = before_path.replace('\\', '/')
        after_path = after_path.replace('\\', '/')
        output_path = output_path.replace('\\', '/')

        # 批量模式下 output_path 为目录；单文件模式下若为目录则拼文件名
        if not batch and os.path.isdir(output_path):
            before_name = os.path.splitext(os.path.basename(before_path))[0]
            output_path = os.path.join(output_path, f"{before_name}_result")

        # 确保输出目录存在
        output_dir = output_path if batch else os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                return {"status": "error", "message": f"创建输出目录失败: {str(e)}"}

        args = self._build_args(before_path, after_path, output_path)

        if batch:
            process_fn = self._load_batch_processor()
        else:
            process_fn = self._load_single_processor()

        return process_fn(args)

    def _build_args(self, before_path, after_path, output_path):
        """
        构造推理参数 Namespace

        入参: 三路径
        方法: 委托 Controller.default_args.build_default_args
        出参: argparse.Namespace
        """
        from Controller.default_args import build_default_args
        return build_default_args(before_path, after_path, output_path, default_model_path)

    def _load_single_processor(self):
        """
        懒加载单影像处理函数

        入参: 无
        方法: import Backend.processing.single_image.process_and_save
        出参: 可调用对象 process_and_save

        做什么: 推迟 Backend 导入到首次检测时，加快应用启动
        为什么: Backend 依赖 torch/GDAL，启动时即导入会显著拖慢首屏渲染
        """
        from Backend.processing.single_image import process_and_save
        return process_and_save

    def _load_batch_processor(self):
        """
        懒加载批量处理函数

        入参: 无
        方法: import Backend.processing.batch_image.process_and_save
        出参: 可调用对象 process_and_save
        """
        from Backend.processing.batch_image import process_and_save
        return process_and_save


# 单例对象，供 local_service / task_manager 导入使用
detection_model = ChangeDetectionModel()
