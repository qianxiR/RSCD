import os
import cv2
import logging
import threading
import time

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QFileDialog

from Frontend.widgets import ZoomableLabel


# 线程间通信信号桥：子线程发结果，主线程显示
class ThreadCommunicator(QObject):
    """用于线程间通信的对象"""
    display_result_signal = Signal(object)


thread_communicator = ThreadCommunicator()

# 本地检测服务（取代原 api_client.detect_changes）
from Controller.local_service import submit_detection, get_task_status


class ExecuteChangeDetectionTask:
    """变化检测任务（本地进程内调用，无 HTTP）"""

    def __init__(self, navigation_functions):
        """
        初始化变化检测任务

        入参:
            navigation_functions: 导航功能类实例
        方法: 保存引用 → 连接结果显示信号
        出参: 无
        """
        self.navigation_functions = navigation_functions

        self.result_image_path = None
        self.result_image = None
        self.task_directory = None

        # 使用 Qt.QueuedConnection 确保信号在主线程中处理
        thread_communicator.display_result_signal.connect(self._display_result, Qt.QueuedConnection)

    def log_message(self, message, level_or_show_in_ui=True):
        """
        记录消息到日志

        入参:
            message: 消息内容
            level_or_show_in_ui: 日志级别字符串("START"/"COMPLETE"/"ERROR")或是否显示在UI
        出参: 无
        """
        show_in_ui = True
        level = None

        if isinstance(level_or_show_in_ui, str):
            level = level_or_show_in_ui
        else:
            show_in_ui = level_or_show_in_ui
            level = "ERROR" if ("错误" in message or "失败" in message) else "COMPLETE"

        if show_in_ui and hasattr(self.navigation_functions, 'log_message'):
            self.navigation_functions.log_message(message, level)

    def on_begin_clicked(self):
        """
        开始按钮点击处理

        入参: 无
        方法: 校验前后时相已导入 → 清除旧结果 → 选择输出目录 → 起子线程执行检测
        出参: 无
        """
        if not getattr(self.navigation_functions, 'file_path', None):
            self.log_message("错误: 未导入前时相影像", True)
            return
        if not getattr(self.navigation_functions, 'file_path_after', None):
            self.log_message("错误: 未导入后时相影像", True)
            return

        self._clear_previous_result()

        default_dir = os.path.dirname(self.navigation_functions.file_path_after)
        output_folder = QFileDialog.getExistingDirectory(None, "选择保存文件夹", default_dir)
        if not output_folder:
            return

        before_path = os.path.abspath(self.navigation_functions.file_path).replace("\\", "/")
        after_path = os.path.abspath(self.navigation_functions.file_path_after).replace("\\", "/")
        output_path = os.path.abspath(output_folder).replace("\\", "/")
        os.makedirs(output_path, exist_ok=True)

        self.log_message("开始执行变化检测，请稍后...", "START")

        detection_thread = threading.Thread(
            target=self._execute_detection,
            args=(before_path, after_path, output_path)
        )
        detection_thread.daemon = True
        detection_thread.start()

    def _execute_detection(self, before_path, after_path, output_path):
        """
        在子线程中提交检测任务并轮询结果

        入参: 三路径
        方法: submit_detection → 循环 get_task_status 至终态 → 读取 display_image_path 显示
        出参: 无（通过信号桥回主线程显示）

        做什么: 本地提交 + 进程内轮询，取代原 HTTP 轮询
        为什么: 本地轮询间隔可大幅缩短（0.5s），无网络开销
        """
        try:
            before_path = os.path.abspath(before_path).replace("\\", "/")
            after_path = os.path.abspath(after_path).replace("\\", "/")
            output_path = os.path.abspath(output_path).replace("\\", "/")
            os.makedirs(output_path, exist_ok=True)

            task_id = submit_detection(before_path, after_path, output_path, batch=False)
            self.log_message(f"已创建检测任务 {task_id}，正在执行...", "INFO")

            # 进程内轮询至终态
            task_result = self._poll_until_done(task_id)
            final_status = task_result.get("status")
            display_image_path = task_result.get("display_image_path")

            if final_status != "completed":
                self.log_message(
                    f"检测任务未完成: 状态={final_status}, 消息={task_result.get('message')}",
                    "ERROR"
                )
                thread_communicator.display_result_signal.emit(None)
                return

            if not display_image_path or not os.path.exists(display_image_path):
                self.log_message(
                    f"错误: 未找到结果图像。路径: '{display_image_path}'", "ERROR"
                )
                thread_communicator.display_result_signal.emit(None)
                return

            self.log_message(f"读取结果图像: {display_image_path}", "INFO")

            # 优先用 GDAL 读取（GeoTIFF 掩膜），失败回退 OpenCV
            result_img = self._read_result_image(display_image_path)
            if result_img is None:
                self.log_message(f"错误: 无法读取结果图像 '{display_image_path}'", "ERROR")
                thread_communicator.display_result_signal.emit(None)
                return

            thread_communicator.display_result_signal.emit(result_img)
            self.log_message("变化检测完成！", "COMPLETE")

        except Exception as e:
            logging.error(f"_execute_detection 出错: {e}", exc_info=True)
            self.log_message(f"变化检测过程中出错: {str(e)}", "ERROR")
            thread_communicator.display_result_signal.emit(None)

    def _poll_until_done(self, task_id, poll_interval=0.5, max_wait=3600):
        """
        轮询任务直至终态

        入参: task_id/poll_interval/max_wait
        方法: 循环 get_task_status → 命中 completed/failed/error 返回
        出参: 终态任务状态字典
        """
        start = time.time()
        while time.time() - start < max_wait:
            result = get_task_status(task_id)
            if result["status"] in ("completed", "failed", "error"):
                return result
            time.sleep(poll_interval)
        return {"status": "error", "message": "轮询超时"}

    def _read_result_image(self, image_path):
        """
        读取结果掩膜图像（兼容 PNG 与 GeoTIFF）

        入参: image_path
        方法: .tif/.tiff 优先 GDAL 单波段读取 → 否则 cv2.imread
        出参: numpy 数组或 None
        """
        lower = image_path.lower()
        if lower.endswith(('.tif', '.tiff')):
            try:
                from osgeo import gdal
                dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
                if dataset is not None:
                    arr = dataset.ReadAsArray()
                    dataset = None
                    if arr is not None and arr.ndim == 3:
                        arr = arr[0]
                    return arr
            except Exception:
                pass
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def _display_result(self, result_img):
        """
        显示检测结果（主线程槽）

        入参: result_img — 检测结果 NumPy 数组（可为 None）
        出参: bool 是否成功显示
        """
        try:
            if result_img is None:
                self.log_message("未生成结果图像，无法显示", True)
                return False

            self.result_image = result_img

            if hasattr(self.navigation_functions, 'set_result_image'):
                return self.navigation_functions.set_result_image(result_img, 'memory_image')

            if not hasattr(self.navigation_functions, 'label_result'):
                self.log_message("NavigationFunctions缺少label_result属性", True)
                return False

            label_result = self.navigation_functions.label_result
            height, width = result_img.shape[:2]
            bytes_per_line = 3 * width

            # 统一转 RGB 三通道以兼容 Format_RGB888
            if result_img.ndim == 2:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
            else:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            from PySide6.QtGui import QImage, QPixmap
            q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            if pixmap.isNull():
                self.log_message("无法创建结果图像的Pixmap", True)
                return False

            if hasattr(self.navigation_functions, 'result_image_path'):
                self.navigation_functions.result_image_path = 'memory_image'
            if hasattr(self.navigation_functions, 'result_image'):
                self.navigation_functions.result_image = result_img

            if hasattr(label_result, 'set_pixmap'):
                label_result.set_pixmap(pixmap)
            elif hasattr(label_result, 'setPixmap'):
                label_result.setPixmap(pixmap)
                label_result.setScaledContents(True)

            return True

        except Exception as e:
            self.log_message(f"显示结果图像时出错: {str(e)}", True)
            return False

    def _clear_previous_result(self):
        """
        清除之前的结果图像

        入参: 无
        方法: 置空结果引用 → 清空结果标签 → 处理待机事件刷新 UI
        出参: 无
        """
        try:
            if hasattr(self, 'result_image'):
                self.result_image = None
            if hasattr(self, 'result_image_path'):
                self.result_image_path = None

            if hasattr(self.navigation_functions, 'result_image'):
                self.navigation_functions.result_image = None
            if hasattr(self.navigation_functions, 'result_image_path'):
                self.navigation_functions.result_image_path = None

            if hasattr(self.navigation_functions, 'label_result'):
                label = self.navigation_functions.label_result
                if hasattr(label, 'clear'):
                    label.clear()
                    if hasattr(label, 'setText'):
                        label.setText("正在处理...")

            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

            self.log_message("已清除之前的结果图像", "INFO")

        except Exception as e:
            self.log_message(f"清除之前结果图像时出错: {str(e)}", "WARNING")
