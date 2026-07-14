"""
影像导入模块
提供前时相和后时相影像导入功能（统一支持普通图像与栅格影像）
"""
import os
from PySide6.QtWidgets import QFileDialog


class ImportBeforeImage:
    def __init__(self, navigation_functions):
        """
        初始化导入前时相影像模块

        Args:
            navigation_functions: NavigationFunctions实例，用于日志记录和图像显示
        """
        self.navigation_functions = navigation_functions

    def on_import_clicked(self):
        """导入前时相影像"""

        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            None,
            "选择前时相影像文件",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff);;所有文件 (*)",
            options=options
        )

        if file_path:
            self.navigation_functions.log_message(f"已选择前时相影像: {file_path}", "INFO")

            # 直接使用用户选择的路径
            self.navigation_functions.file_path = file_path

            # 更新图像显示
            self.navigation_functions.update_image_display()
            self.navigation_functions.log_message("前时相影像导入完成", "COMPLETE")

            return self.navigation_functions.file_path
        else:
            self.navigation_functions.log_message("未选择文件", "INFO")

        return None

    def save_image_to_dir(self, source_path, prefix=""):
        """
        保存图像到数据目录

        Args:
            source_path: 源文件路径
            prefix: 文件名前缀

        Returns:
            str: 保存后的文件路径，如果保存失败则返回None
        """
        try:
            # 获取文件名和扩展名
            file_name = os.path.basename(source_path)

            # 添加前缀（如果有）
            if prefix:
                file_name = f"{prefix}_{file_name}"

            # 构建目标路径
            # 使用源文件所在目录作为数据目录
            data_dir = os.path.dirname(source_path)
            target_path = os.path.join(data_dir, file_name)

            # 如果源路径和目标路径相同，则不需要复制
            if os.path.normpath(source_path) == os.path.normpath(target_path):
                return source_path

            # 复制文件
            import shutil
            shutil.copy2(source_path, target_path)

            return target_path

        except Exception as e:
            self.navigation_functions.log_message(f"保存图像失败: {str(e)}", "ERROR")
            return None


class ImportAfterImage:
    def __init__(self, navigation_functions):
        """
        初始化导入后时相影像模块

        Args:
            navigation_functions: NavigationFunctions实例，用于日志记录和图像显示
        """
        self.navigation_functions = navigation_functions

    def import_after_image(self):
        """导入后时相影像"""
        # 保存原始log_message以便稍后恢复
        original_log_message = self.navigation_functions.log_message

        # 覆盖log_message方法以防止START级别的日志
        def filtered_log_message(message, level="INFO"):
            # 跳过"开始导入后时相影像"消息
            if level == "START" and "导入后时相影像" in message:
                return
            # 其他消息正常记录
            original_log_message(message, level)

        # 替换为过滤版本的log_message
        self.navigation_functions.log_message = filtered_log_message

        try:
            options = QFileDialog.Options()
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                None,
                "选择后时相影像文件",
                "",
                "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff);;所有文件 (*)",
                options=options
            )

            if file_path:
                # 直接使用用户选择的路径
                self.navigation_functions.file_path_after = file_path

                # 更新图像显示
                self.navigation_functions.update_image_display(is_before=False)
                self.navigation_functions.log_message("后时相影像导入完成", "COMPLETE")

                return self.navigation_functions.file_path_after
            else:
                self.navigation_functions.log_message("未选择文件", "INFO")

            return None
        finally:
            # 恢复原始log_message方法
            self.navigation_functions.log_message = original_log_message

    def save_image_to_dir(self, source_path, prefix=""):
        """
        保存图像到数据目录

        Args:
            source_path: 源文件路径
            prefix: 文件名前缀

        Returns:
            str: 保存后的文件路径，如果保存失败则返回None
        """
        try:
            # 获取文件名和扩展名
            file_name = os.path.basename(source_path)

            # 添加前缀（如果有）
            if prefix:
                file_name = f"{prefix}_{file_name}"

            # 构建目标路径
            # 使用源文件所在目录作为数据目录
            data_dir = os.path.dirname(source_path)
            target_path = os.path.join(data_dir, file_name)

            # 如果源路径和目标路径相同，则不需要复制
            if os.path.normpath(source_path) == os.path.normpath(target_path):
                return source_path

            # 复制文件
            import shutil
            shutil.copy2(source_path, target_path)

            return target_path

        except Exception as e:
            self.navigation_functions.log_message(f"保存图像失败: {str(e)}", "ERROR")
            return None
