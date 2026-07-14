"""
RSCD 遥感影像变化检测系统 - Frontend 视图功能模块
"""
import os
import sys

# 处理不同的导入情况
try:
    # 相对导入（当作为包导入时）
    from .image_import import ImportBeforeImage, ImportAfterImage
    from .change_detection import ExecuteChangeDetectionTask
    from .clear_task import ClearTask
    from .grid_cropping import GridCropping
    from .image_display import ImageDisplay
    from .batch_dialog import BatchProcessing
    from .training_dialog import TrainingModule

except ImportError:
    # 绝对导入（当直接运行时）
    try:
        from Frontend.views.image_import import ImportBeforeImage, ImportAfterImage
        from Frontend.views.change_detection import ExecuteChangeDetectionTask
        from Frontend.views.clear_task import ClearTask
        from Frontend.views.grid_cropping import GridCropping
        from Frontend.views.image_display import ImageDisplay
        from Frontend.views.batch_dialog import BatchProcessing
        from Frontend.views.training_dialog import TrainingModule

    except ImportError as e:
        # 记录错误但不抛出异常，允许程序继续运行
        import traceback
        print(f"警告: 功能模块导入出错: {e}")
        print(traceback.format_exc())

__all__ = [
    'GridCropping',
    'ImportBeforeImage',
    'ImportAfterImage',
    'ExecuteChangeDetectionTask',
    'ClearTask',
    'ImageDisplay',
    'BatchProcessing',
    'TrainingModule',
]
