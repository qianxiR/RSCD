"""
跨线程日志 Mixin

提供从工作线程安全地向 UI 日志控件写入消息的能力。

入参: 子类需提供 log_widget 属性（QListWidget）和可选的 grid_log_widget
方法: _add_log 跨线程派发 → _add_log_in_main_thread 槽函数更新 UI
出参: Mixin 类，供 QDialog/普通类多继承使用
"""
from PySide6.QtCore import Qt, QThread, QCoreApplication, QMetaObject, Slot, Q_ARG


class ThreadSafeLogMixin:
    """
    跨线程日志 Mixin

    入参: 子类需设置 self.log_widget（QListWidget）后调用 _add_log(message)
    方法: _add_log 检测当前线程 → 非主线程则 QMetaObject.invokeMethod 派发 →
          _add_log_in_main_thread 槽在主线程更新 UI 并滚动到底
    出参: 无返回值，副作用为向 log_widget 添加一行

    做什么: 让批量处理的工作线程能安全地把日志显示到 UI
    为什么: Qt 禁止在工作线程直接操作 UI 控件，必须通过 invokeMethod 切换线程；
            batch_dialog 与 raster_batch 各有一份近乎逐字相同的实现，统一消除重复
    """

    def _add_log(self, message):
        """
        添加日志消息（自动跨线程派发）

        入参: message — 日志文本
        方法: 当前在 UI 线程则直接调用槽；否则 invokeMethod 派发到主线程
        出参: 无
        """
        try:
            if QThread.currentThread() != QCoreApplication.instance().thread():
                QMetaObject.invokeMethod(
                    self,
                    "_add_log_in_main_thread",
                    Qt.QueuedConnection,
                    Q_ARG(str, message)
                )
            else:
                self._add_log_in_main_thread(message)
        except Exception as e:
            print(f"添加日志失败: {str(e)}")

    @Slot(str)
    def _add_log_in_main_thread(self, message):
        """
        在 UI 线程中向 log_widget 添加一行日志（槽函数）

        入参: message — 日志文本
        方法: addItem → scrollToBottom
        出参: 无

        注意: 子类需确保 self.log_widget 已初始化（QListWidget）。
              兼容历史命名：若子类用 preview_list 命名，可在 __init__ 中
              设置 self.log_widget = self.preview_list 即可复用本 Mixin。
        """
        try:
            log_widget = getattr(self, 'log_widget', None) or getattr(self, 'preview_list', None)
            if log_widget is None:
                return
            log_widget.addItem(message)
            log_widget.scrollToBottom()
        except Exception as e:
            print(f"在主线程中添加日志失败: {str(e)}")
