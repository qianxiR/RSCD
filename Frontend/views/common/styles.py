"""
主题 QSS 生成器

集中存放批量对话框中重复的 QSS 字符串模板。

入参: 各函数接收 colors dict（来自 ThemeManager.get_colors）
方法: 用 colors 占位符填充 QSS 模板
出参: str，可直接传给 setStyleSheet
"""


def tab_widget_qss(colors):
    """
    生成 QTabWidget/QTabBar 的 QSS

    入参: colors — 主题色字典，需含 border/background/background_secondary/text/
          button_primary_bg/button_primary_text/button_secondary_hover
    方法: 返回 pane/tab-bar/tab/:selected/:hover:!selected 的完整 QSS
    出参: str

    做什么: 统一 Tab 控件在不同主题下的样式
    为什么: batch_dialog 的 apply_theme/update_theme 与 raster_batch 的 show_dialog
            三处有逐字节相同的 TabBar QSS，集中后避免维护时遗漏同步
    """
    return f"""
    QTabWidget::pane {{
        border: 1px solid {colors['border']};
        background-color: {colors['background']};
    }}
    QTabWidget::tab-bar {{
        left: 5px;
    }}
    QTabBar::tab {{
        background-color: {colors['background_secondary']};
        color: {colors['text']};
        padding: 8px 12px;
        margin-right: 2px;
        border: 1px solid {colors['border']};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{
        background-color: {colors['button_primary_bg']};
        color: {colors['button_primary_text']};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {colors['button_secondary_hover']};
    }}
    """


def line_edit_qss(colors):
    """
    生成 QLineEdit 的 QSS

    入参: colors — 主题色字典，需含 background_secondary/border/text
    方法: 返回 QLineEdit 的背景/边框/内边距 QSS
    出参: str

    做什么: 统一输入框样式
    为什么: batch_dialog 的 apply_theme 与 update_theme 两处重复
    """
    return f"""
    QLineEdit {{
        background-color: {colors['background_secondary']};
        color: {colors['text']};
        border: 1px solid {colors['border']};
        padding: 5px;
        border-radius: 3px;
    }}
    QLineEdit:focus {{
        border: 2px solid {colors['button_primary_bg']};
    }}
    """


def dialog_base_qss(colors):
    """
    生成 QDialog 基础 QSS（含 QLabel/QPushButton 等）

    入参: colors — 主题色字典
    方法: 返回对话框整体背景与基础控件样式
    出参: str

    做什么: 提供对话框主题切换的基础样式集
    为什么: batch_dialog.apply_theme 的主体 QSS，抽取后可在两个对话框复用
    """
    return f"""
    QDialog {{
        background-color: {colors['background']};
        color: {colors['text']};
    }}
    QLabel {{
        color: {colors['text']};
        background-color: transparent;
    }}
    QPushButton {{
        background-color: {colors['button_primary_bg']};
        color: {colors['button_primary_text']};
        border: none;
        padding: 6px 12px;
        border-radius: 3px;
    }}
    QPushButton:hover {{
        background-color: {colors['button_primary_hover']};
    }}
    QPushButton:pressed {{
        background-color: {colors['button_primary_pressed']};
    }}
    QPushButton:disabled {{
        background-color: {colors['background_secondary']};
        color: {colors['text_secondary']};
    }}
    QListWidget {{
        background-color: {colors['background_secondary']};
        color: {colors['text']};
        border: 1px solid {colors['border']};
        border-radius: 3px;
    }}
    """
