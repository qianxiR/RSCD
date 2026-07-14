"""
通用工具函数

入参: parse_grid_size(text)
方法: 解析 "行,列" 格式字符串
出参: tuple[int, int]
"""


def parse_grid_size(text):
    """
    解析网格大小文本

    入参: text — 形如 "2,3" 的字符串
    方法: split(",") → map(int) → 校验正整数
    出参: tuple(rows, cols)

    做什么: 把 UI 输入框的网格规格文本解析为行列整数
    为什么: batch_dialog 与 raster_batch 各有一份相同的解析+校验逻辑，
            集中后避免校验规则分歧

    Raises:
        ValueError: 格式错误或非正整数时抛出，由调用方决定如何提示用户
    """
    rows, cols = map(int, text.strip().split(","))
    if rows <= 0 or cols <= 0:
        raise ValueError("网格行数和列数必须为正整数")
    return rows, cols
