"""
像元 ↔ 地理坐标变换

入参: pixel_to_geo_coords(x, y, geo_transform) / transform_polygon_to_geo(polygon, geo_transform)
方法: 按 GDAL 仿射变换公式换算坐标；多边形遍历顶点逐个换算
出参: (geo_x, geo_y) / shapely.Polygon

注意: 本模块保留 single_raster.py 的 Y 方向反转语义（- y * abs(gt[5])），
      与 batch_raster.py 的加法实现存在【行为不一致-待确认】，重构期间未变更逻辑。
"""
from shapely.geometry import Polygon, LineString


def pixel_to_geo_coords(x, y, geo_transform):
    """
    将像素坐标转换为地理坐标

    【行为不一致-待确认】Y方向用减法反转（- y * abs(geo_transform[5])），
    与 batch_raster.py 中同名函数的加法实现（+ y * geo_transform[5]）不一致。
    GDAL标准约定 geo_transform[5] 为负值（北向上影像），此处两种写法在
    常规数据上结果可能相同，但极端数据会发散。重构期间仅标注，未变更逻辑。

    入参:
        x, y: 像素坐标
        geo_transform: GDAL地理变换参数（6 元组）

    出参:
        tuple: (地理x坐标, 地理y坐标)
    """
    geo_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2]
    # 反转Y坐标计算方向
    geo_y = geo_transform[3] + x * geo_transform[4] - y * abs(geo_transform[5])
    return geo_x, geo_y


def transform_polygon_to_geo(polygon, geo_transform):
    """
    将多边形的像素坐标转换为地理坐标

    入参:
        polygon: shapely 多边形（像素坐标）
        geo_transform: GDAL地理变换参数
    方法: 遍历外环与内环顶点，逐点调 pixel_to_geo_coords 换算；非法多边形 buffer(0) 修复
    出参: shapely.Polygon（地理坐标），无效返回 None

    做什么: 把矢量化得到的像素多边形转回真实地理坐标，供 GeoJSON/SHP 落盘
    为什么: 统一通过 pixel_to_geo_coords 计算坐标，消除 single_raster.py 历史中
            外环/内环/pixel_to_geo_coords 三处重复内联公式的维护负担
    """
    if not isinstance(polygon, Polygon) or not polygon.is_valid:
        return None

    try:
        # 转换外环
        exterior_coords = []
        for x, y in polygon.exterior.coords:
            geo_x, geo_y = pixel_to_geo_coords(x, y, geo_transform)
            exterior_coords.append((geo_x, geo_y))

        # 检查外环是否有足够的点
        if len(exterior_coords) < 3:
            return None

        # 转换内环（如果有）
        interior_coords = []
        for interior in polygon.interiors:
            hole_coords = []
            for x, y in interior.coords:
                geo_x, geo_y = pixel_to_geo_coords(x, y, geo_transform)
                hole_coords.append((geo_x, geo_y))
            if len(hole_coords) >= 3:  # 内环至少需要3个点
                interior_coords.append(hole_coords)

        # 创建新的多边形
        if interior_coords:
            geo_polygon = Polygon(exterior_coords, interior_coords)
        else:
            geo_polygon = Polygon(exterior_coords)

        # 验证多边形有效性
        if not geo_polygon.is_valid:
            geo_polygon = geo_polygon.buffer(0)
            if not geo_polygon.is_valid or geo_polygon.is_empty:
                return None

        return geo_polygon
    except Exception:
        return None


def transform_line_to_geo(line, geo_transform):
    """
    将线要素的像素坐标转换为地理坐标

    入参:
        line: shapely LineString（像素坐标）
        geo_transform: GDAL地理变换参数
    方法: 遍历顶点逐点调 pixel_to_geo_coords 换算
    出参: shapely.LineString（地理坐标），无效返回 None

    做什么: 线要素的坐标转换（与多边形版本对称）
    为什么: 历史代码中保留但未被主流程调用，统一收口到本模块便于未来按需启用
    """
    if not isinstance(line, LineString) or not line.is_valid:
        return None

    try:
        coords = []
        for x, y in line.coords:
            geo_x, geo_y = pixel_to_geo_coords(x, y, geo_transform)
            coords.append((geo_x, geo_y))

        if len(coords) < 2:
            return None

        geo_line = LineString(coords)
        if not geo_line.is_valid or geo_line.is_empty:
            return None

        return geo_line
    except Exception:
        return None
