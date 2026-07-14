"""
掩膜矢量化与矢量导出

入参: mask_to_lines(mask, ...) / export_vector(polygons, areas, output_path, ...)
方法: 二值掩膜经形态学平滑 → rasterio.features.shapes 提取多边形 → shapely 简化 →
      GeoDataFrame 转地理坐标后写出 SHP/GeoJSON
出参: (polygons, areas) / bool
"""
import os

import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from rasterio import features
from osgeo import osr

from .geo_transform import transform_polygon_to_geo


def mask_to_lines(mask, min_area=100, simplify=True, simplify_tolerance=0.5):
    """
    从二值掩码中提取矢量多边形

    入参:
        mask: 二值掩码（任意 dtype，函数内部统一为 0/1 的 uint8）
        min_area: 最小多边形面积（像素单位），小于此值的多边形被过滤
        simplify: 是否对多边形做 Douglas-Peucker 简化
        simplify_tolerance: 简化容差（实际放大 2 倍传给 simplify）
    方法: 数据类型归一化 → 形态学开闭+中值+高斯平滑边缘 → rasterio.features.shapes
          提取 → shapely 转换 + 面积过滤 + 简化 + 有效性校验
    出参: tuple(list[shapely.Polygon], list[float])，分别多边形与对应像素面积

    做什么: 把栅格变化掩膜转成可导出的矢量多边形列表
    为什么: 取自 single_raster.py 版本（vs batch_raster 简化版），保留更强的边缘平滑，
            使导出的矢量边界更光滑、噪声更少
    """
    # 确保掩码是有效的二维数组
    if mask is None or mask.size == 0:
        return [], []

    # 首先确保mask是2D数组
    if len(mask.shape) > 2:
        # 如果是多通道图像，转换为单通道
        mask = mask[:, :, 0] if mask.shape[2] > 0 else np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    # 智能数据类型处理
    if mask.dtype != np.uint8:
        if mask.dtype == np.bool_ or mask.dtype == bool:
            # 如果是布尔类型，直接转换为0和1
            binary_mask = mask.astype(np.uint8)
        elif np.issubdtype(mask.dtype, np.floating):
            # 如果是浮点类型，使用阈值二值化
            binary_mask = (mask > 0.5).astype(np.uint8)
        else:
            # 其他整数类型，确保只有0和1
            binary_mask = (mask > 0).astype(np.uint8)
    else:
        # 对于uint8类型，确保只有0和1或0和255
        if np.max(mask) > 1:
            binary_mask = (mask > 127).astype(np.uint8)
        else:
            binary_mask = mask

    # 增强边缘平滑处理 - 增加更大的核和更多迭代次数
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)

    # 先腐蚀再膨胀（开运算）去除小噪点
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # 先膨胀再腐蚀（闭运算）填充小孔洞
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

    # 应用中值滤波平滑边缘
    binary_mask = cv2.medianBlur(binary_mask, 5)

    # 使用更强的高斯滤波平滑边界
    smoothed_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (9, 9), 2.0)
    binary_mask = (smoothed_mask > 0.5).astype(np.uint8)

    # 如果掩码全为0，则直接返回空列表
    if np.sum(binary_mask) == 0:
        return [], []

    try:
        # 使用 rasterio.features 提取形状
        # 注意：rasterio 合法 API 为 features.shapes（小写 s）
        shapes = features.shapes(binary_mask, mask=binary_mask > 0, connectivity=8)

        # 转换为shapely几何对象
        polygons = []
        areas = []  # 存储对应的面积

        for geom, value in shapes:
            if value == 0:  # 跳过背景
                continue

            try:
                # 转换为shapely对象
                polygon = shape(geom)

                # 只处理多边形类型
                if not isinstance(polygon, (Polygon, MultiPolygon)):
                    continue

                # 按面积过滤
                if polygon.area < min_area:
                    continue

                # 确保是有效的多边形
                if not polygon.is_valid:
                    # 尝试修复无效多边形
                    polygon = polygon.buffer(0)
                    if not polygon.is_valid or polygon.is_empty:
                        continue

                # 简化多边形（如果需要）- 增加简化强度
                if simplify and simplify_tolerance > 0:
                    # 使用Douglas-Peucker算法简化多边形，增大容差值
                    polygon = polygon.simplify(simplify_tolerance * 2, preserve_topology=True)

                # 再次检查简化后的多边形
                if not polygon.is_valid or polygon.is_empty or polygon.area < min_area:
                    continue

                # 添加到结果列表
                polygons.append(polygon)
                areas.append(polygon.area)
            except Exception:
                # 如果处理单个多边形时出错，跳过此多边形
                continue

        return polygons, areas
    except Exception:
        # 如果整个处理过程出错，返回空结果
        return [], []


def export_vector(polygons, areas, output_path, geo_transform=None, projection=None,
                  attributes=None, export_format='shp'):
    """
    导出矢量文件

    入参:
        polygons: shapely 多边形列表（像素坐标）
        areas: 对应的像素面积列表
        output_path: 输出文件路径
        geo_transform: 地理变换参数（None 时按像素坐标直接导出）
        projection: 投影 WKT（None 时默认 EPSG:4326）
        attributes: 额外属性字典 {字段名: [值列表]}
        export_format: 'shp'（ESRI Shapefile）或 'geojson'
    方法: 多边形像素→地理坐标 → 构建 GeoDataFrame（含 ID/Area/Perimeter/自定义属性）→
          设置 CRS → 写出；失败时尝试 buffer(0) 修复后重试
    出参: bool，是否成功导出

    做什么: 把矢量化结果落盘为可在 GIS 软件打开的 SHP/GeoJSON
    为什么: 取自 single_raster.py 版本，坐标转换通过 transform_polygon_to_geo 集中处理，
            避免 batch_raster 内联坐标公式导致的重复与不一致
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 如果多边形列表为空，返回失败
        if not polygons or len(polygons) == 0:
            return False

        # 准备几何对象列表（像素坐标 → 地理坐标）
        geometries = []
        for polygon in polygons:
            # 只处理多边形类型
            if isinstance(polygon, (Polygon, MultiPolygon)):
                if geo_transform is not None:
                    try:
                        # 转换多边形坐标
                        if isinstance(polygon, Polygon):
                            geo_polygon = transform_polygon_to_geo(polygon, geo_transform)
                            if geo_polygon and geo_polygon.is_valid and not geo_polygon.is_empty:
                                geometries.append(geo_polygon)

                        # 处理MultiPolygon
                        elif isinstance(polygon, MultiPolygon):
                            geo_polygons = []
                            for single_polygon in polygon.geoms:
                                geo_polygon = transform_polygon_to_geo(single_polygon, geo_transform)
                                if geo_polygon and geo_polygon.is_valid and not geo_polygon.is_empty:
                                    geo_polygons.append(geo_polygon)

                            if geo_polygons:
                                geometries.append(MultiPolygon(geo_polygons))
                    except Exception:
                        # 坐标转换失败时退回原始多边形
                        if polygon.is_valid and not polygon.is_empty:
                            geometries.append(polygon)
                else:
                    # 没有地理变换，直接添加像素坐标多边形
                    if polygon.is_valid and not polygon.is_empty:
                        geometries.append(polygon)

        # 转换后为空则失败
        if not geometries:
            return False

        # 准备属性数据
        attr_data = {"ID": list(range(1, len(geometries) + 1))}

        # 添加面积属性
        if areas is not None:
            attr_data["Area"] = []
            attr_data["Perimeter"] = []
            for i, geom in enumerate(geometries):
                if i < len(areas):
                    attr_data["Area"].append(float(areas[i]))
                    try:
                        attr_data["Perimeter"].append(float(geom.length))
                    except Exception:
                        attr_data["Perimeter"].append(0.0)
                else:
                    attr_data["Area"].append(0.0)
                    attr_data["Perimeter"].append(0.0)

        # 添加自定义属性
        if attributes is not None:
            for attr_name, attr_values in attributes.items():
                attr_data[attr_name] = []
                for i in range(len(geometries)):
                    if i < len(attr_values):
                        attr_data[attr_name].append(attr_values[i])
                    else:
                        attr_data[attr_name].append(None)

        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame(attr_data, geometry=geometries)

        # 设置坐标参考系统
        crs = _resolve_crs(projection)
        if crs:
            try:
                gdf.crs = crs
            except Exception:
                pass

        # 导出文件
        try:
            _write_vector(gdf, output_path, export_format)
            return True
        except Exception:
            # 导出失败，尝试 buffer(0) 修复无效几何后重试
            try:
                fixed_geometries = []
                for geom in geometries:
                    if not geom.is_valid:
                        fixed_geom = geom.buffer(0)
                        if fixed_geom.is_valid and not fixed_geom.is_empty:
                            fixed_geometries.append(fixed_geom)
                    else:
                        fixed_geometries.append(geom)

                if fixed_geometries:
                    gdf = gpd.GeoDataFrame(attr_data, geometry=fixed_geometries)
                    if crs:
                        try:
                            gdf.crs = crs
                        except Exception:
                            pass
                    _write_vector(gdf, output_path, export_format)
                    return True
                return False
            except Exception:
                return False
    except Exception:
        return False


def _resolve_crs(projection):
    """
    从投影 WKT 解析 CRS 字符串

    入参: projection — WKT 字符串或 None
    方法: osr.SpatialReference 导入 WKT，优先取 EPSG 编码，否则回退 WKT 本体或 EPSG:4326
    出参: str 形式的 CRS（"EPSG:xxxx" 或 WKT），无法解析返回 None
    """
    if projection:
        try:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(projection)
            epsg = srs.GetAuthorityCode(None)
            if epsg:
                return f"EPSG:{epsg}"
            return projection
        except Exception:
            return "EPSG:4326"
    return "EPSG:4326"


def _write_vector(gdf, output_path, export_format):
    """
    按 format 驱动写出 GeoDataFrame

    入参: gdf / output_path / export_format('shp'|'geojson')
    方法: geojson 用 GeoJSON 驱动，其余用 ESRI Shapefile
    出参: 无（gdf.to_file 自身返回 None，失败抛异常由调用方捕获）
    """
    if export_format.lower() == 'geojson':
        gdf.to_file(output_path, driver='GeoJSON')
    else:
        gdf.to_file(output_path, driver='ESRI Shapefile')
