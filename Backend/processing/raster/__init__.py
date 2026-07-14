"""
RSCD Backend 栅格公共层

集中存放 single_raster / batch_raster 共享的栅格 IO、坐标变换、矢量导出逻辑。
依赖 GDAL / rasterio / geopandas / shapely，仅在 GDAL 可用时由 processing/__init__ 导入。

入参: 见各子模块
方法: 提供 GeoTIFF 读写、像元↔地理坐标转换、掩膜矢量化三类工具
出参: 图像/掩膜数组、地理坐标、矢量文件
"""
from .geotiff_io import read_geotiff, save_geotiff_result
from .geo_transform import pixel_to_geo_coords, transform_polygon_to_geo, transform_line_to_geo
from .vector_export import mask_to_lines, export_vector

__all__ = [
    "read_geotiff",
    "save_geotiff_result",
    "pixel_to_geo_coords",
    "transform_polygon_to_geo",
    "transform_line_to_geo",
    "mask_to_lines",
    "export_vector",
]
