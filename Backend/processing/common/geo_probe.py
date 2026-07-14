"""
地理坐标探测

入参: has_georeference(raster_path) — 影像文件路径
方法: GDAL 打开 → 取 GeoTransform + Projection → 两者皆有效判定为有坐标
出参: bool，有坐标返回 True；无坐标/GDAL 不可用/非栅格返回 False

做什么: 统一影像流程下，决定输出是否保留坐标系（GeoTIFF+矢量 vs PNG）的唯一判据
为什么: 取消图像版/影像版二元划分后，是否保留坐标完全由影像自身属性决定，
        不再由用户模式选择；GDAL 不可用时统一退化为无坐标路径，保证流程不中断
"""


def has_georeference(raster_path):
    """
    判断影像是否带有有效地理坐标参考

    入参:
        raster_path: 影像文件路径（GeoTIFF/TIFF/PNG/JPG 等任意可被 GDAL 打开的格式）
    方法: GDAL 打开 → 读取 GeoTransform 与 Projection →
          GeoTransform 六参数全为 0 视为无效（GDAL 对无坐标文件的默认占位）→
          Projection 为空字符串视为无投影 → 两者皆有效返回 True
    出参: bool；任一条件不满足或 GDAL 不可用或打开失败返回 False

    做什么: 提供坐标系保留与否的唯一判定入口
    为什么: 参考 batch_raster.py 中 has_geo = geotransform is not None and projection
            的既有判定，补充对全零 GeoTransform（GDAL 默认占位）的剔除，避免误判
    """
    try:
        from osgeo import gdal
    except ImportError:
        return False

    try:
        dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if dataset is None:
            return False

        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        dataset = None

        if geo_transform is None:
            return False

        # GDAL 对无坐标文件返回 (0, 1, 0, 0, 0, 1) 默认占位，需剔除
        is_default_placeholder = all(
            abs(v - default) < 1e-9
            for v, default in zip(geo_transform, (0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        )
        if is_default_placeholder:
            return False

        if not projection or not projection.strip():
            return False

        return True
    except Exception:
        return False
