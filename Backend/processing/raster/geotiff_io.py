"""
GeoTIFF 读写

入参: read_geotiff(raster_path, selected_bands) / save_geotiff_result(mask, output_path, ...)
方法: GDAL 读取（含 NoData 与百分位拉伸）/ GDAL 写出（含边缘平滑与 LZW 压缩）
出参: (image, geo_transform, projection, ds) / bool
"""
import os

import numpy as np
import cv2
from osgeo import gdal


def read_geotiff(raster_path, selected_bands=None):
    """
    读取 GeoTIFF 栅格文件，保留地理参考信息

    入参:
        raster_path: 栅格文件路径
        selected_bands: 要读取的波段索引列表（GDAL 从 1 开始），None 时默认取前 3 波段
    方法: GDAL 打开 → 按精度规范 geo_transform → 单/多波段读取并补齐到 3 通道 →
          非 uint8 数据按 NoData 掩码做 2/98 百分位拉伸
    出参: tuple(image: np.uint8 (h,w,3), geo_transform: tuple, projection: str, ds: gdal.Dataset)

    做什么: 将任意波段/数据类型的栅格统一为 3 通道 uint8 图像 + 完整地理参考
    为什么: 取自 batch_raster.py 的健壮版本（vs single_raster 硬编码 RGB 波段、忽略 NoData），
            避免极值与 NoData 像素污染拉伸统计
    """
    try:
        # 打开栅格数据集
        ds = gdal.Open(raster_path)
        if ds is None:
            raise IOError(f"无法打开栅格文件: {raster_path}")

        # 获取栅格信息
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands_count = ds.RasterCount
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()

        # 统一处理地理变换参数的精度和类型
        geo_transform = list(geo_transform)
        # 对坐标值（第1和第4个参数）使用2位小数
        geo_transform[0] = float(f"{geo_transform[0]:.2f}")  # X坐标
        geo_transform[3] = float(f"{geo_transform[3]:.2f}")  # Y坐标
        # 对分辨率值（第2、第3、第5、第6个参数）使用6位小数
        geo_transform[1] = float(f"{geo_transform[1]:.6f}")  # X分辨率
        geo_transform[2] = float(f"{geo_transform[2]:.6f}")  # 行旋转
        geo_transform[4] = float(f"{geo_transform[4]:.6f}")  # 列旋转
        geo_transform[5] = float(f"{geo_transform[5]:.6f}")  # Y分辨率
        geo_transform = tuple(geo_transform)

        # 如果未指定波段，默认使用RGB波段（如果存在）
        if selected_bands is None:
            if bands_count >= 3:
                selected_bands = [1, 2, 3]  # RGB波段（GDAL波段索引从1开始）
            else:
                selected_bands = [1]  # 只选第一个波段

        # 确保选择的波段不超过可用波段数
        selected_bands = [b for b in selected_bands if b <= bands_count]
        if not selected_bands:
            raise ValueError(f"选择的波段{selected_bands}超出了可用范围(1-{bands_count})")

        # 读取选定的波段（保留最后一个 band 句柄用于后续 NoData 查询）
        band = None
        if len(selected_bands) == 1:
            # 单波段读取
            band = ds.GetRasterBand(selected_bands[0])
            image_array = band.ReadAsArray()
            # 转为RGB（三通道相同）
            image = np.stack([image_array, image_array, image_array], axis=2)
        else:
            # 多波段读取
            bands = []
            for band_idx in selected_bands[:3]:  # 最多取前3个波段作为RGB
                band = ds.GetRasterBand(band_idx)
                bands.append(band.ReadAsArray())

            # 创建RGB图像数组
            while len(bands) < 3:
                # 如果不足3个波段，复制最后一个波段
                bands.append(bands[-1])

            image = np.stack(bands, axis=2)

        # 数据类型转换和归一化
        if image.dtype != np.uint8:
            # 自动缩放到0-255
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                # 获取NoData值并安全处理（使用当前波段句柄）
                try:
                    nodata_value = band.GetNoDataValue() if band is not None else None
                    if nodata_value is not None:
                        non_nodata = (channel != nodata_value)
                    else:
                        non_nodata = np.ones_like(channel, dtype=bool)
                except Exception:
                    non_nodata = np.ones_like(channel, dtype=bool)

                if np.any(non_nodata):
                    # 使用百分位数进行拉伸，避免极值影响
                    try:
                        min_val = np.percentile(channel[non_nodata], 2)
                        max_val = np.percentile(channel[non_nodata], 98)
                        # 避免同值问题
                        if min_val == max_val:
                            min_val = np.min(channel[non_nodata])
                            max_val = np.max(channel[non_nodata])
                            if min_val == max_val:
                                min_val = 0
                                max_val = 255

                        # 应用拉伸
                        normalized = np.zeros_like(channel, dtype=np.float32)
                        normalized[non_nodata] = np.clip(channel[non_nodata], min_val, max_val)
                        normalized[non_nodata] = ((normalized[non_nodata] - min_val) / (max_val - min_val) * 255)
                        image[:, :, i] = normalized.astype(np.uint8)
                    except Exception:
                        # 简单线性拉伸作为备选方案
                        min_val = np.min(channel)
                        max_val = np.max(channel)
                        if min_val != max_val:
                            image[:, :, i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            image[:, :, i] = np.zeros_like(channel, dtype=np.uint8)

            # 确保最终结果是uint8
            image = image.astype(np.uint8)

        # 检查图像是否有效
        if np.isnan(image).any() or np.isinf(image).any():
            image = np.nan_to_num(image, nan=0, posinf=255, neginf=0).astype(np.uint8)

        return image, geo_transform, projection, ds

    except Exception:
        raise


def save_geotiff_result(mask, output_path, geo_transform=None, projection=None, src_ds=None):
    """
    保存带有地理参考信息的 GeoTIFF 结果

    入参:
        mask: 二值掩膜或概率图（uint8 或 float）
        output_path: 输出路径
        geo_transform: 地理变换参数（可None）
        projection: 投影 WKT（可None）
        src_ds: 源数据集（保留参数以兼容调用方，当前实现未使用）
    方法: uint8 掩膜先经中值滤波+形态学开闭运算+高斯模糊再二值化以平滑齿状边缘 →
          GDAL GTiff 写出（LZW 压缩，写入 geo/projection 与 NoData=0）
    出参: bool，成功 True；GDAL 失败时退回 cv2.imwrite，再失败 False

    做什么: 输出可与原栅格叠合的地理参考 GeoTIFF
    为什么: 取自 single_raster.py 的带边缘平滑版本（vs batch_raster 无平滑），
            形态学平滑能显著降低滑窗接缝处的锯齿
    """
    try:
        # 对掩码进行边缘平滑处理，减少齿状边缘
        if mask.dtype == np.uint8:
            # 对二值掩码应用平滑处理
            # 使用中值滤波平滑边缘
            smooth_mask = cv2.medianBlur(mask, 5)
            # 使用形态学操作进一步平滑边缘
            kernel = np.ones((3, 3), np.uint8)
            smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            # 使用高斯模糊后重新二值化
            smooth_mask = cv2.GaussianBlur(smooth_mask.astype(np.float32), (5, 5), 1.0)
            mask = (smooth_mask > 127).astype(np.uint8) * 255

        # 获取数据类型
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            gdal_dtype = gdal.GDT_Float32
        else:
            gdal_dtype = gdal.GDT_Byte

        # 创建驱动
        driver = gdal.GetDriverByName('GTiff')

        # 创建数据集
        ds = driver.Create(
            output_path,
            mask.shape[1],  # 宽度
            mask.shape[0],  # 高度
            1,              # 波段数
            gdal_dtype,
            options=['COMPRESS=LZW']  # 使用LZW压缩
        )

        # 设置地理参考信息
        if geo_transform is not None:
            ds.SetGeoTransform(geo_transform)

        if projection is not None:
            ds.SetProjection(projection)

        # 写入数据
        ds.GetRasterBand(1).WriteArray(mask)

        # 设置NoData值
        ds.GetRasterBand(1).SetNoDataValue(0)

        # 刷新
        ds.FlushCache()

        # 关闭数据集
        ds = None

        return True

    except Exception:
        # GDAL 失败时退回 cv2 普通保存
        try:
            return bool(cv2.imwrite(output_path, mask))
        except Exception:
            return False
