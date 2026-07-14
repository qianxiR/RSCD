"""
滑窗与权重融合工具

入参: 见各函数签名
方法: 在图像上生成滑窗坐标、生成融合权重核、按 args 计算最优 patch/stride
出参: 坐标列表 / 权重 numpy 数组 / patch_size 整数 / stride 整数
"""
import numpy as np


def create_sliding_windows(image_shape, patch_size, stride):
    """
    创建滑动窗口坐标列表

    入参:
        image_shape: 图像形状 (h, w[, c])
        patch_size: 窗口边长
        stride: 滑动步长
    方法: 双层循环枚举窗口起点，最后一个窗口对齐到图像边缘以保证全覆盖
    出参: List[Tuple[int, int, int, int]]，元素为 (x1, y1, x2, y2)

    做什么: 生成覆盖整张图像的窗口坐标（含边缘对齐）
    为什么: 步长不能整除图像尺寸时，常规枚举会漏掉右下角，对齐避免漏检
    """
    h, w = image_shape[:2]
    windows = []

    # 确保最后一个窗口能覆盖到图像边缘
    for y in range(0, h - patch_size + 1, stride):
        # 调整最后一个窗口以确保覆盖图像边缘
        if y + stride > h - patch_size and y < h - patch_size:
            y = h - patch_size

        for x in range(0, w - patch_size + 1, stride):
            # 调整最后一个窗口以确保覆盖图像边缘
            if x + stride > w - patch_size and x < w - patch_size:
                x = w - patch_size

            windows.append((x, y, x + patch_size, y + patch_size))

    return windows


def create_weight_map(patch_size, stride, weight_type='gaussian'):
    """
    创建权重图，用于融合重叠区域

    入参:
        patch_size: 窗口边长
        stride: 滑动步长（当前实现未直接使用，保留以兼容调用约定）
        weight_type: 'gaussian'（中心高斯）/ 'linear'（边缘线性衰减）/ 其他（均匀）
    方法: 按类型生成二维权重核并归一化
    出参: numpy.ndarray，形状 (patch_size, patch_size)

    做什么: 生成滑窗重叠区域的融合权重，使中心像素权重高于边缘
    为什么: 滑窗重叠区会被多次预测，简单平均会使边缘预测（视野受限）权重过高
    """
    if weight_type == 'gaussian':
        # 创建二维高斯权重
        sigma = patch_size / 4
        x = np.linspace(-patch_size/2, patch_size/2, patch_size)
        y = np.linspace(-patch_size/2, patch_size/2, patch_size)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        # 归一化
        kernel = kernel / kernel.max()
        return kernel
    elif weight_type == 'linear':
        # 创建线性边缘衰减权重
        x = np.linspace(0, 1, patch_size//2)
        y = np.linspace(0, 1, patch_size//2)
        xx, yy = np.meshgrid(x, y)
        center = np.ones((patch_size//2, patch_size//2))
        top_left = xx * yy
        top_right = np.fliplr(xx) * yy
        bottom_left = xx * np.flipud(yy)
        bottom_right = np.fliplr(xx) * np.flipud(yy)

        kernel = np.zeros((patch_size, patch_size))
        kernel[:patch_size//2, :patch_size//2] = top_left
        kernel[:patch_size//2, patch_size//2:] = top_right
        kernel[patch_size//2:, :patch_size//2] = bottom_left
        kernel[patch_size//2:, patch_size//2:] = center
        return kernel
    else:
        # 均匀权重
        return np.ones((patch_size, patch_size))


def determine_optimal_patch_size(image_shape, args):
    """
    根据图像尺寸自动确定最优滑动窗口大小

    入参:
        image_shape: 图像形状 (h, w[, c])
        args: 需含 patch_size/auto_patch_divisor/max_patch_size/min_patch_size 字段
    方法: 用户指定则用指定值；否则按图像短边除以 auto_patch_divisor 估算，
          并对齐到 16 的倍数（模型下采样因子）
    出参: int，patch_size

    做什么: 自适应计算 patch 大小，使其既能覆盖足够上下文又不超过显存预算
    为什么: 固定 patch 在差异巨大的图像尺寸下要么浪费显存要么视野不足
    """
    h, w = image_shape[:2]

    # 如果用户指定了块大小，则使用用户设置
    if args.patch_size > 0:
        patch_size = args.patch_size
        return patch_size

    # 根据图像尺寸自动计算合适的块大小
    # 目标是将图像分成大约 auto_patch_divisor 个块
    h_patch = max(min(h // args.auto_patch_divisor, args.max_patch_size), args.min_patch_size)
    w_patch = max(min(w // args.auto_patch_divisor, args.max_patch_size), args.min_patch_size)

    # 确保块大小是模型输入的倍数(通常是8或16，这里取16)
    model_factor = 16
    h_patch = (h_patch // model_factor) * model_factor
    w_patch = (w_patch // model_factor) * model_factor

    # 为简单起见，使用正方形块
    patch_size = min(h_patch, w_patch)

    # 确保块大小至少为最小值
    patch_size = max(patch_size, args.min_patch_size)

    return patch_size


def determine_optimal_stride(patch_size, args):
    """
    根据块大小确定合适的步长

    入参:
        patch_size: 已确定的窗口边长
        args: 需含 stride_ratio 字段（默认 0.5~0.75）
    方法: stride = patch_size * stride_ratio，下限 32 像素
    出参: int，stride

    做什么: 由 patch_size 与重叠比例反推步长，保证最小重叠 32 像素
    为什么: 取自 batch_image.py 的灵活版本（依赖 stride_ratio），
            取代 single_image.py 中硬编码 patch_size//2 的实现
    """
    stride = int(patch_size * args.stride_ratio)
    # 确保步长至少为32像素，避免极端大 patch 下步长过小导致窗口数量爆炸
    stride = max(stride, 32)
    return stride
