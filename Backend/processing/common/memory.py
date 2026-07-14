"""
显存估算与 batch_size 自适应

入参: adjust_batch_size(patch_size, args) — args 需含 device/batch_size/auto_memory 字段
方法: 估算单 batch 显存占用 → 读取设备可用显存 → 反推最优 batch_size
出参: int，建议的 batch_size
"""
import torch


def estimate_memory_usage(patch_size, batch_size, device):
    """
    估计每批次的内存/显存占用

    入参:
        patch_size: 窗口边长
        batch_size: 批大小
        device: 设备标识（仅用于签名一致，未参与计算）
    方法: 按 float32 估算输入/输出/特征图字节数 + 固定模型开销
    出参: float，显存占用（MB）

    做什么: 提供粗略的显存需求估算
    为什么: 实测显存成本高，估算公式足以作为 batch_size 自适应的依据
    """
    # 假设使用的是float32类型的张量
    bytes_per_value = 4

    # 每个patch的输入(前后时相)和输出(预测)的大小
    input_size_per_patch = 2 * 3 * patch_size * patch_size  # 两个RGB图像
    output_size_per_patch = patch_size * patch_size  # 单通道输出

    # 模型参数和中间特征的大致估计
    model_overhead = 100 * 1024 * 1024  # 100MB固定开销
    feature_maps = 20 * 3 * patch_size * patch_size  # 粗略估计特征图占用

    # 批次的总开销
    batch_memory = batch_size * (input_size_per_patch + output_size_per_patch + feature_maps) * bytes_per_value

    # 总内存占用 (模型 + 批次数据)
    total_memory = model_overhead + batch_memory

    return total_memory / (1024 * 1024)  # 转换为MB


def adjust_batch_size(patch_size, args):
    """
    根据滑动窗口大小和可用内存调整批处理大小

    入参:
        patch_size: 窗口边长
        args: 需含 device/batch_size/auto_memory 字段
    方法: CPU 按 patch_size 阈值返回 2 或 4；GPU 读取可用显存反推并限制在 [1, 16]
    出参: int，调整后的 batch_size

    做什么: 防止 OOM 的自适应 batch_size 计算
    为什么: 不同 patch_size 与设备的显存容量差异巨大，固定 batch_size 易 OOM
    """
    # 对于CPU设备，根据patch_size调整批处理大小
    if str(args.device) == 'cpu':
        # 根据patch_size大小调整批处理大小，避免CPU内存不足
        if patch_size > 512:
            return min(args.batch_size, 2)  # 大patch使用更小的批次
        else:
            return min(args.batch_size, 4)  # 小patch可以用稍大的批次

    # 以下是GPU相关的逻辑
    if not args.auto_memory:
        return args.batch_size

    # 确保使用的是GPU设备
    if not torch.cuda.is_available():
        return args.batch_size

    try:
        # 获取GPU信息
        device_idx = int(str(args.device).split(':')[-1]) if ':' in str(args.device) else 0
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024 * 1024)  # 转换为MB
        allocated_memory = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)

        # 计算可用内存 (留出20%的余量)
        available_memory = (total_memory - allocated_memory) * 0.8

        # 估算每个批次所需内存
        memory_per_batch = estimate_memory_usage(patch_size, 1, args.device)

        # 计算最优批处理大小
        optimal_batch_size = max(1, int(available_memory / memory_per_batch))

        # 将批处理大小限制在合理范围内
        optimal_batch_size = min(optimal_batch_size, 16)
        optimal_batch_size = max(optimal_batch_size, 1)

        if optimal_batch_size != args.batch_size:
            return optimal_batch_size
        else:
            return args.batch_size

    except Exception:
        # 出错时使用默认批大小（吞掉异常以避免阻断推理）
        return args.batch_size
