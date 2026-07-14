"""
四联图可视化

入参: visualize_results(pre_img, post_img, pred_mask, save_path, is_raw_output)
方法: 缩放大图 → 染色变化区域 → 横排拼接四联图（前时相/后时相/检测叠加/二值掩膜）→ PIL 落盘
出参: bool，保存成功返回 True
"""
import os

import numpy as np
import cv2


def visualize_results(pre_img, post_img, pred_mask, save_path, is_raw_output=False):
    """
    可视化结果 - 将前后时相图像、变化检测结果和二值掩码共四张图横排显示

    入参:
        pre_img: 前时相图像（BGR uint8，与 cv2 约定一致）
        post_img: 后时相图像（BGR uint8）
        pred_mask: 预测掩膜；is_raw_output=True 时为 [0,1] 概率图，否则为 0/255 二值图
        save_path: 输出图片路径
        is_raw_output: pred_mask 是否为概率图
    方法: 大图先按比例缩放到 1200px 内 → 二值/概率掩膜染色（变化=红色）→
          四图横排拼接 → 顶部加英文标题 → PIL 保存（规避 cv2.imwrite 中文路径问题）
    出参: bool，保存成功 True，失败 False

    做什么: 生成可读的四联图供人工核查
    为什么: 取自 batch_image.py 带标题版本；并用 single_image.py 的 bool 返回值
            与异常兜底补全契约，使两个调用方都能基于返回值判定成功与否
    """
    # 如果图像太大，先缩小以便可视化
    max_size = 1200
    h, w = pre_img.shape[:2]
    if h > max_size or w > max_size:
        scale = min(max_size / h, max_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pre_img = cv2.resize(pre_img, (new_w, new_h))
        post_img = cv2.resize(post_img, (new_w, new_h))
        pred_mask = cv2.resize(pred_mask, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST if not is_raw_output else cv2.INTER_LINEAR)

    # 调整预测掩码为RGB
    h, w = pre_img.shape[:2]

    # 创建原始二值掩码的可视化（变化区域用绿色高亮，便于与红色叠加图区分）
    binary_mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    if is_raw_output:
        # 对于概率图，先二值化
        binary = (pred_mask > 0.5).astype(np.uint8) * 255
        binary_mask_colored[..., 0] = 0  # B通道
        binary_mask_colored[..., 1] = binary  # G通道
        binary_mask_colored[..., 2] = 0  # R通道
    else:
        # 已经是二值化的结果
        binary_mask_colored[..., 0] = 0  # B通道
        binary_mask_colored[..., 1] = pred_mask  # G通道
        binary_mask_colored[..., 2] = 0  # R通道

    # 处理叠加在原图上的变化检测结果
    if is_raw_output:
        # 创建热力图
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_colored = cv2.addWeighted(post_img, 0.7, heatmap, 0.3, 0)
    else:
        # 二值化结果，红色表示变化区域
        pred_colored = np.copy(post_img)
        pred_colored[..., 0] = np.where(pred_mask > 127, 255, post_img[..., 0])
        pred_colored[..., 1] = np.where(pred_mask > 127, 0, post_img[..., 1])
        pred_colored[..., 2] = np.where(pred_mask > 127, 0, post_img[..., 2])

    # 创建一个空白画布 - 现在需要容纳4张图像
    gap = 5  # 图像之间的间隔
    canvas_width = w * 4 + gap * 3
    canvas_height = h
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 在画布上放置四张图像
    canvas[0:h, 0:w] = pre_img
    canvas[0:h, w+gap:w*2+gap] = post_img
    canvas[0:h, w*2+gap*2:w*3+gap*2] = pred_colored
    canvas[0:h, w*3+gap*3:w*4+gap*3] = binary_mask_colored

    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (0, 0, 0)  # 黑色

    # 计算标题位置
    title1 = "Pre-event Image"
    title2 = "Post-event Image"
    title3 = "Change Detection"
    title4 = "Binary Mask"

    title1_size = cv2.getTextSize(title1, font, font_scale, font_thickness)[0]
    title2_size = cv2.getTextSize(title2, font, font_scale, font_thickness)[0]
    title3_size = cv2.getTextSize(title3, font, font_scale, font_thickness)[0]
    title4_size = cv2.getTextSize(title4, font, font_scale, font_thickness)[0]

    title1_x = int(w / 2 - title1_size[0] / 2)
    title2_x = int(w + gap + w / 2 - title2_size[0] / 2)
    title3_x = int(w * 2 + gap * 2 + w / 2 - title3_size[0] / 2)
    title4_x = int(w * 3 + gap * 3 + w / 2 - title4_size[0] / 2)

    # 在画布顶部添加标题
    canvas = cv2.copyMakeBorder(canvas, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, title1, (title1_x, 20), font, font_scale, text_color, font_thickness)
    cv2.putText(canvas, title2, (title2_x, 20), font, font_scale, text_color, font_thickness)
    cv2.putText(canvas, title3, (title3_x, 20), font, font_scale, text_color, font_thickness)
    cv2.putText(canvas, title4, (title4_x, 20), font, font_scale, text_color, font_thickness)

    # 保存结果
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 规范化路径，确保使用正确的路径分隔符
    normalized_save_path = os.path.normpath(save_path)

    try:
        # 直接使用PIL保存图像，跳过OpenCV的imwrite（规避中文路径写入失败）
        from PIL import Image
        Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(normalized_save_path)
        return True
    except Exception:
        # PIL 失败时退回 cv2.imwrite，仍失败则返回 False
        try:
            return bool(cv2.imwrite(normalized_save_path, canvas))
        except Exception:
            return False
