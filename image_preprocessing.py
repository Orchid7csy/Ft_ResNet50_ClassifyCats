import cv2
import numpy as np
from PIL import Image

def letterbox_preprocess(img, target_size):
    """
    对图像进行letterboxing预处理，保持原始宽高比
    
    Args:
        img: 输入图像 (numpy array, RGB格式)
        target_size: 目标尺寸 (height, width)
    
    Returns:
        处理后的图像 (numpy array, RGB格式)
    """
    target_h, target_w = target_size
    
    # 获取原始图像尺寸
    h, w = img.shape[:2]
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    
    # 计算缩放后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建目标尺寸的黑色背景
    letterboxed_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 计算居中位置
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # 将缩放后的图像放置在中心
    letterboxed_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    
    return letterboxed_img