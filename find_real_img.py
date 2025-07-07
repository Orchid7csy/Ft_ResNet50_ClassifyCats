import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- 全局常量 ---
MODEL_PATH = 'cats.keras'  # 您训练好的 ResNet50 模型路径
TARGET_SIZE = (224, 224)   # ResNet50 的目标输入尺寸
classes = ['Pallas', 'Persian', 'Ragdoll', 'Singapura', 'Sphynx']


def load_cat_model(path):
    """加载 .keras 模型"""
    print(f"Loading model from {path}...")
    try:
        model = load_model(path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_for_cnn(roi):
    """
    使用 'Letterboxing' 方法预处理ROI以送入CNN，不丢失长宽比信息。
    """
    h, w, _ = roi.shape
    target_h, target_w = TARGET_SIZE

    # 计算缩放比例，并确定缩放后的新尺寸
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 保持长宽比缩放
    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建一个灰色的画布
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)

    # 计算粘贴位置（使其居中）
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # 将缩放后的图像粘贴到画布中央
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_roi

    # 准备送入模型
    img_array = np.expand_dims(canvas, axis=0)  # 增加 batch 维度 -> (1, 224, 224, 3)
    # img_array = img_array.astype(np.float32)    # 确保数据类型为float32
    # img_array = img_array / 255.0               # 归一化到[0,1]范围，与训练时一致    
    # return img_array          # 应用 ResNet50 的专属预处理
    return preprocess_input(img_array)  # 使用 ResNet50 的预处理函数

def classify(model, preprocessed_roi):
    """使用加载的模型进行分类"""
    if model is None:
        return "No Model", 0.0, -1

    prediction = model.predict(preprocessed_roi)[0] # 获取单个预测结果
    idx = np.argmax(prediction)
    prob = float(prediction[idx])
    label = classes[idx]
    
    return label, prob, idx

# def detect_a4_and_extract(frame):
#     """
#     在一帧图像中检测 A4 纸，返回透视矫正后的 A4 图像（如无检测返回 None）。
#     """
#     # 1. 预处理：灰度 -> 高斯模糊 -> 边缘
#     gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5,5), 0)
#     edges   = cv2.Canny(blurred, 50, 150)

#     # 2. 找外部轮廓
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 按面积降序
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

#     for cnt in contours:
#         peri   = cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#         # A4 纸近似为四边形，且面积需足够大
#         if len(approx) == 4 and cv2.contourArea(approx) > 10000:
#             pts = approx.reshape(4, 2)
#             # 对角排序： tl, tr, br, bl
#             s = pts.sum(axis=1)
#             diff = np.diff(pts, axis=1)
#             tl = pts[np.argmin(s)]
#             br = pts[np.argmax(s)]
#             tr = pts[np.argmin(diff)]
#             bl = pts[np.argmax(diff)]
#             src = np.array([tl, tr, br, bl], dtype="float32")

#             # 计算透视变换目标尺寸（A4 210×297 mm，按宽高比约 1:1.414）
#             widthA  = np.linalg.norm(br - bl)
#             widthB  = np.linalg.norm(tr - tl)
#             maxW    = max(int(widthA), int(widthB))
#             heightA = np.linalg.norm(tr - br)
#             heightB = np.linalg.norm(tl - bl)
#             maxH    = max(int(heightA), int(heightB))

#             dst = np.array([
#                 [0, 0],
#                 [maxW - 1, 0],
#                 [maxW - 1, maxH - 1],
#                 [0, maxH - 1]
#             ], dtype="float32")

#             M = cv2.getPerspectiveTransform(src, dst)
#             warped = cv2.warpPerspective(frame, M, (maxW, maxH))
#             return warped
#     return None

def find_cat_with_haar(a4_img):
    """
    使用Haar Cascades检测A4图像中的猫脸区域。
    
    Args:
        a4_img: 透视矫正后的A4图像
    
    Returns:
        tuple: (bbox, roi) 其中bbox为(x,y,w,h)，roi为猫脸区域图像；如果未检测到则返回(None, None)
    """
    gray = cv2.cvtColor(a4_img, cv2.COLOR_BGR2GRAY)
    
    # 加载OpenCV自带的猫脸haar特征分类器
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
    if not os.path.exists(cascade_path):
        print("未找到猫脸检测器haar文件，请安装opencv-data或手动下载。")
        return None, None
    
    cat_cascade = cv2.CascadeClassifier(cascade_path)
    
    # 多尺度检测猫脸
    cats = cat_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,   # 更细致的尺度缩放
        minNeighbors=5,     # 更严格的邻域验证
        minSize=(80, 80),   # 只检测较大猫脸
        # maxSize=(300, 300)  # 限制最大尺寸
        # flags=cv2.CASCADE_SCALE_IMAGE  # 可选：优化检测性能
    )
    
    if len(cats) > 0:
        # 选择面积最大的猫脸区域
        areas = [w * h for (x, y, w, h) in cats]
        max_idx = np.argmax(areas)
        x, y, w, h = cats[max_idx]
        
        # 提取猫脸区域，稍微扩大边界以包含更多上下文
        margin = 20
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(a4_img.shape[1], x + w + margin)
        y_end = min(a4_img.shape[0], y + h + margin)
        
        roi = a4_img[y_start:y_end, x_start:x_end]
        bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
        
        print(f"检测到猫脸区域: {bbox}")
        return bbox, roi
    else:
        print("未检测到猫脸")
        return None, None
        
# def find_cat_photo(a4_img):
#     """
#     在透视矫正后的 A4 图像中，寻找满足条件的猫咪彩色照片区域：
#     - 矩形轮廓
#     - 区域内部有颜色（HSV 中 S 通道较高）
#     - 区域周围为白色边框
#     - 区域上方存在黑色文字（简单以黑色像素较多判断）
#     返回 ROI 的坐标和图像或 None。
#     """
#     h, w = a4_img.shape[:2]
#     gray = cv2.cvtColor(a4_img, cv2.COLOR_BGR2GRAY)
#     # 对 A4 灰度图阈值，提取可能的内图矩形（区别于白色背景）
#     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < 5000:  # 过滤过小区域
#             continue
#         # 多边形近似
#         peri = cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#         if len(approx) != 4:
#             continue

#         x, y, rw, rh = cv2.boundingRect(approx)
#         # 确保矩形在 A4 较中央区域，且留有白边
#         if x < 20 or y < 20 or x+rw > w-20 or y+rh > h-20:
#             continue

#         roi = a4_img[y:y+rh, x:x+rw]
#         # 1) 检验彩色程度：HSV 中 S 通道均值
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         s_mean = hsv[:, :, 1].mean()
#         if s_mean < 30:  # 过低则可能只是黑白或文字
#             continue

#         # 2) 检验上方文字：在 roi 上方 10~30 像素高区域找黑色像素
#         text_band = gray[max(y-30,0):y-10, x:x+rw]
#         if text_band.size == 0:
#             continue
#         black_pixels = np.sum(text_band < 50)
#         if black_pixels < (rw * 10 * 0.1):  # 黑色像素少于总像素 10%
#             continue

#         # 满足所有条件，认为找到有效照片
#         return (x, y, rw, rh), roi

#     return None, None


def main():
    model = load_cat_model(MODEL_PATH)
    if model is None:
        return
        
    cap = cv2.VideoCapture(0)
    # 设置摄像头分辨率为您的640x480输入
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy() # 创建一个用于显示的副本

        # === A4检测路线（已注释） ===
        # a4 = detect_a4_and_extract(frame)
        # if a4 is not None:
        #     display_frame = a4 # 如果检测到A4，则显示矫正后的图像
        #     # 使用原版基于颜色和轮廓的猫咪照片检测
        #     bbox, cat_roi = find_cat_photo(a4)
        #     # 或者使用Haar检测
        #     # bbox, cat_roi = find_cat_with_haar(a4)
        
        # === 直接Haar猫脸检测路线 ===
        bbox, cat_roi = find_cat_with_haar(frame)
        
        if bbox is not None and cat_roi is not None:
            prep = preprocess_for_cnn(cat_roi)
            label, prob, idx = classify(model, prep)

            x, y, rw, rh = bbox
            cv2.rectangle(display_frame, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            text = f"{label}: {prob:.2f}"
            cv2.putText(display_frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Direct Cat Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
