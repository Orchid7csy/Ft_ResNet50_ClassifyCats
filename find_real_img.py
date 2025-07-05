import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 全局常量 ---
MODEL_PATH = 'cats.keras'  # 您训练好的 ResNet50 模型路径
TARGET_SIZE = (224, 224)   # ResNet50 的目标输入尺寸
CLASSES = ['Pallas', 'Persian', 'Ragdoll', 'Singapura', 'Sphynx']


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
    return preprocess_input(img_array)          # 应用 ResNet50 的专属预处理

def classify(model, preprocessed_roi):
    """使用加载的模型进行分类"""
    if model is None:
        return "No Model", 0.0, -1

    prediction = model.predict(preprocessed_roi)[0] # 获取单个预测结果
    idx = np.argmax(prediction)
    prob = float(prediction[idx])
    label = CLASSES[idx]
    
    return label, prob, idx

def detect_a4_and_extract(frame):
    """
    在一帧图像中检测 A4 纸，返回透视矫正后的 A4 图像（如无检测返回 None）。
    """
    # 1. 预处理：灰度 -> 高斯模糊 -> 边缘
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # 2. 找外部轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 按面积降序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # A4 纸近似为四边形，且面积需足够大
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            pts = approx.reshape(4, 2)
            # 对角排序： tl, tr, br, bl
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            src = np.array([tl, tr, br, bl], dtype="float32")

            # 计算透视变换目标尺寸（A4 210×297 mm，按宽高比约 1:1.414）
            widthA  = np.linalg.norm(br - bl)
            widthB  = np.linalg.norm(tr - tl)
            maxW    = max(int(widthA), int(widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH    = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxW - 1, 0],
                [maxW - 1, maxH - 1],
                [0, maxH - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(frame, M, (maxW, maxH))
            return warped
    return None


def find_cat_photo(a4_img):
    """
    在透视矫正后的 A4 图像中，寻找满足条件的猫咪彩色照片区域：
    - 矩形轮廓
    - 区域内部有颜色（HSV 中 S 通道较高）
    - 区域周围为白色边框
    - 区域上方存在黑色文字（简单以黑色像素较多判断）
    返回 ROI 的坐标和图像或 None。
    """
    h, w = a4_img.shape[:2]
    gray = cv2.cvtColor(a4_img, cv2.COLOR_BGR2GRAY)
    # 对 A4 灰度图阈值，提取可能的内图矩形（区别于白色背景）
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:  # 过滤过小区域
            continue
        # 多边形近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        x, y, rw, rh = cv2.boundingRect(approx)
        # 确保矩形在 A4 较中央区域，且留有白边
        if x < 20 or y < 20 or x+rw > w-20 or y+rh > h-20:
            continue

        roi = a4_img[y:y+rh, x:x+rw]
        # 1) 检验彩色程度：HSV 中 S 通道均值
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s_mean = hsv[:, :, 1].mean()
        if s_mean < 30:  # 过低则可能只是黑白或文字
            continue

        # 2) 检验上方文字：在 roi 上方 10~30 像素高区域找黑色像素
        text_band = gray[max(y-30,0):y-10, x:x+rw]
        if text_band.size == 0:
            continue
        black_pixels = np.sum(text_band < 50)
        if black_pixels < (rw * 10 * 0.1):  # 黑色像素少于总像素 10%
            continue

        # 满足所有条件，认为找到有效照片
        return (x, y, rw, rh), roi

    return None, None


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

        a4 = detect_a4_and_extract(frame)
        display_frame = frame.copy() # 创建一个用于显示的副本

        if a4 is not None:
            display_frame = a4 # 如果检测到A4，则显示矫正后的图像
            bbox, cat_roi = find_cat_photo(a4)
            if bbox is not None:
                prep = preprocess_for_cnn(cat_roi)
                label, prob, idx = classify(model, prep)

                x, y, rw, rh = bbox
                cv2.rectangle(display_frame, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
                text = f"{label}: {prob:.2f}"
                cv2.putText(display_frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("A4 & Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
