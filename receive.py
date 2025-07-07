import tensorflow as tf
import tf_keras
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2
import queue # <-- 引入队列模块
import threading # <-- 引入线程模块
import time
from image_preprocessing import letterbox_preprocess
from find_real_img import preprocess_for_cnn, classes, find_cat_with_haar

# --- 全局常量 ---
FILENAME = "cats.keras"
model = None
task_queue = queue.Queue() # <-- 创建一个线程安全的任务队列

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully with result code", rc)
        client.subscribe("Group2/IMAGE/classify_request")
    else:
        print("Connection failed with result code", rc)

def classify_cat(filename, img_data):
    # ... (此函数不变) ...
    prediction = model.predict(img_data)
    win = np.argmax(prediction)
    score = float(np.max(prediction))
    return {
        "filename": filename,
        "prediction": classes[win],
        "score": score,
        "index": str(win)
    }

def on_message(client, userdata, msg):
    """
    此函数现在是“生产者”，只负责快速接收消息并放入队列。
    """
    print(f"Received message for topic {msg.topic}, adding to queue.")
    try:
        # 直接处理二进制数据
        image_data = msg.payload
        
        # 将二进制数据转换为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)
        
        # 使用OpenCV解码图像
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode image")
            return
            
        # 转换BGR到RGB（与训练时一致）
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 应用letterboxing预处理（假设target_size是(224, 224)）
        frame = letterbox_preprocess(frame, (224, 224))
            
        # 生成文件名（或从topic中提取）
        filename = f"received_image_{int(time.time())}.jpg"
        
        # 将耗时任务所需的数据放入队列
        task_queue.put((filename, frame))
        
    except Exception as e:
        print(f"Error parsing message or adding to queue: {e}")

def worker(client):
    """
    此函数是“消费者”，在一个独立的线程中运行，负责处理所有耗时任务。
    """
    print("Worker thread started.")
    while True:
        try:
            # get() 方法会阻塞，直到队列中有任务为止
            filename, frame = task_queue.get()
            print(f"Worker processing task for: {filename}")

            # --- 使用Haar检测猫脸的新逻辑 ---
            print(f"Starting Haar cascade cat detection for {filename}")
            
            # 1) 首先尝试使用Haar检测猫脸
            try:
                cat_faces = find_cat_with_haar(frame)
                
                # 更严格的检查：确保返回值有效且第一个元素不是None
                if (cat_faces is not None and 
                    len(cat_faces) > 0 and 
                    cat_faces[0] is not None and 
                    len(cat_faces[0]) == 4):  # 确保是四元组
                    
                    # 检测到有效的猫脸，使用第一个检测到的猫脸区域
                    print(f"Haar detection successful for {filename}, found {len(cat_faces)} cat face(s)")
                    x, y, w, h = cat_faces[0]  # 使用第一个检测到的猫脸
                    
                    # 验证坐标值的有效性
                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        cat_face_roi = frame[y:y+h, x:x+w]  # 提取猫脸区域
                        classification_image = cat_face_roi
                        print(f"Using detected cat face region: x={x}, y={y}, w={w}, h={h}")
                    else:
                        print(f"Invalid coordinates detected for {filename}, using original image")
                        classification_image = frame
                else:
                    # 没有检测到有效的猫脸，使用原图
                    print(f"No valid cat face detected with Haar cascade for {filename}, using original image")
                    classification_image = frame
                    
            except Exception as e:
                # 如果Haar检测出错，也使用原图
                print(f"Haar detection error for {filename}: {e}")
                print(f"Using original image for {filename}")
                classification_image = frame

            # 1) 尺寸检查 (如果需要)
            # ...

            # 2) 检测 A4
            # a4 = detect_a4_and_extract(frame)
            # if a4 is None:
            #     # 没检测到A4，使用原始图像
            #     print(f"No A4 detected for {filename}, using original image")
            #     classification_image = frame
            #     bbox = None
            # else:
            #     # 3) 检测猫ROI
            #     bbox, cat_roi = find_cat_photo(a4)
            #     if bbox is None or cat_roi is None:
            #         # 没检测到猫ROI，使用A4图像
            #         print(f"No cat ROI detected for {filename}, using A4 image")
            #         classification_image = a4
            #         bbox = None
            #     else:
            #         # 检测到猫ROI，使用ROI
            #         print(f"Cat ROI detected for {filename}")
            #         classification_image = cat_roi

            # 4) 预处理和分类（总是执行）
            prep = preprocess_for_cnn(classification_image)
            classification = classify_cat(filename, prep)

            # 5) 发布结果
            client.publish("Group2/IMAGE/predict_result", json.dumps(classification))
            print("Published classification result:", classification)

            # 6) 在终端打印结果
            print(f"✅ Classification Result for {filename}:")
            if "error" in classification:
                print(f"   Error: {classification['error']}")
            else:
                # 根据你的实际数据结构调整
                for key, value in classification.items():
                    if key != "filename":  # 避免重复打印filename
                        print(f"   {key}: {value}")
            print("-" * 50)

        except Exception as e:
            print(f"An error occurred in worker thread: {e}")
            # 可选：发布错误结果
            try:
                error_result = {
                    "filename": filename if 'filename' in locals() else "unknown",
                    "error": str(e)
                }
                client.publish("Group2/IMAGE/predict_result", json.dumps(error_result))
            except:
                pass
        finally:
            # 标记任务完成
            try:
                task_queue.task_done()
            except:
                pass


def setup(hostname):
    client = mqtt.Client()
    client.username_pw_set("csy", "1")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

def main():
    global model
    print("Loading model from file:", FILENAME)
    model = tf_keras.models.load_model(FILENAME)             #模型是用tf_keras训练的也必须用tf_keras加载
    print("Model loaded successfully")

    client = setup("127.0.0.1")

    # --- 创建并启动 worker 线程 ---
    worker_thread = threading.Thread(target=worker, args=(client,), daemon=True)
    worker_thread.start()

    print("Waiting for incoming messages...")
    try:
        # 主线程可以保持运行或执行其他任务
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down.")
        # client.loop_stop() # 如果需要，可以优雅地停止循环
        exit(0)

if __name__ == "__main__":
    main()