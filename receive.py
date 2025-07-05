# receive.py (更健壮的版本)

import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2
import queue # <-- 引入队列模块
import threading # <-- 引入线程模块

# (请确保 find_real_img.py 在同一目录下)
from find_real_img import detect_a4_and_extract, find_cat_photo, preprocess_for_cnn, classes

# --- 全局常量 ---
FILENAME = "cats.keras"
model = None
task_queue = queue.Queue() # <-- 创建一个线程安全的任务队列

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully with result code", rc)
        client.subscribe("Group2/IMAGE/classify")
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
        recv = json.loads(msg.payload)
        frame = np.array(recv['data'], dtype=np.uint8)
        filename = recv['filename']
        
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

            # --- 所有耗时操作都在这里执行 ---
            # 1) 尺寸检查 (如果需要)
            # ...

            # 2) 检测 A4
            a4 = detect_a4_and_extract(frame)
            if a4 is None:
                client.publish("Group2/IMAGE/predict", json.dumps({"filename": filename, "error": "No A4 detected"}))
                continue # 继续处理下一个任务

            # 3) 检测猫ROI
            bbox, cat_roi = find_cat_photo(a4)
            if bbox is None:
                client.publish("Group2/IMAGE/predict", json.dumps({"filename": filename, "error": "No valid cat photo found"}))
                continue

            # 4) 预处理和分类
            prep = preprocess_for_cnn(cat_roi)
            classification = classify_cat(filename, prep)

            # 5) 发布结果
            client.publish("Group2/IMAGE/predict", json.dumps(classification))
            print("Published classification result:", classification)

        except Exception as e:
            print(f"An error occurred in worker thread: {e}")


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
    model = load_model(FILENAME)
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