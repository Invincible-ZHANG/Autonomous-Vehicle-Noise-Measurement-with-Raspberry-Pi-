#!/usr/bin/env python3


import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.signal import welch
from numpy import log10, sqrt
import queue
import threading

# 限定为汽车相关类别的标签
LABEL_MAP = {
    3: "car",
    8: "truck",
    6: "bus"
}

# 音频队列
audio_queue = queue.Queue()

# 视频队列
video_queue = queue.Queue()

# 分贝值锁
db_lock = threading.Lock()

def audio_callback(indata, frames, time, status):
    """音频回调函数，处理音频数据并存入队列"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio():
    """从音频队列中取出数据，计算并更新分贝值"""
    global current_dB
    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            rms_value = sqrt(np.mean(data**2))
            reference_value = 1e-6  # 1微帕斯卡为参考声压
            with db_lock:
                current_dB = 20 * log10(rms_value/reference_value)

def display_results():
    """从视频队列中取出帧，显示检测结果和当前分贝值"""
    while True:
        ret, frame = video_queue.get()
        if not ret:
            break

        with db_lock:
            db_to_display = current_dB

        cv2.putText(frame, f'Current dB: {db_to_display:.2f} dB', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Detections', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def load_and_test_model(model_path, cap):
    """加载模型，并进行实时视频帧的车辆检测，将结果放入视频队列"""
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                video_queue.put((False, None))
                break

            original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np = np.expand_dims(original_image, axis=0)
            input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
            detections = model(input_tensor)

            num_detections = int(detections['num_detections'])
            detection_boxes = detections['detection_boxes'][0].numpy()
            detection_scores = detections['detection_scores'][0].numpy()
            detection_classes = detections['detection_classes'][0].numpy()

            for i in range(num_detections):
                class_id = int(detection_classes[i])
                if class_id in LABEL_MAP and detection_scores[i] >= 0.5:
                    ymin, xmin, ymax, xmax = detection_boxes[i]
                    cv2.rectangle(frame, (int(xmin * frame.shape[1]), int(ymin * frame.shape[0])),
                                  (int(xmax * frame.shape[1]), int(ymax * frame.shape[0])), (0, 255, 0), 2)
                    label = f'{LABEL_MAP[class_id]}: {detection_scores[i]:.2f}'
                    cv2.putText(frame, label, (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            video_queue.put((True, frame))
    finally:
        cap.release()

if __name__ == '__main__':
    current_dB = 0.0  # 初始化分贝值
    cap = cv2.VideoCapture(0)

    # 启动声音处理线程
    audio_thread = threading.Thread(target=process_audio)
    audio_thread.start()

    # 启动视频显示线程
    display_thread = threading.Thread(target=display_results)
    display_thread.start()

    # 配置声音捕捉
    with sd.InputStream(callback=audio_callback) as stream:
        # 运行车辆检测模型
        model_path = '/home/pi/Mycode/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
        load_and_test_model(model_path, cap)

    # 结束声音捕捉和处理
    audio_thread.join()
    display_thread.join()

