#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import cv2

# 限定为汽车相关类别的标签
LABEL_MAP = {
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    7: "train",
    8: "truck"
}

def load_and_test_model(model_path):
    # 加载TensorFlow模型
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully.")

    # 打开USB摄像头
    cap = cv2.VideoCapture(0)  # 0 通常是默认摄像头
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    try:
        while True:
            # 捕获摄像头的下一帧
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video device.")
                break

            # 转换颜色空间到RGB
            original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 准备图像数据
            image = np.array(original_image, dtype=np.uint8)
            image_np = np.expand_dims(image, axis=0)

            # 转换图像数据类型以匹配模型要求
            input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)

            # 执行推理
            detections = model(input_tensor)

            # 解析检测结果
            num_detections = int(detections['num_detections'])
            detection_boxes = detections['detection_boxes'][0].numpy()
            detection_scores = detections['detection_scores'][0].numpy()
            detection_classes = detections['detection_classes'][0].numpy()

            # 绘制检测框和标签
            for i in range(num_detections):
                class_id = int(detection_classes[i])
                if class_id in LABEL_MAP and detection_scores[i] >= 0.5:
                    ymin, xmin, ymax, xmax = detection_boxes[i]
                    (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                                  ymin * frame.shape[0], ymax * frame.shape[0])
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    label = f'{LABEL_MAP[class_id]}: {detection_scores[i]:.2f}'
                    cv2.putText(frame, label, (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 显示结果
            cv2.imshow('Detections', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = '/home/pi/Mycode/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
    load_and_test_model(model_path)
