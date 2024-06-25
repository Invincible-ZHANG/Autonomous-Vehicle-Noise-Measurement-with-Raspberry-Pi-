#!/usr/bin/env python3

import cv2

cap = cv2.VideoCapture(0)  # 开启摄像头
while True:
    ret, frame = cap.read()  # 读取一帧
    if not ret:
        break  # 如果无法获取帧，则退出
    cv2.imshow('Video Stream', frame)  # 显示当前帧
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # 按'q'键退出

cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
