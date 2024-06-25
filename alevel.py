#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.signal import butter, sosfilt
from numpy import log10, sqrt
import queue
import threading

# Define vehicle categories
LABEL_MAP = {
    3: "car",
    8: "truck",
    6: "bus"
}

# Audio and Frame data queue
audio_queue = queue.Queue()
frame_queue = queue.Queue()
stop_flag = False

# A-weighting filter design
def A_weighting(fs):
    """Design an A-weighting filter."""
    return butter(4, [20/(fs/2), 20000/(fs/2)], btype='band', output='sos')  # No change needed

# Audio callback function
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Process audio data to compute A-weighted dB level
def process_audio(fs):
    sos = A_weighting(fs)
    global current_dB
    current_dB = "Calculating dB..."
    while not stop_flag or not audio_queue.empty():
        if not audio_queue.empty():
            data = audio_queue.get()
            filtered_data = sosfilt(sos, data[:, 0])  # Apply A-weighting filter
            rms = sqrt(np.mean(filtered_data**2))
            reference_value = 20e-6  # Reference sound pressure (20 µPa)
            if rms > 0:
                db = 20 * log10(rms / reference_value)
                current_dB = f"{db:.2f} dB"
            else:
                current_dB = "Silence or very low sound level"

# Load model globally to avoid re-loading per frame
model = tf.saved_model.load('/home/pi/Mycode/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')

# Process and display frames
def display_frames():
    global stop_flag
    while not stop_flag or not frame_queue.empty():
        if not frame_queue.empty():
            frame, detections = frame_queue.get()
            process_frame(frame, detections)

def process_frame(frame, detections):
    num_detections = int(detections['num_detections'])
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy()

    for i in range(num_detections):
        class_id = int(detection_classes[i])
        if class_id in LABEL_MAP and detection_scores[i] >= 0.5:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{LABEL_MAP[class_id]}: {detection_scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display current dB level
    cv2.putText(frame, current_dB, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Detections', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_flag = True

if __name__ == '__main__':
    fs = 44100  # Sampling rate
    audio_thread = threading.Thread(target=process_audio, args=(fs,))
    frame_display_thread = threading.Thread(target=display_frames)
    audio_thread.start()
    frame_display_thread.start()

    stream = sd.InputStream(callback=audio_callback, samplerate=fs, channels=1)
    stream.start()

    cap = cv2.VideoCapture(0)
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_np = np.expand_dims(frame_rgb, axis=0)
        input_tensor = tf.convert_to_tensor(frame_np, dtype=tf.uint8)
        detections = model(input_tensor)
        frame_queue.put((frame, detections))

    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()
    stop_flag = True
    audio_thread.join()
    frame_display_thread.join()
