#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import onnxruntime as ort
import serial
from datetime import datetime
from picamera2 import Picamera2

from utils import crop_card, parse_mtg_filename


class MTGRecognizer:
    """Encapsulates the ONNX model and database."""

    def __init__(self, model_path: str, db_path: str, img_size: int = 512):
        self.img_size = img_size

        print("[INFO] Loading Database...")
        db = np.load(db_path)
        self.db_vectors = db["embeddings"].astype(np.float32)
        self.db_names = db["names"]

        print("[INFO] Initializing ONNX Runtime...")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        print("[INFO] Model and Database loaded successfully.")

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        img_float = img_padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std

        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        return np.expand_dims(img_transposed, axis=0)

    def extract_normalized_vector(self, tensor: np.ndarray) -> np.ndarray:
        vec = self.ort_session.run(None, {"input": tensor})[0][0]
        return vec / np.linalg.norm(vec)

    def predict(self, frame: np.ndarray) -> str:
        cropped_img_0, _ = crop_card(image_path=frame)
        if cropped_img_0 is None:
            return "ERROR: No card detected in image."

        cropped_img_180 = cv2.rotate(cropped_img_0, cv2.ROTATE_180)
        tensor_180 = self.preprocess_image(cropped_img_180)
        vec_180 = self.extract_normalized_vector(tensor_180)

        sims_180 = np.dot(self.db_vectors, vec_180)
        best_idx = np.argmax(sims_180)

        top1_name, top1_edition = parse_mtg_filename(self.db_names[best_idx])
        confidence = sims_180[best_idx] * 100

        return f"{top1_name} [{top1_edition.upper()}] (Conf: {confidence:.2f}%)"


def main():
    SERIAL_PORT = "/dev/ttyACM0"
    BAUD_RATE = 57600
    LOG_FILE = "mtg_results.txt"
    MODEL_PATH = "mtg_recon_edge.onnx"
    DB_PATH = "mtg_database.npz"

    recognizer = MTGRecognizer(model_path=MODEL_PATH, db_path=DB_PATH)

    print(f"[INFO] Connecting to Arduino on {SERIAL_PORT}...")
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=None)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open serial port: {e}")
        return

    print("[INFO] Initializing Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("\n[INFO] System Ready. Priming the first card...")
    card_counter = 0

    arduino.write(b"A")
    time.sleep(1)
    arduino.write(b"B")

    ready = False
    while not ready:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode("utf-8").strip()
            if line == "R":
                ready = True

    print("[INFO] First card ready in position B. Starting main loop...")

    try:
        while True:
            frame = picam2.capture_array()

            arduino.write(b"C")
            time.sleep(0.6)

            arduino.write(b"A")

            print("[INFO] Running the inference...")
            result = recognizer.predict(frame)
            card_counter += 1

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} | Card #{card_counter:04d} | Result: {result}\n"
            print(log_entry.strip())
            with open(LOG_FILE, "a") as f:
                f.write(log_entry)

            arduino.write(b"B")

            ready = False
            while not ready:
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode("utf-8").strip()
                    if line == "R":
                        ready = True

    except KeyboardInterrupt:
        print("\n[INFO] Pipeline stopped by user.")


if __name__ == "__main__":
    main()
