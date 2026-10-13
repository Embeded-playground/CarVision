import cv2
import numpy as np
import supervision as sv
from app.detector import ParkingDetector

# 1. Inisialisasi detector & annotator
detector = ParkingDetector(model_path="yolov8n.pt")
box_annotator = sv.BoxAnnotator()

# Buka file video
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

# Setup Window
window_name = "Parking Detection Test"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Ambil data hasil deteksi dari detector.py
    result = detector.process_frame(frame)

    # 3. Gambar ulang polygon slot parkir secara visual di frame ini
    annotated_frame = frame.copy()

    # Ambil deteksi ulang khusus visualisasi
    results_raw = detector.model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results_raw)
    detections = detections[np.isin(detections.class_id, detector.target_classes)]

    for slot_id, polygon in detector.parking_slots.items():
        zone = sv.PolygonZone(polygon=polygon)
        is_occupied = bool(zone.trigger(detections=detections).any())

        # Merah jika terisi, Hijau jika kosong
        color = sv.Color.RED if is_occupied else sv.Color.GREEN

        polygon_annotator = sv.PolygonZoneAnnotator(zone=zone, color=color, thickness=2)
        annotated_frame = polygon_annotator.annotate(scene=annotated_frame)

    # Gambar bounding box mobil
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

    # 4. Tampilkan frame
    cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
