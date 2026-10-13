import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

class ParkingDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        # Load pretrained model YOLOv8
        self.model = YOLO(model_path)
        
        # Kelas kendaraan pada dataset COCO: car(2), motorcycle(3), bus(5), truck(7)
        self.target_classes = [2, 3, 5, 7]
        
        # Contoh koordinat slot parkir (Polygon 4 titik)
        # Nanti koordinat ini bisa dibaca dari DB atau config .json
        self.parking_slots = {
            "A1": np.array([[100, 100], [300, 100], [300, 300], [100, 300]]),
            "A2": np.array([[320, 100], [520, 100], [520, 300], [320, 300]])
        }

    def process_frame(self, image: np.ndarray):
        # Run inference YOLO
        results = self.model(image, verbose=False)[0]
        
        # Convert ke objek Detections milik Supervision
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter hanya kendaraan
        detections = detections[np.isin(detections.class_id, self.target_classes)]

        slots_status = []
        occupied_count = 0

        # Cek ketersediaan untuk setiap slot
        for slot_id, polygon in self.parking_slots.items():
            zone = sv.PolygonZone(polygon=polygon)
            is_occupied = bool(zone.trigger(detections=detections).any())

            if is_occupied:
                occupied_count += 1

            slots_status.append({
                "slot_id": slot_id,
                "is_occupied": is_occupied
            })

        return {
            "total_slots": len(self.parking_slots),
            "occupied_slots": occupied_count,
            "empty_slots": len(self.parking_slots) - occupied_count,
            "slots_detail": slots_status
        }
