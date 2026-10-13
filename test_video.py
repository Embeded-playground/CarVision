import cv2
from app.detector import ParkingDetector

# Inisialisasi detector
detector = ParkingDetector(model_path="yolov8n.pt")

# Buka file video (ganti dengan path videomu)
cap = cv2.VideoCapture("video1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    result = detector.process_frame(frame)
    print(result) # Cek data JSON di terminal

    # Tampilkan frame di window
    cv2.imshow("Parking Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
