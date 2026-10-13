from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from app.detector import ParkingDetector

app = FastAPI(
    title="Parking Identification Service",
    version="1.0.0"
)

# Inisialisasi detector saat service mulai
detector = ParkingDetector(model_path="yolov8n.pt")

@app.get("/")
def health_check():
    return {"status": "online", "service": "Parking Vision AI"}

@app.post("/api/v1/detect")
async def detect_parking(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Gambar corrupt atau tidak valid")

    # Jalankan deteksi
    result = detector.process_frame(image)

    return {
        "success": True,
        "data": result
    }
