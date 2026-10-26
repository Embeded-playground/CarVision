import argparse
import os
import sys
from pathlib import Path

import cv2

from app.detector import ParkingDetector
from app.sources import choose_source, resolve_source

WINDOW_NAME = "Parking Detection Test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Uji deteksi slot parkir pada video/webcam")
    parser.add_argument("--source", help="Path video / 0 untuk webcam")
    parser.add_argument("--model", help="Path model .pt (default: models/yolo26m.pt)")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Skala resize tampilan (mis. 0.75)")
    parser.add_argument("--no-loop", action="store_true", help="Jangan mengulang video saat habis")
    args = parser.parse_args()

    if args.source is None:
        source: str | int = choose_source()
    else:
        source = resolve_source(args.source)

    model_path = args.model or os.getenv("MODEL_PATH") or None
    detector = ParkingDetector(model_path=model_path)
    print(f"[INFO] model={detector.model_path_loaded} device={detector.device}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber: {source}")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("=" * 40)
    print("Tekan Q untuk keluar")
    print("=" * 40)

    is_file = isinstance(source, str) and Path(source).is_file()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_file and not args.no_loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        result = detector.process_frame(frame)
        annotated = detector.annotate(frame, result)

        display = annotated
        if args.display_scale != 1.0:
            display = cv2.resize(annotated, None, fx=args.display_scale, fy=args.display_scale)

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
