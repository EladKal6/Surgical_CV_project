#!/usr/bin/env python3
"""
video.py — YOLOv8 segmentation on a video file.
Writes an annotated video with masks overlayed (via result.plot()).

Usage:
  python video.py --model runs/segment/FINAL_FINAL_FINAL3/weights/best.pt --source path/to/video.mp4

Optional:
  --conf 0.25 --iou 0.7 --imgsz 784 --device 0 --save_dir runs/predict_seg --out_video out.mp4 --show
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from time import perf_counter

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Ultralytics not installed. Try: pip install ultralytics==8.*") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLOv8-seg model weights (e.g., best.pt)")
    ap.add_argument("--source", required=True, help="Path to a video file")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    ap.add_argument("--imgsz", type=int, default=784, help="Inference image size")
    ap.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    ap.add_argument("--save_dir", default="runs/predict_seg", help="Output directory")
    ap.add_argument("--out_video", default="", help="Output video path (default: <save_dir>/<stem>_seg.mp4)")
    ap.add_argument("--show", action="store_true", help="Show a live preview window")
    args = ap.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        raise FileNotFoundError(f"Video not found: {src_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.out_video:
        out_path = Path(args.out_video)
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = save_dir / f"{src_path.stem}_seg.mp4"

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc: use mp4v for broad compatibility. If issues, try 'avc1' or 'XVID'.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    model = YOLO(args.model)

    frame_idx = 0
    t0 = perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Run segmentation for this frame
            results = model.predict(
                source=frame,
                task='segment',
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
                stream=False,
                retina_masks=False,  # toggle if desired
            )

            # results is a list with 1 item for single frame
            result = results[0]
            annotated = result.plot()  # BGR frame with overlays

            writer.write(annotated)
            frame_idx += 1

            if args.show:
                cv2.imshow("segmentation (q=quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    dt = perf_counter() - t0
    print(f"Wrote: {out_path}  ({frame_idx} frames, {fps:.1f} FPS input, {frame_idx/max(dt,1e-6):.1f} FPS processed)")


if __name__ == "__main__":
    main()
