#!/usr/bin/env python3
"""
predict.py — YOLOv8 segmentation on a single image (or glob).
Saves an annotated image and (optionally) per-instance masks.

Usage:
  python predict.py --model runs/segment/FINAL_FINAL_FINAL3/weights/best.pt --source path/to/image.jpg
Optional:
  --conf 0.25 --iou 0.7 --imgsz 784 --device 0 --save_dir runs/predict_seg --save_masks --show
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Ultralytics not installed. Try: pip install ultralytics==8.*") from e


def save_masks(result, save_dir: Path, stem: str):
    """
    Save each instance mask as a PNG (255=foreground) and a JSON metadata file.
    """
    import json
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    names = result.names if hasattr(result, "names") else {}

    if masks is None or masks.data is None:
        print("No masks to save.")
        return

    m = masks.data  # [N, H, W] torch.Tensor (bool/0-1)
    m = m.detach().cpu().numpy().astype(np.uint8) * 255  # to 0/255

    meta = []
    inst_dir = save_dir / f"{stem}_masks"
    inst_dir.mkdir(parents=True, exist_ok=True)

    for i in range(m.shape[0]):
        mask_png = inst_dir / f"{stem}_mask_{i:03d}.png"
        cv2.imwrite(str(mask_png), m[i])

        cls_id = None
        conf = None
        if boxes is not None and len(boxes) > i:
            b = boxes[i]
            # b.cls and b.conf are tensors of shape (1,)
            cls_id = int(b.cls.item()) if hasattr(b, 'cls') and b.cls is not None else None
            conf = float(b.conf.item()) if hasattr(b, 'conf') and b.conf is not None else None

        meta.append({
            "mask": mask_png.name,
            "class_id": cls_id,
            "class_name": names.get(cls_id, str(cls_id)) if cls_id is not None else None,
            "confidence": conf
        })

    with open(inst_dir / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(meta)} masks to: {inst_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLOv8-seg model weights (e.g., best.pt)")
    ap.add_argument("--source", required=True, help="Path to an image (or glob).")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    ap.add_argument("--imgsz", type=int, default=784, help="Inference image size")
    ap.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    ap.add_argument("--save_dir", default="runs/predict_seg", help="Output directory")
    ap.add_argument("--save_masks", action="store_true", help="Also save each instance mask as PNG")
    ap.add_argument("--show", action="store_true", help="Show annotated image in a window")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Run prediction
    results = model.predict(
        source=args.source,
        task='segment',
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=True,
        stream=False,
        save=False,
        retina_masks=False,  # can be toggled if desired
    )

    # Handle possibly multiple images (if glob passed). Iterate all results.
    for result in results:
        # result.path is the original image path
        in_path = Path(getattr(result, "path", "image"))
        stem = in_path.stem
        annotated = result.plot()  # BGR array with masks/boxes/labels drawn

        out_path = save_dir / f"{stem}_seg.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated image: {out_path}")

        if args.save_masks:
            save_masks(result, save_dir, stem)

        if args.show:
            cv2.imshow("segmentation", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
