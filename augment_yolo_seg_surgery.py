
import argparse
from pathlib import Path
import random
import os

import cv2
import numpy as np


def parse_yolo_seg_line(line: str):
    parts = line.strip().split()
    cls_id = int(float(parts[0]))
    xc, yc, w, h = map(float, parts[1:5])
    coords = list(map(float, parts[5:])) if len(parts) > 5 else []
    poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)] if coords else None
    return cls_id, (xc, yc, w, h), poly


def load_yolo_label(txt_path: Path):
    objs = []
    if not txt_path.exists():
        return objs
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines:
        cls_id, bbox, poly_norm = parse_yolo_seg_line(ln)
        objs.append({"cls": cls_id, "bbox": bbox, "poly_norm": poly_norm})
    return objs


def polygon_area(poly_xy_abs: np.ndarray) -> float:
    if poly_xy_abs is None or len(poly_xy_abs) < 3:
        return 0.0
    x = poly_xy_abs[:, 0]
    y = poly_xy_abs[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clamp_polygon(poly_xy_abs: np.ndarray, W: int, H: int) -> np.ndarray:
    poly_xy_abs[:, 0] = np.clip(poly_xy_abs[:, 0], 0, W - 1)
    poly_xy_abs[:, 1] = np.clip(poly_xy_abs[:, 1], 0, H - 1)
    return poly_xy_abs


def yolo_bbox_from_polygon(poly_xy_abs: np.ndarray, W: int, H: int):
    if poly_xy_abs is None or poly_xy_abs.size == 0:
        return None
    x_min = max(0.0, float(np.min(poly_xy_abs[:, 0])))
    y_min = max(0.0, float(np.min(poly_xy_abs[:, 1])))
    x_max = min(W - 1.0, float(np.max(poly_xy_abs[:, 0])))
    y_max = min(H - 1.0, float(np.max(poly_xy_abs[:, 1])))
    bw = max(0.0, x_max - x_min)
    bh = max(0.0, y_max - y_min)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return (cx / W, cy / H, bw / W, bh / H)


def rect_poly_from_yolo_bbox(bbox, W: int, H: int) -> np.ndarray:
    xc, yc, w, h = bbox
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def save_yolo_label(objs_aug, out_txt: Path, W: int, H: int):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for obj in objs_aug:
        cls_id = obj["cls"]
        poly_abs = obj["poly_abs"].astype(np.float32)
        # normalize polygon
        poly_norm = (poly_abs / np.array([[W, H]], dtype=np.float32)).reshape(-1)
        bbox_yolo = yolo_bbox_from_polygon(poly_abs, W, H)
        if bbox_yolo is None:
            continue
        xc, yc, w, h = bbox_yolo
        nums = [cls_id, xc, yc, w, h] + poly_norm.tolist()
        line = " ".join(f"{v:.6f}" if i > 0 else str(int(v)) for i, v in enumerate(nums))
        lines.append(line)
    if lines:
        with open(out_txt, "w") as f:
            f.write("\n".join(lines) + "\n")

# ----------------------------
# Geometric transforms (OpenCV)
# ----------------------------

def affine_matrix(center, scale, angle_deg, tx, ty):
    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M  # 2x3


def apply_affine_to_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])  # Nx3
    return (pts_h @ M.T).astype(np.float32)


def perspective_matrix(W, H, rng, strength=0.07):
    jitter = strength * min(W, H)
    src = np.float32([
        [0 + rng.uniform(-jitter, jitter), 0 + rng.uniform(-jitter, jitter)],
        [W - 1 + rng.uniform(-jitter, jitter), 0 + rng.uniform(-jitter, jitter)],
        [W - 1 + rng.uniform(-jitter, jitter), H - 1 + rng.uniform(-jitter, jitter)],
        [0 + rng.uniform(-jitter, jitter), H - 1 + rng.uniform(-jitter, jitter)],
    ])
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    return cv2.getPerspectiveTransform(src, dst)

# ----------------------------
# Photometric / video effects
# ----------------------------

def motion_blur(img, k=7, angle_deg=0.0):
    k = max(3, int(k) | 1)  # odd
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle_deg, 1.0)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def add_noise(img, sigma=10.0):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_contrast_brightness(img, alpha=1.0, beta=0):
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def shift_hsv(img, dh=0, ds=0, dv=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = ((h.astype(np.int16) + dh) % 180).astype(np.uint8)
    s = np.clip(s.astype(np.int16) + ds, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.int16) + dv, 0, 255).astype(np.uint8)
    hsv2 = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def jpeg_compress(img, quality=60):
    q = int(max(10, min(95, quality)))
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def add_vignette(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    H, W = image.shape[:2]
    Y, X = np.ogrid[:H, :W]
    cx, cy = W / 2, H / 2
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    mask = 1.0 - strength * (r / max_r) ** 2
    mask = np.clip(mask, 0.2, 1.0).astype(np.float32)
    out = (image.astype(np.float32) * mask[..., None]).astype(np.uint8)
    return out


def add_specular_highlights(image: np.ndarray, n: int = 5) -> np.ndarray:
    H, W = image.shape[:2]
    overlay = image.copy()
    for _ in range(n):
        radius = np.random.randint(3, max(4, int(0.02 * min(H, W))))
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        cv2.circle(overlay, (x, y), int(radius), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

# ----------------------------
# Pipeline per image
# ----------------------------

def process_one(img_path: Path, lbl_path: Path, out_img: Path, out_lbl: Path, rng: np.random.Generator,
                min_area_px: float = 50.0):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    objs = load_yolo_label(lbl_path)
    if not objs:
        return False

    # Assemble absolute polygons
    polys_abs = []
    metas = []
    for obj in objs:
        if obj["poly_norm"]:
            pts = np.array(obj["poly_norm"], dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
        else:
            pts = rect_poly_from_yolo_bbox(obj["bbox"], W, H)
        if len(pts) >= 3:
            polys_abs.append(pts)
            metas.append(obj)
    if not polys_abs:
        return False

    # --- Geometric ---
    if rng.random() < 0.75:
        # Single affine for both image and polygons
        angle = float(rng.uniform(-10, 10))
        scale = float(rng.uniform(0.9, 1.1))
        tx = float(rng.uniform(-0.08, 0.08) * W)
        ty = float(rng.uniform(-0.08, 0.08) * H)
        M = affine_matrix((W / 2.0, H / 2.0), scale, angle, tx, ty)
        img_geo = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        polys_geo = [apply_affine_to_points(p, M) for p in polys_abs]
    else:
        Hm = perspective_matrix(W, H, rng, strength=0.07)
        img_geo = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # project points
        polys_geo = []
        for p in polys_abs:
            pts_h = np.hstack([p, np.ones((p.shape[0], 1), dtype=np.float32)])
            warp = (pts_h @ Hm.T)
            w = warp[:, 2:3]
            polys_geo.append((warp[:, :2] / np.clip(w, 1e-6, None)).astype(np.float32))

    # Horizontal flip with 0.5 prob
    if rng.random() < 0.5:
        img_geo = cv2.flip(img_geo, 1)
        Wg = img_geo.shape[1]
        polys_geo = [np.vstack([Wg - 1 - p[:, 0], p[:, 1]]).T.astype(np.float32) for p in polys_geo]

    # Clean polygons (clip & filter)
    valid_polys = []
    valid_meta = []
    for pts, meta in zip(polys_geo, metas):
        if pts is None or len(pts) < 3:
            continue
        pts = clamp_polygon(pts.astype(np.float32), img_geo.shape[1], img_geo.shape[0])
        if polygon_area(pts) >= min_area_px:
            valid_polys.append(pts)
            valid_meta.append(meta)
    if not valid_polys:
        return False

    # --- Photometric / video-like ---
    out_img_np = img_geo
    if rng.random() < 0.6:
        k = int(rng.integers(3, 10))
        ang = float(rng.uniform(-20, 20))
        out_img_np = motion_blur(out_img_np, k=k, angle_deg=ang)
    if rng.random() < 0.5:
        out_img_np = cv2.GaussianBlur(out_img_np, (0, 0), sigmaX=float(rng.uniform(0.0, 1.0)))
    if rng.random() < 0.5:
        out_img_np = add_noise(out_img_np, sigma=float(rng.uniform(0.0, 8.0)))
    if rng.random() < 0.6:
        out_img_np = adjust_contrast_brightness(out_img_np, alpha=float(rng.uniform(0.8, 1.3)), beta=float(rng.uniform(-15, 15)))
    if rng.random() < 0.5:
        out_img_np = shift_hsv(out_img_np, dh=int(rng.integers(-8, 9)), ds=int(rng.integers(-18, 19)), dv=int(rng.integers(-18, 19)))
    if rng.random() < 0.6:
        out_img_np = jpeg_compress(out_img_np, quality=int(rng.integers(45, 85)))
    if rng.random() < 0.5:
        out_img_np = add_vignette(out_img_np, strength=float(rng.uniform(0.3, 0.7)))
    if rng.random() < 0.5:
        out_img_np = add_specular_highlights(out_img_np, n=int(rng.integers(2, 7)))

    # Save
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), out_img_np)

    objs_aug = []
    H_out, W_out = out_img_np.shape[:2]
    for meta, pts in zip(valid_meta, valid_polys):
        objs_aug.append({"cls": meta["cls"], "poly_abs": pts})
    save_yolo_label(objs_aug, out_lbl, W_out, H_out)
    return True

# ----------------------------
# Entry
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str)
    ap.add_argument("--labels_dir", required=True, type=str)
    ap.add_argument("--out_images_dir", required=True, type=str)
    ap.add_argument("--out_labels_dir", required=True, type=str)
    ap.add_argument("--multiplier", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_images_dir = Path(args.out_images_dir)
    out_labels_dir = Path(args.out_labels_dir)

    exts = ("*.png", "*.jpg", "*.jpeg")
    img_paths = []
    for e in exts:
        img_paths.extend(sorted(images_dir.glob(e)))

    total = 0
    for img_path in img_paths:
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue
        for k in range(args.multiplier):
            out_img = out_images_dir / f"{stem}_aug{k:02d}.jpg"
            out_lbl = out_labels_dir / f"{stem}_aug{k:02d}.txt"
            ok = process_one(img_path, lbl_path, out_img, out_lbl, rng, min_area_px=50.0)
            if ok:
                total += 1
    print(f"Done. Wrote {total} augmented samples.")

if __name__ == "__main__":
    main()
