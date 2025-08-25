import random, math, shutil
from pathlib import Path
import numpy as np
import cv2
import argparse

DATASET_ROOT = Path("../synthetic_output/")
SRC_IMG_DIR  = DATASET_ROOT / "images"
SRC_LBL_DIR  = DATASET_ROOT / "masks"

DST_IMG_DIR  = DATASET_ROOT / "images_occ"
DST_LBL_DIR  = DATASET_ROOT / "masks_occ"

OCCLUDER_DIR = Path("occluders")

TARGET_OCC_FRAC      = 0.30
CENTER_JITTER_FRAC   = 0.08
TARGET_TOL           = 0.05
MAX_ITERS            = 6
MIN_INSTANCE_AREA_PX = 80
NUM_CLASSES          = 2
RAND_SEED            = 123


MIDDLE_ELLIPSE_FRAC  = 0.20
CENTER_RING_INNER    = 0.12
CENTER_RING_OUTER    = 0.65
MAX_CENTER_TRIES     = 10


def read_image(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"bad image: {p}")
    return im

def _parse_yolo_line_any(line, w, h):
    """
    Parse YOLO-seg line:
      A) class cx cy w h x1 y1 x2 y2 ...
      B) class x1 y1 x2 y2 ...
      C) class cx cy w h  (bbox-only -> rectangle polygon)
    Returns (cid, pts_float32_pixels).
    """
    parts = line.strip().split()
    if not parts:
        return None
    cid = int(float(parts[0]))
    vals = list(map(float, parts[1:]))

    pts = None
    if len(vals) >= 10 and (len(vals) - 4) % 2 == 0:
        poly = np.asarray(vals[4:], dtype=np.float32).reshape(-1, 2)
        pts = np.empty_like(poly)
        pts[:, 0] = np.clip(poly[:, 0] * w, 0, w - 1)
        pts[:, 1] = np.clip(poly[:, 1] * h, 0, h - 1)
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        poly = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        pts = np.empty_like(poly)
        pts[:, 0] = np.clip(poly[:, 0] * w, 0, w - 1)
        pts[:, 1] = np.clip(poly[:, 1] * h, 0, h - 1)
    elif len(vals) >= 4:
        xc, yc, bw, bh = vals[:4]
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    else:
        return None
    return (cid, pts)

def parse_yolo_seg(txt_path, w, h):
    polys = []
    with open(txt_path, "r") as f:
        for line in f:
            out = _parse_yolo_line_any(line, w, h)
            if out is None:
                continue
            cid, pts = out
            polys.append((cid, pts))
    return polys

def load_occluders():
    if not OCCLUDER_DIR.exists():
        return []
    out = []
    for p in sorted(OCCLUDER_DIR.glob("*.png")):
        rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
            out.append(rgba)
    return out

def bbox_from_mask(m):
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def rasterize_instances(polys, shape):
    """Return list of per-instance binary masks (no class merge)."""
    H, W = shape[:2]
    insts = []
    for cid, pts in polys:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
        if cv2.countNonZero(m) >= MIN_INSTANCE_AREA_PX:
            insts.append({"cid": cid, "mask": m})
    return insts

def place_png(img, occ_rgba, center, scale, angle_deg):
    """Composite RGBA onto BGR at 'center'; return new_img, occ_mask."""
    H, W = img.shape[:2]
    oh, ow = occ_rgba.shape[:2]
    nw, nh = max(8, int(ow * scale)), max(8, int(oh * scale))
    occ = cv2.resize(occ_rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle_deg, 1.0)
    occ = cv2.warpAffine(
        occ, M, (nw, nh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    cx, cy = int(center[0]), int(center[1])
    x0 = np.clip(cx - nw // 2, 0, W - 1); y0 = np.clip(cy - nh // 2, 0, H - 1)
    x1 = np.clip(x0 + nw, 0, W);         y1 = np.clip(y0 + nh, 0, H)

    occ = occ[0:(y1 - y0), 0:(x1 - x0)]
    if occ.size == 0:
        return img, np.zeros((H, W), np.uint8)

    bgr = occ[:, :, :3].astype(np.float32)
    a   = (occ[:, :, 3].astype(np.float32) / 255.0)

    if np.random.rand() < 0.5:
        bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=np.random.uniform(0.5, 1.5))
        a   = cv2.GaussianBlur(a,   (0, 0), sigmaX=np.random.uniform(0.5, 1.5))

    out = img.copy().astype(np.float32)
    roi = out[y0:y1, x0:x1, :]
    for c in range(3):
        roi[..., c] = (1 - a) * roi[..., c] + a * bgr[..., c]
    out[y0:y1, x0:x1, :] = roi

    occ_mask = np.zeros((H, W), np.uint8)
    occ_mask[y0:y1, x0:x1] = (a > 0.02).astype(np.uint8) * 255
    return out.astype(np.uint8), occ_mask

def sample_center_away(bcx, bcy, bw, bh, W, H):
    R = max(bw, bh)
    for _ in range(MAX_CENTER_TRIES):
        ang = np.random.uniform(0, 2*np.pi)
        r   = np.random.uniform(CENTER_RING_INNER*R, CENTER_RING_OUTER*R)
        cx  = np.clip(int(bcx + r*np.cos(ang)), 0, W-1)
        cy  = np.clip(int(bcy + r*np.sin(ang)), 0, H-1)
        if 0 <= cx < W and 0 <= cy < H:
            return float(cx), float(cy)
    return float(bcx + 0.5*bw), float(bcy)

def components_to_polys_and_bboxes_norm(mask, W, H, max_pts=120, min_area=MIN_INSTANCE_AREA_PX):
    out = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            continue
        # polygon
        peri = cv2.arcLength(cnt, True)
        eps  = 0.01 * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) > max_pts:
            step = int(np.ceil(len(approx) / max_pts))
            approx = approx[::step]
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0] / float(W), 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / float(H), 0, 1)

        # bbox
        x, y, bw, bh = cv2.boundingRect(cnt)
        xc = (x + bw / 2.0) / float(W)
        yc = (y + bh / 2.0) / float(H)
        bw_n = bw / float(W)
        bh_n = bh / float(H)

        out.append(((xc, yc, bw_n, bh_n), pts))
    return out

def targeted_occlusion(img, insts, occluders, target_frac=TARGET_OCC_FRAC):
    """
    Occlude exactly one instance ~target_frac; returns (out_img, occ_mask).
    Ensures occluder avoids the target's 'middle' (central ellipse).
    """
    H, W = img.shape[:2]
    if not insts:
        return img, np.zeros((H, W), np.uint8)

    inst = random.choice(insts)
    inst_mask = inst["mask"]
    area_inst = float(cv2.countNonZero(inst_mask))
    if area_inst < MIN_INSTANCE_AREA_PX:
        return img, np.zeros((H, W), np.uint8)

    rgba = random.choice(occluders) if occluders else None

    bb = bbox_from_mask(inst_mask)
    if bb is None:
        return img, np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = bb
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    bcx, bcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # central no-occlude ellipse
    cz = np.zeros((H, W), np.uint8)
    ax = max(6, int(MIDDLE_ELLIPSE_FRAC * bw))
    ay = max(6, int(MIDDLE_ELLIPSE_FRAC * bh))
    cv2.ellipse(cz, (int(bcx), int(bcy)), (ax, ay), 0, 0, 360, 255, -1)

    angle = np.random.uniform(0, 180)

    if rgba is not None:
        base_area = float((rgba[:, :, 3] > 0).sum())
        base_area = max(base_area, 1.0)
        target_area = target_frac * area_inst
        s = float(np.clip(math.sqrt(target_area / base_area), 0.4, 3.0))

        best = {"diff": 1e9, "img": img, "mask": np.zeros((H, W), np.uint8)}
        for _ in range(MAX_CENTER_TRIES):
            cx, cy = sample_center_away(bcx, bcy, bw, bh, W, H)
            s_try = s
            for _ in range(MAX_ITERS):
                cand_img, cand_occ = place_png(img, rgba, (cx, cy), s_try, angle)
                inter_inst = cv2.countNonZero(cv2.bitwise_and(cand_occ, inst_mask))
                cov = inter_inst / (area_inst + 1e-6)

                if cv2.countNonZero(cv2.bitwise_and(cand_occ, cz)) > 0:
                    dx, dy = cx - bcx, cy - bcy
                    if dx == 0 and dy == 0:
                        dx = 1.0
                    cx = np.clip(int(bcx + 1.2 * dx), 0, W-1)
                    cy = np.clip(int(bcy + 1.2 * dy), 0, H-1)
                    continue

                if abs(cov - target_frac) <= TARGET_TOL:
                    return cand_img, cand_occ

                s_try *= math.sqrt(target_frac / max(cov, 1e-6))
                s_try = float(np.clip(s_try, 0.3, 4.0))

                d = abs(cov - target_frac)
                if d < best["diff"]:
                    best.update({"diff": d, "img": cand_img, "mask": cand_occ})

        return best["img"], best["mask"]

    # Fallback synthetic ellipse occluder
    best = {"diff": 1e9, "img": img, "mask": np.zeros((H, W), np.uint8)}
    for _ in range(MAX_CENTER_TRIES):
        cx, cy = sample_center_away(bcx, bcy, bw, bh, W, H)
        rx0 = max(10, int(0.6 * bw)); ry0 = max(10, int(0.6 * bh))
        s_try = 1.0
        for _ in range(MAX_ITERS):
            rx = max(8, int(rx0 * s_try)); ry = max(8, int(ry0 * s_try))
            cand = np.zeros((H, W), np.uint8)
            cv2.ellipse(cand, (int(cx), int(cy)), (rx, ry), angle, 0, 360, 255, -1)
            cand = cv2.GaussianBlur(cand, (0, 0), sigmaX=2)

            if cv2.countNonZero(cv2.bitwise_and(cand, cz)) > 0:
                dx, dy = cx - bcx, cy - bcy
                if dx == 0 and dy == 0:
                    dx = 1.0
                cx = np.clip(int(bcx + 1.2 * dx), 0, W-1)
                cy = np.clip(int(bcy + 1.2 * dy), 0, H-1)
                continue

            inter_inst = cv2.countNonZero(cv2.bitwise_and(cand, inst_mask))
            cov = inter_inst / (area_inst + 1e-6)

            color = (np.random.randint(90, 140), np.random.randint(170, 220), np.random.randint(180, 230))
            patch = np.zeros_like(img); patch[:] = color
            alpha = (cand / 255.0)[..., None]
            out_img = (img * (1 - alpha) + patch * alpha).astype(np.uint8)

            if abs(cov - target_frac) <= TARGET_TOL:
                return out_img, cand

            s_try *= math.sqrt(target_frac / max(cov, 1e-6))
            s_try = float(np.clip(s_try, 0.3, 4.0))

            d = abs(cov - target_frac)
            if d < best["diff"]:
                best.update({"diff": d, "img": out_img, "mask": cand})

    return best["img"], best["mask"]

def recompute_visible_bboxes_and_polys(insts, occ_mask, W, H):
    out = []
    for inst in insts:
        m = inst["mask"].copy()
        if occ_mask is not None:
            m[occ_mask > 0] = 0
        if cv2.countNonZero(m) < MIN_INSTANCE_AREA_PX:
            continue
        for bbox_n, poly_n in components_to_polys_and_bboxes_norm(m, W, H):
            out.append((inst["cid"], bbox_n, poly_n))
    return out

def save_yolo_seg_bbox_plus_poly(path, items):
    # items: list of (cid, (xc, yc, w, h), pts_norm)
    with open(path, "w") as f:
        for cid, (xc, yc, bw, bh), pts in items:
            bbox_str = f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts.tolist()])
            f.write(f"{bbox_str} {poly_str}\n")

def main():
    parser = argparse.ArgumentParser(description="Add occlusions to dataset images.")
    parser.add_argument("--dataset-root", type=str, default="../synthetic_output/", help="Path to dataset root directory")
    args = parser.parse_args()

    DATASET_ROOT = Path(args.dataset_root)
    SRC_IMG_DIR  = DATASET_ROOT / "images"
    SRC_LBL_DIR  = DATASET_ROOT / "masks"
    DST_IMG_DIR  = DATASET_ROOT / "images_occ"
    DST_LBL_DIR  = DATASET_ROOT / "masks_occ"
    
    if RAND_SEED is not None:
        random.seed(RAND_SEED); np.random.seed(RAND_SEED)

    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DST_LBL_DIR.mkdir(parents=True, exist_ok=True)

    occluders = load_occluders()
    print(f"Loaded {len(occluders)} RGBA occluder cutouts from '{OCCLUDER_DIR}/'.")

    img_paths = sorted([p for p in SRC_IMG_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    print(f"Found {len(img_paths)} training images.")

    # choose exactly 50% to occlude
    n = len(img_paths)
    k = n // 2
    occ_indices = set(random.sample(range(n), k))

    num_copied = 0
    num_occluded = 0

    for idx, img_path in enumerate(img_paths):
        stem = img_path.stem
        ext  = img_path.suffix
        lbl_path = SRC_LBL_DIR / f"{stem}.txt"

        out_img_path = DST_IMG_DIR / f"{stem}{ext}"
        out_lbl_path = DST_LBL_DIR / f"{stem}.txt"

        if not lbl_path.exists():
            shutil.copy2(img_path, out_img_path)
            num_copied += 1
            continue

        img = read_image(img_path)
        H, W = img.shape[:2]
        polys = parse_yolo_seg(lbl_path, W, H)

        do_occ = idx in occ_indices and len(polys) > 0
        if do_occ:
            insts = rasterize_instances(polys, img.shape)
            if insts:
                img, occ_mask = targeted_occlusion(img, insts, occluders, target_frac=TARGET_OCC_FRAC)
                new_items = recompute_visible_bboxes_and_polys(insts, occ_mask, W, H)  # bbox+poly
                cv2.imwrite(str(out_img_path), img)
                save_yolo_seg_bbox_plus_poly(out_lbl_path, new_items)
                num_occluded += 1
            else:
                shutil.copy2(img_path, out_img_path)
                shutil.copy2(lbl_path, out_lbl_path)
                num_copied += 1
        else:
            shutil.copy2(img_path, out_img_path)
            shutil.copy2(lbl_path, out_lbl_path)
            num_copied += 1

    print(f"Finished. {num_occluded} images occluded, {num_copied} images copied unchanged.")
    print(f"Output images -> {DST_IMG_DIR}")
    print(f"Labels (bbox+poly, updated) -> {DST_LBL_DIR}")

if __name__ == "__main__":
    main()
