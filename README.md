# YOLOv8 Segmentation — Synthetic Dataset Pipeline & Inference

## TL;DR (Quickstart)

```bash
# 0) Create env (recommended)
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 1) Install requirements
pip install -r requirements.txt
# Then install the *right* PyTorch for your CUDA from https://pytorch.org/get-started/locally/

# 2) Build dataset (interactive)
./build_yolo_dataset.sh    

# 3) Inference on image(s)
python predict.py --model model_fine_tuned/weights/best.pt \
                  --source path/to/image.jpg --save_masks --show

# 4) Inference on video
python video.py --model model_fine_tuned/weights/best.pt \
                --source path/to/video.mp4 --show
```

---

## Contents

```
build_yolo_dataset.sh          # Linux/macOS interactive pipeline

synthetic_data_generator.py    # Blender-based synthetic image + mask renderer
add_occlusions.py              # Applies RGBA occluder(s), rewrites YOLO-seg labels to visible region
augment_yolo_seg_surgery.py    # Geometric/photometric augmentation; rewrites YOLO-seg labels

predict.py                     # YOLOv8-seg inference on image(s)
video.py                       # YOLOv8-seg inference on videos
requirements.txt               # Minimal runtime dependencies

README.md
```

---

## Prerequisites

Install everything basic via:
```bash
pip install -r requirements.txt
```
Then install the correct **torch/torchvision** build for your system.

---

## Pipeline Overview

### Phase 1 — Synthetic Generation (Blender)
- Runs `synthetic_data_generator.py` inside Blender:
  ```bash
  blender --background --python synthetic_data_generator.py -- <START> <END>
  ```
- Produces:
  - RGB renders (e.g., `.../images1/*.png`)
  - Binary masks (e.g., `.../masks/*.png`)
  - (Optionally) a JSON listing for bookkeeping
- **Note:** This script itself does **not** produce YOLO‑seg TXT labels. If you need an automated mask→YOLO conversion step (“Phase 1.5”), see *Extending* below.

### Phase 2 — Occlusions
- `add_occlusions.py` reads **existing YOLO‑seg labels** for the source images, places occluders (e.g., gloves PNGs with alpha) while avoiding instance centers, and **recomputes polygons** to the *visible* region only.
- Outputs occluded images + **new** YOLO‑seg labels consistent with what’s visible.

### Phase 3 — Augmentation
- `augment_yolo_seg_surgery.py` applies augmentation to the occluded set and **renormalizes polygons + bboxes**. Output is YOLO‑seg formatted labels ready for training.

---

## Label Format (YOLO‑seg)

Each line per instance:
```
<class_id> <x1> <y1> <x2> <y2> ... <xK> <yK>
```
- All coordinates are **normalized** to `[0,1]` relative to image width/height.
-  `(x1,y1..)` is the polygon (one or more closed rings depending on your writer—most scripts write a single ring).

---

## Using the Interactive Builders

### Linux/macOS
```bash
chmod +x build_yolo_dataset.sh
./build_yolo_dataset.sh
```
The script prompts for:
- Blender & Python executables
- Script paths
- Synthetic range (`START`, `END`)
- Occlusion source/destination dirs, occluder dir, target occlusion fraction, seed
- Augmentation output dirs, multiplier, seed
---

## Inference

### Images
```bash
python predict.py   --model model_fine_tuned/weights/best.pt   --source /path/to/image_or_folder/*.jpg   --conf 0.25 --iou 0.7 --imgsz 784 --device 0   --save_dir runs/predict_seg --save_masks --show
```
- Writes `<save_dir>/<stem>_seg.png` annotated with masks/boxes/labels.
- `--save_masks` additionally exports per‑instance PNGs and a small JSON (class/conf).

### Video
```bash
python video.py   --model model_fine_tuned/weights/best.pt   --source /path/to/video.mp4   --conf 0.25 --iou 0.7 --imgsz 784 --device 0   --save_dir runs/predict_seg --out_video runs/predict_seg/out_seg.mp4 --show
```
- Default codec is `mp4v`.

---

## Tips & Troubleshooting

- **Ultralytics not found**: `pip install ultralytics==8.*`
- **Torch/CUDA mismatch**: install torch/torchaudio/torchvision from the official index for your CUDA version.
- **CUDA OOM at inference**:
  - Reduce `--imgsz` (e.g., 640)
  - Lower `--conf` if you get too many instances
  - (Advanced) Enable half-precision if supported by your GPU (add `half=True` to `model.predict()` calls)
- **OpenCV cannot write video**: switch fourcc in `video.py` to `XVID` or `avc1`; ensure FFmpeg support is available.
- **Blender path**: on Linux typically `blender`; on Windows `blender.exe`. The builder will prompt.
- **Windows paths**: quote paths with spaces; prefer forward slashes or escape backslashes.
- **Occlusions need YOLO labels**: ensure pre‑occlusion YOLO‑seg labels exist for Phase 2. If you only have binary masks from Phase 1, add the mask→YOLO conversion step (below).

---

### 1) Train YOLOv8m-seg\

```bash
yolo task=segment mode=train model=yolov8m-seg.pt \
     data=/home/student/dataset_YOLO_aug/dataset.yaml \
     epochs=150 patience=50 batch=9 imgsz=784 device=0 workers=4 \
     project=runs/segment name=final_model exist_ok=False \
     pretrained=True optimizer=AdamW verbose=True seed=42 deterministic=True \
     cos_lr=True close_mosaic=10 amp=True cache=ram \
     overlap_mask=True mask_ratio=4 \
     val=True split=val save_json=True plots=True \
     iou=0.7 max_det=300
```

### 2) Inference on your tuned videos → save frames + YOLO-seg pseudo-labels

```bash
yolo task=segment mode=predict model=runs/segment/FINAL_FINAL_FINAL3/weights/best.pt \
     source="vids_tune/*.mp4" imgsz=784 conf=0.25 iou=0.7 device=0 \
     vid_stride=1 stream_buffer=False \
     save=True save_txt=True save_conf=True save_frames=True retina_masks=False \
     project=runs/segment name=PSEUDO_FROM_TUNED_VIDS max_det=300
```

* This writes annotated outputs + **`labels/*.txt`** (YOLO-seg polygons) and **extracted frames** for each video under `runs/segment/PSEUDO_FROM_TUNED_VIDS/*`.

### 3) Self-training (fine-tune on the pseudo-labeled frames from step 2)

```bash
yolo task=segment mode=train model=final_model/weights/best.pt \
     data=/home/student/dataset_YOLO_pseudo/dataset.yaml \
     epochs=50 patience=20 batch=9 imgsz=784 device=0 workers=4 \
     project=runs/segment name=model_fine_tuned exist_ok=False \
     pretrained=True optimizer=AdamW verbose=True seed=42 deterministic=True \
     cos_lr=True amp=True overlap_mask=True mask_ratio=4 \
     val=True split=val plots=True iou=0.7 max_det=300
```

### Example `dataset.yaml` for the pseudo-labeled set

```yaml
# /home/student/dataset_YOLO_pseudo/dataset.yaml
path: /home/student/dataset_YOLO_pseudo
train: images/train         # frames extracted from videos
val: images/val            
nc: 2
names:
  0: needle_holder
  1: tweezers
```

**Expected layout**

```
/home/student/dataset_YOLO_pseudo/
├─ images/
│  ├─ train/              # frame_000001.jpg, frame_000002.jpg, ...
│  └─ val/                # optional
└─ labels/
   ├─ train/              # frame_000001.txt (YOLO-seg), ...
   └─ val/                # optional
```

---

## Notes & Tips

* The **only** training-time flags that matter at inference are things like `imgsz`, `conf`, `iou`, `max_det`, and mask rendering (`retina_masks`). The commands above set them accordingly.
* If you see **CUDA OOM**, lower `imgsz` to 640 or reduce batch size.
* If `save_frames` is supported in your Ultralytics version, you can add `save_frames=True` to step **#2** to force exporting frames; otherwise use FFmpeg as shown.

## License

Use as you like. If you publish results, please credit the original datasets, Blender assets, and Ultralytics YOLOv8.
