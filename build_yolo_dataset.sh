#!/usr/bin/env bash
set -euo pipefail

echo "=== YOLO Dataset Builder (Linux/macOS) ==="

# --- Locate executables and scripts ---
read -r -p "Path to Blender executable [blender]: " BLENDER
BLENDER=${BLENDER:-blender}

read -r -p "Python interpreter [python3]: " PYTHON
PYTHON=${PYTHON:-python3}

read -r -p "Path to synthetic_data_generator.py [./synthetic_data_generator.py]: " SYNTH_PY
SYNTH_PY=${SYNTH_PY:-./synthetic_data_generator.py}

read -r -p "Path to add_occlusions.py [./add_occlusions.py]: " ADD_OCC_PY
ADD_OCC_PY=${ADD_OCC_PY:-./add_occlusions.py}

read -r -p "Path to augment_yolo_seg_surgery.py [./augment_yolo_seg_surgery.py]: " AUG_PY
AUG_PY=${AUG_PY:-./augment_yolo_seg_surgery.py}

# --- Synthetic generation ---
read -r -p "Synthetic START index [0]: " START
START=${START:-0}
read -r -p "Synthetic END index (exclusive) [600]: " END
END=${END:-600}

echo ""
echo ">> Running Blender synthetic generator: ${START}..${END}"
"${BLENDER}" --background --python "${SYNTH_PY}" -- "${START}" "${END}"

# --- Paths for occlusion stage (we'll pass them through a wrapper) ---
read -r -p "Dataset root for occlusion stage [../synthetic_output]: " DATASET_ROOT
DATASET_ROOT=${DATASET_ROOT:-../synthetic_output}

default_src_img="${DATASET_ROOT}/images1"
default_src_lbl="${DATASET_ROOT}/yolo_labels_kerpel_ver/images1"
default_dst_img="${DATASET_ROOT}/images1_occ"
default_dst_lbl="${DATASET_ROOT}/yolo_labels_occ/images1"
default_occ_dir="occluders"

read -r -p "Source images dir [${default_src_img}]: " SRC_IMG_DIR
SRC_IMG_DIR=${SRC_IMG_DIR:-${default_src_img}}

read -r -p "Source YOLO labels dir [${default_src_lbl}]: " SRC_LBL_DIR
SRC_LBL_DIR=${SRC_LBL_DIR:-${default_src_lbl}}

read -r -p "Destination images dir [${default_dst_img}]: " DST_IMG_DIR
DST_IMG_DIR=${DST_IMG_DIR:-${default_dst_img}}

read -r -p "Destination labels dir [${default_dst_lbl}]: " DST_LBL_DIR
DST_LBL_DIR=${DST_LBL_DIR:-${default_dst_lbl}}

read -r -p "Occluders directory [${default_occ_dir}]: " OCCLUDER_DIR
OCCLUDER_DIR=${OCCLUDER_DIR:-${default_occ_dir}}

read -r -p "Target occlusion fraction (0-1) [0.30]: " TARGET_OCC_FRAC
TARGET_OCC_FRAC=${TARGET_OCC_FRAC:-0.30}

read -r -p "Random seed [123]: " RAND_SEED
RAND_SEED=${RAND_SEED:-123}

# --- Make a tiny wrapper so we can parameterize add_occlusions.py without editing it ---
WRAP_PY="$(mktemp /tmp/run_add_occlusions_XXXX.py)"
cat > "${WRAP_PY}" << 'PYEOF'
import importlib.util, sys
from pathlib import Path

# args: 0=add_occ_py 1=dataset_root 2=src_img 3=src_lbl 4=dst_img 5=dst_lbl 6=occluder_dir 7=target_occ_frac 8=rand_seed
if len(sys.argv) < 9:
    print("Usage: run_add_occlusions.py <add_occlusions.py> <dataset_root> <src_img> <src_lbl> <dst_img> <dst_lbl> <occluder_dir> <target_occ_frac> <rand_seed>")
    sys.exit(2)

script_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("add_occ", script_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.DATASET_ROOT = Path(sys.argv[2])
m.SRC_IMG_DIR  = Path(sys.argv[3])
m.SRC_LBL_DIR  = Path(sys.argv[4])
m.DST_IMG_DIR  = Path(sys.argv[5])
m.DST_LBL_DIR  = Path(sys.argv[6])
m.OCCLUDER_DIR = Path(sys.argv[7])
m.TARGET_OCC_FRAC = float(sys.argv[8])
try:
    m.RAND_SEED = int(sys.argv[9])
except Exception:
    pass

if __name__ == "__main__":
    m.main()
PYEOF

echo ""
echo ">> Running occlusion stage (wrapper -> add_occlusions.py)"
"${PYTHON}" "${WRAP_PY}" "${ADD_OCC_PY}" "${DATASET_ROOT}" "${SRC_IMG_DIR}" "${SRC_LBL_DIR}" "${DST_IMG_DIR}" "${DST_LBL_DIR}" "${OCCLUDER_DIR}" "${TARGET_OCC_FRAC}" "${RAND_SEED}"

# --- Augmentation ---
echo ""
echo ">> Running augmentation stage"
default_aug_img="${DATASET_ROOT}/images1_occ_aug"
default_aug_lbl="${DATASET_ROOT}/yolo_labels_aug/images1"

read -r -p "Augmented images output dir [${default_aug_img}]: " AUG_IMG_DIR
AUG_IMG_DIR=${AUG_IMG_DIR:-${default_aug_img}}

read -r -p "Augmented labels output dir [${default_aug_lbl}]: " AUG_LBL_DIR
AUG_LBL_DIR=${AUG_LBL_DIR:-${default_aug_lbl}}

read -r -p "Augmentation multiplier (per image) [3]: " MULTIPLIER
MULTIPLIER=${MULTIPLIER:-3}

read -r -p "Augmentation seed [0]: " AUG_SEED
AUG_SEED=${AUG_SEED:-0}

"${PYTHON}" "${AUG_PY}" \
  --images_dir "${DST_IMG_DIR}" \
  --labels_dir "${DST_LBL_DIR}" \
  --out_images_dir "${AUG_IMG_DIR}" \
  --out_labels_dir "${AUG_LBL_DIR}" \
  --multiplier "${MULTIPLIER}" \
  --seed "${AUG_SEED}"

echo ""
echo "=== DONE ==="
echo "Synthetic images: see paths configured in synthetic_data_generator.py"
echo "Occluded set:     ${DST_IMG_DIR}  + labels -> ${DST_LBL_DIR}"
echo "Augmented set:    ${AUG_IMG_DIR}  + labels -> ${AUG_LBL_DIR}"
