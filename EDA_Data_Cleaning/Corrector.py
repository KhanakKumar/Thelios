from pathlib import Path
import cv2

# ───────────────────────────────────────────────────────────────
# 1. Directory setup
IMG_DIR   = Path("extra/1260 images/images")       # images (already augmented)
LABEL_DIR = Path("extra/1260 images/labels")   # matching YOLO txt files

OUT_IMG   = Path("cleaned_dataset/images")
OUT_LBL   = Path("cleaned_dataset/labels")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────────
# 2. Helper functions
def clip_bbox(bbox):
    """Clip YOLO bbox to [0, 1] range."""
    x, y, w, h = bbox
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(w, 1.0 - x)
    h = min(h, 1.0 - y)
    return [x, y, w, h]

def is_valid_bbox(bbox, threshold=0.0007):
    """Check if bbox area > threshold (fraction of image area)."""
    _, _, w, h = bbox
    return w * h > threshold

# ───────────────────────────────────────────────────────────────
# 3. Main cleaning loop
for img_path in sorted(IMG_DIR.glob("*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    label_path = LABEL_DIR / f"{img_path.stem}.txt"
    if not label_path.exists():
        print(f"[WARN] No label for {img_path.name}; skipping.")
        continue

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[WARN] Could not read {img_path.name}; skipping.")
        continue

    # Read bounding boxes
    bboxes, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            bboxes.append([float(x), float(y), float(w), float(h)])
            class_labels.append(int(float(cls)))

    # Clean boxes
    cleaned_boxes, cleaned_labels = [], []
    for box, lbl in zip(bboxes, class_labels):
        box = clip_bbox(box)
        if is_valid_bbox(box):
            cleaned_boxes.append(box)
            cleaned_labels.append(lbl)

    if not cleaned_boxes:
        print(f"[INFO] {img_path.name}: all boxes removed; skipping.")
        continue

    # Save cleaned image and label
    out_img_path = OUT_IMG / img_path.name
    out_lbl_path = OUT_LBL / f"{img_path.stem}.txt"
    cv2.imwrite(str(out_img_path), image)

    with open(out_lbl_path, "w") as f:
        for cls, (x, y, w, h) in zip(cleaned_labels, cleaned_boxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"[CLEANED] {img_path.name}")

print("\nAll done! Cleaned dataset saved to `cleaned_dataset/`.")
