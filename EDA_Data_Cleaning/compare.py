from pathlib import Path

# ───────────────────────────────────────────────────────────────
# 1. CONFIGURE PATHS
DIR = Path("hash_annotated_dataset_khanak/dataset2")  # root folder
IMG_DIR   = DIR / "images" / "test"            # folder containing .jpg / .png
LABEL_DIR = DIR / "labels" / "test"           # folder containing .txt
IMG_EXTS  = {".jpg", ".jpeg", ".png"} # recognised image extensions (lower-case)

# ───────────────────────────────────────────────────────────────
# 2. SAFETY CHECK
if not IMG_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
if not LABEL_DIR.exists():
    raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

# ───────────────────────────────────────────────────────────────
# 3. COLLECT FILE STEMS
images = {p.stem for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS}
labels = {p.stem for p in LABEL_DIR.glob("*.txt")}

# ───────────────────────────────────────────────────────────────
# 4. RELATIONSHIPS
matched_pairs   = sorted(images & labels)   # both exist
missing_images  = sorted(labels - images)   # label exists but image missing
missing_labels  = sorted(images - labels)   # image exists but label missing

# ───────────────────────────────────────────────────────────────
# 5. REPORT
print(f"Found {len(images)} image files.")
print(f"Found {len(labels)} label files.")
print(f"✓ Matched pairs      : {len(matched_pairs)}")
print(f"✗ Missing images     : {len(missing_images)}")
print(f"✗ Missing labels     : {len(missing_labels)}")

# ───────────────────────────────────────────────────────────────
# 6. SAVE MISSING FILES TO TXT
def dump_list(filename, items):
    with open(filename, "w") as f:
        for it in items:
            f.write(it + "\n")

dump_list("missing_images.txt", missing_images)
dump_list("missing_labels.txt", missing_labels)

print("\nLists saved as  missing_images.txt, missing_labels.txt")
