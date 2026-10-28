from pathlib import Path

import cv2

# Input directories (and output dirs as same)
IMG_DIR = Path('dataset/images')
LAB_DIR = Path('dataset/labels')

# Get list of all images and labels
img_list = list(IMG_DIR.glob('*.jpg'))
lab_list = list(LAB_DIR.glob('*.txt'))

# Get stem names
img_stems = {img.stem for img in img_list}
lab_stems = {lab.stem for lab in lab_list}

# Determine matches and mismatches
matched_stems = img_stems & lab_stems
unmatched_images = img_stems - lab_stems
unmatched_labels = lab_stems - img_stems

# Delete unmatched image files
for stem in unmatched_images:
    img_path = IMG_DIR / f"{stem}.jpg"
    if img_path.exists():
        img_path.unlink()

# Delete unmatched label files
for stem in unmatched_labels:
    lab_path = LAB_DIR / f"{stem}.txt"
    if lab_path.exists():
        lab_path.unlink()

# Logging
if unmatched_images:
    print("Deleted unmatched images:")
    print(sorted(unmatched_images))

if unmatched_labels:
    print("Deleted unmatched labels:")
    print(sorted(unmatched_labels))
else:
    print("Done")
