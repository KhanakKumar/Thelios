import os
import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
LABELS_FOLDER = "annotated_dataset/2017/labels"  # Replace with your actual labels folder
IMAGES_FOLDER = "annotated_dataset/2017/images"  # Replace with folder where corresponding images are
OUTPUT_FOLDER = "filtered_labels"
AREA_THRESHOLD = 500  # Pixels
COLOR_TOLERANCE = 30  # Tolerance for color match

Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def is_mostly_green_or_blue(crop):
    # Calculate average color in BGR
    avg_color = crop.mean(axis=(0, 1))
    blue, green, red = avg_color

    if (green > red + COLOR_TOLERANCE and green > blue + COLOR_TOLERANCE):
        return True  # Mostly green
    if (blue > red + COLOR_TOLERANCE and blue > green + COLOR_TOLERANCE):
        return True  # Mostly blue
    return False

def process_labels(label_file):
    image_name = Path(label_file).stem + ".jpg"  # or ".png"
    image_path = os.path.join(IMAGES_FOLDER, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found for {label_file}, skipping.")
        return

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    new_lines = []
    with open(os.path.join(LABELS_FOLDER, label_file), 'r') as f:
        lines = f.readlines()

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.strip().split())
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_area = (x2 - x1) * (y2 - y1)
        if box_area < AREA_THRESHOLD:
            continue  # Skip small boxes

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or is_mostly_green_or_blue(crop):
            continue  # Skip green/blue only

        new_lines.append(line)

    # Save filtered labels
    with open(os.path.join(OUTPUT_FOLDER, label_file), 'w') as f:
        f.writelines(new_lines)

def main():
    for file in os.listdir(LABELS_FOLDER):
        if file.endswith('.txt'):
            process_labels(file)
    print("Filtering complete.")

if __name__ == "__main__":
    main()
