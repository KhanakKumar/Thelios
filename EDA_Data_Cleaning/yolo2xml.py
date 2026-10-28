import os
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET

# === PATH SETUP ===
txt_labels_base = Path("hash_annotated_dataset/labels/test")
image_dir = Path("hash_annotated_dataset/images/test")
output_file = Path("cvat_annotations.xml")

# === START BUILDING CVAT XML ===
annotations = ET.Element("annotations")
ET.SubElement(annotations, "version").text = "1.1"

meta = ET.SubElement(annotations, "meta")
task = ET.SubElement(meta, "task")
ET.SubElement(task, "name").text = "YOLO to CVAT"
ET.SubElement(task, "labels")
ET.SubElement(meta, "segments")

# === IMAGE ID COUNTER ===
img_id = 0

# === LOOP THROUGH TXT FILES DIRECTLY ===
for txt_file in sorted(txt_labels_base.glob("*.txt")):
    image_name = txt_file.stem + ".jpg"
    image_path = image_dir / image_name

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[!] Skipping: Image not found for {image_path}")
        continue

    height, width, _ = img.shape

    image_tag = ET.SubElement(annotations, "image", {
        "id": str(img_id),
        "name": image_name,
        "width": str(width),
        "height": str(height)
    })

    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, cx, cy, w, h = map(float, parts)

            # Convert YOLO to absolute coordinates
            x_center = cx * width
            y_center = cy * height
            box_width = w * width
            box_height = h * height

            xtl = x_center - box_width / 2
            ytl = y_center - box_height / 2
            xbr = x_center + box_width / 2
            ybr = y_center + box_height / 2

            ET.SubElement(image_tag, "box", {
                "label": "hash",
                "occluded": "0",
                "source": "manual",
                "xtl": f"{xtl:.2f}",
                "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}",
                "ybr": f"{ybr:.2f}"
            })

    img_id += 1

# === WRITE FINAL XML ===
tree = ET.ElementTree(annotations)
tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"[✔] CVAT XML saved to: {output_file}")
