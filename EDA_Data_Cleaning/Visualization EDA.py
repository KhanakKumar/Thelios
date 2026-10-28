from pathlib import Path
import cv2

img_dir    = Path("digit_dataset/images/test")
label_dir  = Path("digit_dataset/labels/test")
out_dir    = Path("output_aug")
out_dir.mkdir(exist_ok=True)

for img_path in img_dir.glob("*.jpg"):
    label_path = label_dir / f"{img_path.stem}.txt"

    img   = cv2.imread(str(img_path))
    h, w, _ = img.shape

    with label_path.open("r") as f:
        for line in f:
            class_id, x_c, y_c, bw, bh = map(float, line.strip().split())

            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            box_width_px = x2 - x1
            label = f"w: {box_width_px}px"
            cv2.putText(img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


    out_file = out_dir / img_path.name
    cv2.imwrite(str(out_file), img)

print("✅ All done.")