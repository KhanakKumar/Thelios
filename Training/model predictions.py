from ultralytics import YOLO
from pathlib import Path
import pandas as pd

image_dir = Path('') #add image dir path
pred_folder = Path('') #add lab dir path

model = YOLO("")  # load a model
pred_folder.mkdir(parents=True, exist_ok=True)

predictions = []

for img_path in image_dir.glob("*.jpg"):
    results = model(img_path)

    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        predictions.append({
            "Image": img_path.name,
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2,
            "confidence": conf,
            "class": cls
        })
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(pred_folder / "predictions.csv", index=False)
print(f"[✔] Saved")

# from ultralytics import YOLO
# from pathlib import Path
# import cv2

# image_dir = Path('images')
# pred_folder = Path('labels')

# model = YOLO("hash_models/yolo11/normal_training_v112/weights/best.pt")  # load a custom model
# pred_folder.mkdir(parents=True, exist_ok=True)

# for img_path in image_dir.glob("*.jpg"):
#     results = model(img_path)

#     # Load image to get width and height for normalization
#     img = cv2.imread(str(img_path))
#     h, w, _ = img.shape

#     label_path = pred_folder / f"{img_path.stem}.txt"

#     with open(label_path, "w") as f:
#         boxes = results[0].boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             cls = int(box.cls[0].item())

#             # Convert to YOLO format
#             x_center = ((x1 + x2) / 2) / w
#             y_center = ((y1 + y2) / 2) / h
#             bw = (x2 - x1) / w
#             bh = (y2 - y1) / h

#             f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

# print(f"[✔] Saved label files to {pred_folder}")