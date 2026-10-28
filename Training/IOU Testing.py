# import pandas as pd
# import numpy as np
# from pathlib import Path
# from collections import defaultdict
# from PIL import Image

# def IOU(pred_boxes, gt_boxes):
#     """Compute IoU between all pred and gt boxes."""
#     iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
#     for i, (px1, py1, px2, py2) in enumerate(pred_boxes):
#         pred_area = (px2 - px1) * (py2 - py1)
#         for j, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
#             gt_area = (gx2 - gx1) * (gy2 - gy1)
#             xA = max(px1, gx1)
#             yA = max(py1, gy1)
#             xB = min(px2, gx2)
#             yB = min(py2, gy2)
#             inter_w = max(0, xB - xA)
#             inter_h = max(0, yB - yA)
#             inter_area = inter_w * inter_h
#             union = pred_area + gt_area - inter_area + 1e-6
#             iou = inter_area / union
#             iou_matrix[i, j] = iou
#     return iou_matrix

# # Paths
# label_dir = Path('Week2_2022_combined_video_form/labels/test')
# image_dir = Path('Week2_2022_combined_video_form/images/test')
# pred_base = Path('train/predicted_labels_seen')

# # Load Ground Truth
# gt_data = []
# for label_file in label_dir.glob("*.txt"):
#     with open(label_file) as f:
#         for line in f:
#             cls, x, y, w, h = map(float, line.strip().split())
#             gt_data.append({
#                 "Image": label_file.stem + ".jpg",
#                 "Class": int(cls),
#                 "x_center": x, "y_center": y, "width": w, "height": h
#             })
# gt_df = pd.DataFrame(gt_data)

# # Image sizes
# sizes = []
# for img_path in image_dir.glob("*.jpg"):
#     with Image.open(img_path) as im:
#         w, h = im.size
#     sizes.append({'Image': img_path.name, 'img_width': w, 'img_height': h})
# size_df = pd.DataFrame(sizes)
# gt_df = gt_df.merge(size_df, on='Image')

# # Convert to pixel coords
# gt_df['x_min'] = gt_df['x_center'] * gt_df['img_width'] - (gt_df['width'] * gt_df['img_width']) / 2
# gt_df['y_min'] = gt_df['y_center'] * gt_df['img_height'] - (gt_df['height'] * gt_df['img_height']) / 2
# gt_df['x_max'] = gt_df['x_center'] * gt_df['img_width'] + (gt_df['width'] * gt_df['img_width']) / 2
# gt_df['y_max'] = gt_df['y_center'] * gt_df['img_height'] + (gt_df['height'] * gt_df['img_height']) / 2

# # Threshold
# threshold = 0.75

# # Overall counters
# total_all = 0
# above_all = 0

# TP_all = 0
# FP_all = 0
# FN_all = 0

# # Per-class stats
# class_stats = defaultdict(lambda: {
#     "total": 0, "above": 0, "ious": [],
#     "TP": 0, "FP": 0, "FN": 0
# })

# pred_path = pred_base / "predictions.csv"
# if not pred_path.exists():
#     print("[❌] predictions.csv not found.")
# else:
#     pred_df = pd.read_csv(pred_path)

#     for image_name in pred_df['Image'].unique():
#         preds = pred_df[pred_df['Image'] == image_name]
#         gts = gt_df[gt_df['Image'] == image_name]

#         if len(preds) == 0 and len(gts) > 0:
#             # All GTs missed → FN
#             for _, gt in gts.iterrows():
#                 class_stats[int(gt["Class"])]["FN"] += 1
#                 FN_all += 1
#             continue

#         if len(preds) > 0 and len(gts) == 0:
#             # All preds false → FP
#             for _, pred in preds.iterrows():
#                 class_stats[int(pred["class"])]["FP"] += 1
#                 FP_all += 1
#             continue

#         pred_boxes = preds[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
#         gt_boxes = gts[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
#         iou_matrix = IOU(pred_boxes, gt_boxes)

#         # Track matched GTs to avoid double counting
#         matched_gt = set()

#         for i, (iou_row, (_, pred)) in enumerate(zip(iou_matrix, preds.iterrows())):
#             best_idx = np.argmax(iou_row)
#             best_iou = iou_row[best_idx]
#             cls = int(pred["class"])

#             total_all += 1
#             class_stats[cls]["total"] += 1
#             class_stats[cls]["ious"].append(best_iou)

#             if best_iou > threshold and best_idx not in matched_gt and cls == int(gts.iloc[best_idx]["Class"]):
#                 # True Positive
#                 TP_all += 1
#                 above_all += 1
#                 class_stats[cls]["above"] += 1
#                 class_stats[cls]["TP"] += 1
#                 matched_gt.add(best_idx)
#             else:
#                 # False Positive
#                 FP_all += 1
#                 class_stats[cls]["FP"] += 1

#         # Unmatched GTs → False Negatives
#         for j, gt in gts.iterrows():
#             if j not in matched_gt:
#                 FN_all += 1
#                 class_stats[int(gt["Class"])]["FN"] += 1

# # Overall report
# precision = TP_all / (TP_all + FP_all + 1e-6)
# recall = TP_all / (TP_all + FN_all + 1e-6)

# print("Overall:")
# print(f"   Total Boxes Processed  : {total_all}")
# print(f"   Boxes > {threshold} IOU : {above_all}")
# print(f"   % above threshold      : {above_all / total_all * 100:.2f}%")
# print(f"   TP                     : {TP_all}")
# print(f"   FP                     : {FP_all}")
# print(f"   FN                     : {FN_all}")
# print(f"   Precision              : {precision:.4f}")
# print(f"   Recall                 : {recall:.4f}\n")

# # Per-class report
# print("Per-class stats:")
# for cls, stats in sorted(class_stats.items()):
#     total = stats["total"]
#     above = stats["above"]
#     mean_iou = np.mean(stats["ious"]) if stats["ious"] else 0
#     TP = stats["TP"]
#     FP = stats["FP"]
#     FN = stats["FN"]
#     prec = TP / (TP + FP + 1e-6)
#     rec = TP / (TP + FN + 1e-6)

#     print(f" Class {cls}:")
#     print(f"   Total Boxes            : {total}")
#     print(f"   > {threshold} IOU      : {above}")
#     print(f"   % above threshold      : {above / total * 100:.2f}%")
#     print(f"   Mean IoU               : {mean_iou:.4f}")
#     print(f"   TP={TP}, FP={FP}, FN={FN}")
#     print(f"   Precision={prec:.4f}, Recall={rec:.4f}\n")

import pandas as pd
import numpy as np
from pathlib import Path

def IOU(pred_boxes, gt_boxes):
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

    for i, (px1, py1, px2, py2) in enumerate(pred_boxes):
        pred_area = (px2 - px1) * (py2 - py1)

        for j, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            gt_area = (gx2 - gx1) * (gy2 - gy1)

            xA = max(px1, gx1)
            yA = max(py1, gy1)
            xB = min(px2, gx2)
            yB = min(py2, gy2)

            inter_w = max(0, xB - xA)
            inter_h = max(0, yB - yA)
            inter_area = inter_w * inter_h

            u = pred_area + gt_area - inter_area + 1e-6
            iou = inter_area / u
            iou_matrix[i,j] = iou

    return iou_matrix

# Paths
label_dir = Path('hash_dataset_2.0/labels/test')
image_dir = Path('hash_dataset_2.0/images/test')
pred_base = Path('train/predicted_labels_seen')

# Load Ground Truth
gt_data = []
for label_file in label_dir.glob("*.txt"):
    with open(label_file) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            gt_data.append({
                "Image": label_file.stem + ".jpg",
                "Class": int(cls),
                "x_center": x, "y_center": y, "width": w, "height": h
            })

gt_df = pd.DataFrame(gt_data)

# Get image sizes
from PIL import Image
sizes = []
for img_path in image_dir.glob("*.jpg"):
    with Image.open(img_path) as im:
        w, h = im.size
    sizes.append({'Image': img_path.name, 'img_width': w, 'img_height': h})
size_df = pd.DataFrame(sizes)
gt_df = gt_df.merge(size_df, on='Image')

# Convert to pixel coordinates
gt_df['x_min'] = gt_df['x_center'] * gt_df['img_width'] - (gt_df['width'] * gt_df['img_width']) / 2
gt_df['y_min'] = gt_df['y_center'] * gt_df['img_height'] - (gt_df['height'] * gt_df['img_height']) / 2
gt_df['x_max'] = gt_df['x_center'] * gt_df['img_width'] + (gt_df['width'] * gt_df['img_width']) / 2
gt_df['y_max'] = gt_df['y_center'] * gt_df['img_height'] + (gt_df['height'] * gt_df['img_height']) / 2

# Threshold
threshold = 0.50

total_all = 0
above_all = 0

for fold in range(1):
    pred_path = pred_base / "predictions.csv"
    if not pred_path.exists():
        print(f"[❌] Fold {fold} predictions.csv not found.")
        continue

    pred_df = pd.read_csv(pred_path)
    results = []

    for image_name in pred_df['Image'].unique():
        preds = pred_df[pred_df['Image'] == image_name][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
        gts = gt_df[gt_df['Image'] == image_name][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
        if len(preds) == 0 or len(gts) == 0:
            continue
        ious = IOU(preds, gts).max(axis=1)
        results.extend(ious)

    total = len(results)
    above = np.sum(np.array(results) > threshold)
    below = total - above

    total_all += total
    above_all += above

    print(f"Fold {fold}:")
    print(f"   Total Bounding Boxes   : {total}")
    print(f"   > 0.75 IOU             : {above}")
    print(f"   <= 0.75 IOU            : {below}")
    print(f"   % above threshold      : {above / total * 100:.2f}%\n")

# Overall
print(f"Overall:")
print(f"   Total Boxes Processed  : {total_all}")
print(f"   Boxes > 0.75 IOU       : {above_all}")
print(f"   % above threshold      : {above_all / total_all * 100:.2f}%")