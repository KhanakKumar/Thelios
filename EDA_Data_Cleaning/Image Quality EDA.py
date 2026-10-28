import cv2
from pathlib import Path
import pandas as pd

# Folder of images
img_dir = Path("dataset/images")
blurry_threshold = 100.0  # You can tune this (typical values: 50–150)

results = []

for img_path in img_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Couldn't read {img_path.name}, skipping.")
        continue

    # Compute Laplacian variance
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()

    results.append({
        "Image": img_path.name,
        "Laplacian_Variance": lap_var,
        "Blurry": lap_var < blurry_threshold
    })

# Create DataFrame
df_blur = pd.DataFrame(results)
print(df_blur.head())

# Save results
df_blur.to_csv("blur_check_results.csv", index=False)
print("✅ Blur analysis saved to blur_check_results.csv")
