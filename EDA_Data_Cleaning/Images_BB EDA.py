import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Adding images in Dataset to a DataFrame
img_dir = Path('dataset/images')
img_paths = list(img_dir.glob("*.jpg"))
data = []

for path in img_paths:
    try:
        with Image.open(path) as img:
            width, height = img.size
            data.append({
                'Image': path.name,
                'img_width': width,
                'img_height': height
            })
    except Exception as e:
        print(f"Couldn't open {path.name}:{e}")

df1 = pd.DataFrame(data)
df1['aspect ratio'] = df1['img_width']/df1['img_height'] # Calculating Aspect Ratio of Image and adding to df
print(df1.head())

# All images that have less width than 640 are displayed on scatter plot
plt.figure(figsize=(10,5))
plt.scatter(df1.index, df1['img_width'], color='teal')
plt.axhline(640, color='red', linestyle='--', label='Threshold (640)')
plt.ylim([0,1920])
plt.title('Scatter plot of image size')
plt.xlabel('Image Index')
plt.ylabel('Image Width')
plt.legend()
plt.show()

#Adding annotations in labels to a dataframe
labels_dir = Path('dataset/labels')
data = []
for label_file in labels_dir.glob("*.txt"):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id, x, y, w, h = map(float, line.strip().split())
            data.append({
                "Image": label_file.stem + ".jpg",
                "x_center": x,
                "y_center": y,
                "width": w,
                "height": h
            })

df2 = pd.DataFrame(data)
df = pd.merge(df2, df1, on="Image")

# Values are normalized therefore they are converted to pixels
df['bb_x_center_px'] = df['x_center'] * df['img_width']
df['bb_y_center_px'] = df['y_center'] * df['img_height']
df['bb_width_px'] = df['width'] * df['img_width']
df['bb_height_px'] = df['height'] * df['img_height']

df['bb_area_px'] = df['bb_width_px'] * df['bb_height_px']
df['bb_aspect_ratio'] = df['bb_width_px'] / df['bb_height_px']
df.drop(['x_center', 'y_center', 'width', 'height', 'img_width', 'img_height', 'aspect ratio'], axis=1, inplace=True)
print(df.head())

# Bounding Boxes whose area is very less i.e. very small boxes are represented.
plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['bb_area_px'], color='skyblue', s=1)
plt.title('Scatter Plot of Bounding Box areas')
plt.xlabel('Bounding Boxes')
plt.ylabel('Area')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df.index, df['bb_aspect_ratio'], color='skyblue', s=1)
plt.title('Scatter Plot of Bounding Box AR')
plt.xlabel('Bounding Boxes')
plt.ylabel('AR')
plt.grid(True)
plt.show()

df['x_min'] = (df['bb_x_center_px'] - df['bb_width_px']/2)
df['x_max'] = (df['bb_x_center_px'] + df['bb_width_px']/2)
df['y_min'] = (df['bb_y_center_px'] - df['bb_height_px']/2)
df['y_max'] = (df['bb_y_center_px'] + df['bb_height_px']/2)
print(df['x_min'].mean())
print(df['y_min'].mean())
print(df['x_max'].mean())
print(df['y_max'].mean())
df.drop(['bb_x_center_px', 'bb_y_center_px', 'bb_width_px', 'bb_height_px'], axis=1, inplace=True)

plt.figure(figsize=(10,5))
plt.scatter(df['x_min'], df['y_min'], color='limegreen', s=0.2)
plt.scatter(df['x_max'], df['y_max'], color='darkgreen', s=0.2)
plt.axvline(0, color='red', linestyle='--')
plt.axvline(1280, color='red', linestyle='--')
plt.axhline(0, color='red', linestyle='--')
plt.axhline(720, color='red', linestyle='--')
plt.ylim([0,800])
plt.xlim([0,1400])
plt.show()

duplicate = df[df.duplicated()]
print("Duplicated rows: ")
print(duplicate.head())

data = []
count=0
for label_file in labels_dir.glob("*.txt"):
    c=0
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            c+=1
            count+=1
        data.append({
            "Image": label_file.stem + ".jpg",
            "Annotations": c
        })
df3 = pd.DataFrame(data)
print(df3.head())
print(list(df3['Annotations']))
print("Average: ", count/len(data))

plt.figure(figsize=(10, 6))
plt.bar(range(len(df3)), df3['Annotations'], color='teal', edgecolor='black')
plt.xticks(range(len(df3)), df3["Image"], rotation=90, fontsize=8)
plt.xlabel("Image")
plt.ylabel("Number of Annotations")
plt.title("Number of Annotations per Image")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()