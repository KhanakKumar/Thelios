import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict

# -------------------------------
# 1. Load pretrained ResNet18
# -------------------------------
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC layer
resnet.eval()

# -------------------------------
# 2. Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_vector(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = resnet(x).squeeze().numpy()
    return vec / np.linalg.norm(vec)  # normalize to unit vector

def get_all_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

# -------------------------------
# 3. Collect embeddings
# -------------------------------
image_folder = os.path.join("digit_dataset", "images")
image_paths = get_all_image_paths(image_folder)

vectors = {}
for img_path in image_paths:
    try:
        vectors[img_path] = get_vector(img_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# -------------------------------
# 4. Find duplicates
# -------------------------------
similarity_threshold = 0.95  # adjust as needed (0.95 = very close, 0.99 = almost exact)

duplicates = defaultdict(list)
checked = set()

for i, img1 in enumerate(image_paths):
    if img1 in checked:
        continue
    group = [img1]
    for j in range(i + 1, len(image_paths)):
        img2 = image_paths[j]
        sim = np.dot(vectors[img1], vectors[img2])  # cosine similarity
        if sim >= similarity_threshold:
            group.append(img2)
            checked.add(img2)
    if len(group) > 1:
        duplicates[img1] = group
        checked.update(group)

# -------------------------------
# 5. Print results
# -------------------------------
if duplicates:
    print("\nFound duplicate groups:")
    for base, group in duplicates.items():
        print("\nGroup:")
        for img in group:
            print("  ", os.path.basename(img))
else:
    print("No duplicates found.")
