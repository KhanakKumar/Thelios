from pathlib import Path
from PIL import Image

# Path to your folder
img_dir = Path("digit_dataset/images")

# Loop through all PNG images
for img_path in img_dir.glob("*.png"):
    # Open image
    img = Image.open(img_path).convert("RGB")  # convert to RGB so JPG works
    
    # New filename with .jpg extension
    new_path = img_path.with_suffix(".jpg")
    
    # Save as JPG
    img.save(new_path, "JPEG")
    
    print(f"Converted: {img_path.name} -> {new_path.name}")
