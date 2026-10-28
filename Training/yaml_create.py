import os

def count_files(folder, extensions=None):
    """Count files in folder with given extensions"""
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and
        (extensions is None or f.lower().endswith(extensions))
    ])

def create_custom_yaml(dataset_dir, output_file='dataset.yaml'):
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    train_img = os.path.join(images_dir, 'train')
    test_img = os.path.join(images_dir, 'test')
    train_lbl = os.path.join(labels_dir, 'train')
    test_lbl = os.path.join(labels_dir, 'test')

    train_images = count_files(train_img, extensions=('.jpg', '.jpeg', '.png'))
    test_images = count_files(test_img, extensions=('.jpg', '.jpeg', '.png'))
    train_labels = count_files(train_lbl, extensions=('.txt',))
    test_labels = count_files(test_lbl, extensions=('.txt',))

    total_images = train_images + test_images
    total_labels = train_labels + test_labels

    # Write exactly like your template
    content = f"""# Dataset configuration for YOLO training
# Hash mark detection dataset

# Dataset paths
path: {dataset_dir}  # Root dataset directory
train: images/train  # Train images directory (relative to 'path')
val: images/test     # Validation images directory (relative to 'path')
test: images/test    # Test images directory (same as validation)

# Classes
nc: 1  # Number of classes
names:
  0: hash

# Dataset info
dataset_name: "Hash Detection"
description: "Dataset for detecting hash marks in images"
version: "1.0"

# Dataset split
train_images: {train_images}
test_images: {test_images}
total_images: {total_images}
train_labels: {train_labels}
test_labels: {test_labels}
total_labels: {total_labels}

# Additional metadata
image_format: "jpg"
annotation_format: "YOLO"
split_ratio: "90% train, 10% test"
"""

    with open(os.path.join(dataset_dir, output_file), 'w') as f:
        f.write(content)

    print(f"Custom dataset.yaml created at {os.path.join(dataset_dir, output_file)}")

# Example usage:
dataset_folder = 'C:\\Users\\Khanak Kumar\\Documents\\Projects\\Thelios 2.0\\hash_dataset_2.0'
create_custom_yaml(dataset_folder)
