from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")
path = Path("train/yolo_kfold")

for i in range(5):
    data_file = path / f'fold_{i}' / "data.yaml"
result = model.tune(data=str(data_file), iterations=10, epochs=20, resume=True, device='cuda:0')
print(f"done")

# from pathlib import Path

# path = Path("train/yolo_kfold")

# for i in range(5):
#     yaml = path / f'fold_{i}' / 'data.yaml'

#     print("YAML path:", yaml)
#     print("Exists:", yaml.exists())

# import yaml

# with open("train\data.yaml", "r") as f:
#     content = f.read()
#     print("YAML raw content:\n", content)
#     print("\nParsed content:")
#     print(yaml.safe_load(content))
