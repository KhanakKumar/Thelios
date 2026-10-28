from ultralytics import YOLO
import wandb
from pathlib import Path

# Set your dataset directory and dataset.yaml path
DATA_DIR = Path('C:\\Users\\Khanak Kumar\\Documents\\Projects\\Thelios 2.0\\hash_dataset_1')
DATA_YAML = DATA_DIR / "dataset.yaml"

# Hyperparameters
epochs = 150
imgsz = 640
batch_size = 16
patience = 20
optimizer = "AdamW"
lr0 = 0.01
lrf = 0.01
momentum = 0.937
weight_decay = 0.0005
warmup_epochs = 3
warmup_momentum = 0.8
freeze = 11
box = 7.5
cls = 0.5
dfl = 1.5

if __name__ == "__main__":
    wandb.login()

    wandb.init(
        project="hash_detection_v11",
        name="YOLOv11n_normal_training",
        config={
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "patience": patience,
            "learning_rate": lr0,
            "lr_multiplier": lrf,
            'device': 'cuda',  # Auto-detect GPU/CPU
            'workers': 8,
            'freeze': freeze,
            "warmup_epochs": warmup_epochs,
            "momentum": momentum,
            "auto_augment": "augmix",
            "warmup_momentum": warmup_momentum,
            "weight_decay": weight_decay,
            "model": "yolov11n",
            "data_yaml": str(DATA_YAML)
        }
    )

    model = YOLO("YOLOv11n/normal_training_v114/weights/best.pt")  # load a pretrained model (recommended for training)
    print("Training Started")

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch_size,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        device='cuda',  # Auto-detect GPU/CPU
        workers=8,
        warmup_epochs=warmup_epochs,
        momentum=momentum,
        warmup_momentum=warmup_momentum,
        weight_decay=weight_decay,
        freeze=freeze,
        auto_augment='augmix',
        box=box,
        cls=cls,
        dfl=dfl,
        project="YOLOv11n",
        name="normal_training_v11",
    )

    print("Training completed.")
    wandb.finish()
