import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Integrate wandb
import wandb

# Try to import YOLOv9 - adjust import based on your installation
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Alternative import for official YOLOv9 repository
try:
    import sys
    sys.path.append('yolov9t')  # Add YOLOv9 repo path if needed
    from train import train
    OFFICIAL_YOLOV9_AVAILABLE = True
except ImportError:
    OFFICIAL_YOLOV9_AVAILABLE = False

def validate_dataset_yaml(yaml_path: str) -> dict:
    """
    Validate and load dataset.yaml file.
    
    Args:
        yaml_path: Path to dataset.yaml file
        
    Returns:
        Dictionary containing dataset configuration
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['train', 'val', 'names']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in dataset.yaml")
    
    # Validate paths exist
    for split in ['train', 'val']:
        if split in data:
            path = data[split]
            if not os.path.exists(path):
                # Try relative to yaml file
                yaml_dir = os.path.dirname(yaml_path)
                alt_path = os.path.join(yaml_dir, path)
                if os.path.exists(alt_path):
                    data[split] = alt_path
                else:
                    raise FileNotFoundError(f"Dataset path not found: {path}")
    
    # Get number of classes
    if isinstance(data['names'], dict):
        data['nc'] = len(data['names'])
    elif isinstance(data['names'], list):
        data['nc'] = len(data['names'])
        # Convert list to dict
        data['names'] = {i: name for i, name in enumerate(data['names'])}
    
    return data

def get_model_config(model_size: str = 'small_v9') -> str:
    """
    Get YOLOv9 model configuration.
    
    Args:
        model_size: Model size ('small', 'medium', 'large', 'extra_large')
        
    Returns:
        Model configuration path or name
    """
    model_configs = {
        'nano_v11': 'yolo11n.pt',
        'small_v11': 'yolo11s.pt',
        'medium_v11': 'yolov11m.pt', 
        'large_v11': 'yolov11l.pt',
        'small_v9': 'yolov9s.pt',
        'medium_v9': 'yolov9m.pt', 
        'large_v9': 'yolov9l.pt',
        'extra_large_v9': 'yolov9x.pt'
    }
    
    return model_configs.get(model_size, model_configs[model_size])

def train_with_ultralytics(dataset_yaml: str, model_size: str = 'small_v9', 
                          epochs: int = 30, batch_size: int = 16, 
                          img_size: int = 640, output_dir: str = './runs/train'):
    """
    Train YOLOv9 using Ultralytics implementation.
    
    Args:
        dataset_yaml: Path to dataset.yaml file
        model_size: Model size configuration
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        output_dir: Output directory for training results
    """
    
    # Load pre-trained model
    model_name = get_model_config(model_size)
    print(f"Loading model: {model_name}")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        raise
    
    # Training parameters
    train_params = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'project': output_dir,
        'name': f'yolov11_{model_size}{datetime.now().strftime("%Y%m%d%H%M%S")}',
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,
        'device': 'cuda',  # Auto-detect GPU/CPU
        'workers': 8,
        'patience': 25,
        'optimizer': 'Adam',
        'lr0': 0.01,
        'lrf': 0.01, #final learning rate = 0.0001
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'auto_augment' : 'augmix',
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'iou': 0.6,
        'max_det': 3,
        'fraction': 1.0,
        'multi_scale': True,
        'augment': True,
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': True,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': False,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': 12,  
        'multi_scale': False,
        'overlap_mask': True,
        'dropout': 0.3,
        'val': True,
        'plots': True,
        'exist_ok': False,
        'pretrained': True,
        'task': 'detect',
        'save_frames': True,
        'save_txt': True,
        'save_json': True,
        'save_conf': True,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True
    }
    
    print(f"Starting training with parameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {img_size}")
    print(f"  - Model: {model_name}")
    
    # Initialize wandb run
    wandb.init(
        project="YOLOv9-Training",
        name=train_params['name'],
        config=train_params
    )

    # Start training
    try:
        results = model.train(**train_params)
        print("Training completed successfully!")
        # Log final results to wandb
        wandb.log({"results": str(results)})
        wandb.finish()
        return results
    except Exception as e:
        print(f"Training failed: {e}")
        raise

# def create_training_script(dataset_yaml: str, output_dir: str = './runs/train'):
#     """
#     Create a complete training script with all configurations.
    
#     Args:
#         dataset_yaml: Path to dataset.yaml file
#         output_dir: Output directory for training results
#     """
    
#     # Validate dataset
#     print("Validating dataset configuration...")
#     dataset_config = validate_dataset_yaml(dataset_yaml)
#     print(f"Dataset validated. Found {dataset_config['nc']} classes: {list(dataset_config['names'].values())}")
    
#     # Check GPU availability
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     if device == 'cuda':
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA Version: {torch.version.cuda}")
#         print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Copy dataset.yaml to output directory for reference
#     shutil.copy2(dataset_yaml, os.path.join(output_dir, 'dataset_config.yaml'))
    
#     # Determine batch size based on available memory
#     if device == 'cuda':
#         gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
#         # if gpu_memory < 8:
#         #     batch_size = 8
#         # elif gpu_memory < 16:
#         #     batch_size = 16
#         # else:
#         #     batch_size = 32
#         batch_size = 8
#     else:
#         batch_size = 4
    
#     print(f"Using batch size: {batch_size}")
    
#     # Start training
#     if ULTRALYTICS_AVAILABLE:
#         print("Using Ultralytics YOLOv9 implementation")
#         results = train_with_ultralytics(
#             dataset_yaml=dataset_yaml,
#             model_size='medium',
#             epochs=30,
#             batch_size=32,
#             img_size=640,
#             output_dir=output_dir
#         )
#         return results
#     else:
#         print("No YOLOv9 implementation available. Please install ultralytics or clone YOLOv9 repository.")
#         raise ImportError("YOLOv9 not available")

if _name_ == "_main_":
    results = train_with_ultralytics(
        dataset_yaml="/home/adit/work/yolo_hash_training/hash_annotated_dataset copy/dataset.yaml",
        model_size="small_v9",  # or "small", "large", "extra_large"
        epochs=130,
        batch_size=16,
        img_size=640
    )