import os
import argparse
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
    "gl": 0,
    "me": 1,
    "no": 2,
    "pi": 3
}

def mask_to_polygons(mask_path, class_id):
    """Converts a grayscale mask to YOLO polygon format with noise filtering."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = mask.shape
    polygons = []
    min_area_threshold = (h * w) * 0.001
    
    for contour in contours:
        if len(contour) < 3:
            continue
            
        area = cv2.contourArea(contour)
        if area < min_area_threshold:
            continue
            
        poly = contour.reshape(-1, 2).astype(np.float32)
        poly[:, 0] /= w
        poly[:, 1] /= h
        poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
        polygons.append(f"{class_id} {poly_str}")
    
    return "\n".join(polygons)

def prepare_segmentation_labels(base_dir):
    """Generates .txt label files from .png masks for YOLO segmentation."""
    print(f"Checking segmentation labels in {base_dir}...")
    for split in ["train", "test"]:
        split_dir = Path(base_dir) / split
        mask_dir = split_dir / "masks"
        label_dir = split_dir / "labels"
        
        if label_dir.exists() and any(label_dir.iterdir()):
            print(f"Labels already exist in {label_dir}, skipping conversion.")
            continue
            
        label_dir.mkdir(parents=True, exist_ok=True)
        mask_files = list(mask_dir.glob("*.png"))
        
        for mask_path in mask_files:
            parts = mask_path.stem.split("_")
            class_key = parts[3] if len(parts) > 3 else "gl"
            class_id = CLASS_MAP.get(class_key, 0)
            
            polygons = mask_to_polygons(mask_path, class_id)
            
            label_path = label_dir / f"{mask_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(polygons)
    print("Segmentation labels ready.")

def create_seg_yaml(data_root):
    """Creates the data.yaml file for YOLO segmentation."""
    data_root = Path(data_root).absolute()
    yaml_content = {
        "path": str(data_root),
        "train": "train/images",
        "val": "test/images",
        "names": {
            0: "glioma",
            1: "meningioma",
            2: "no_tumor",
            3: "pituitary"
        }
    }
    yaml_path = data_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return str(yaml_path)

def train_classification(data_dir, results_dir, model_out_dir, epochs=50, batch=32, imgsz=320):
    """Trains YOLO11 classification model."""
    print("Starting Classification Training...")
    
    # YOLO classification expects 'train' and 'val' folders.
    # If 'val' is missing but 'test' exists, we'll use 'test' as 'val'.
    data_path = Path(data_dir)
    val_path = data_path / "val"
    test_path = data_path / "test"
    
    if not val_path.exists() and test_path.exists():
        print(f"Creating symlink from {test_path} to {val_path} for YOLO validation...")
        try:
            os.symlink(test_path.name, val_path)
        except OSError:
            # Fallback for environments where symlinks might fail
            shutil.copytree(test_path, val_path, dirs_exist_ok=True)

    model = YOLO("yolo11m-cls.pt")
    
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=results_dir,
        name="classification",
        exist_ok=True,   
        device=0 # RTX 5060 Ti
    )
    
    # Save best model
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        target_path = Path(model_out_dir) / "yolo11m-cls-brain-best.pt"
        shutil.copy(best_model, target_path)
        print(f"Best classification model saved to {target_path}")

def train_segmentation(data_dir, results_dir, model_out_dir, epochs=50, batch=32, imgsz=320):
    """Trains YOLO11 segmentation model."""
    print("Starting Segmentation Training...")
    
    # 1. Prepare labels
    prepare_segmentation_labels(data_dir)
    
    # 2. Create YAML
    yaml_path = create_seg_yaml(data_dir)
    
    # 3. Train
    model = YOLO("yolo11m-seg.pt")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=results_dir,
        name="segmentation",
        workers=2,
        exist_ok=True,
        device=0
    )
    
    # Save best model
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        target_path = Path(model_out_dir) / "yolo11m-seg-brain-best.pt"
        shutil.copy(best_model, target_path)
        print(f"Best segmentation model saved to {target_path}")

def main():
    parser = argparse.ArgumentParser(description="Brain MRI YOLO11 Training Pipeline")
    parser.add_argument("--task", type=str, default="both", choices=["cls", "seg", "both"],
                        help="Task to perform: cls (classification), seg (segmentation), or both")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size")
    
    args = parser.parse_args()
    
    # Paths
    base_path = Path(__file__).resolve().parents[2]
    data_cls = base_path / "brisc2025" / "classification_task"
    data_seg = base_path / "brisc2025" / "segmentation_task"
    results_dir = base_path / "results"
    models_dir = base_path / "models"
    
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    try:
        if args.task in ["cls", "both"]:
            train_classification(
                str(data_cls), str(results_dir), str(models_dir),
                epochs=args.epochs, batch=args.batch, imgsz=args.imgsz
            )
            
        if args.task in ["seg", "both"]:
            train_segmentation(
                str(data_seg), str(results_dir), str(models_dir),
                epochs=args.epochs, batch=args.batch, imgsz=args.imgsz
            )
            
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# # 분류 및 분할 학습 전체 실행
# python src/training/main.py --task both --epochs 50
# # 분류 학습만 실행
# python src/training/main.py --task cls --epochs 50 --batch 32
# # 세그멘테이션 학습만 실행
# python src/training/main.py --task seg --epochs 50 --imgsz 640