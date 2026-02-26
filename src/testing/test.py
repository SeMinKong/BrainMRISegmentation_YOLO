import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def run_test(task, model_path, data_dir, results_dir, imgsz=320, conf=0.25):
    """Loads a model and runs inference on all test images."""
    print(f"Loading {task} model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # Task-specific setup
    output_dir = Path(results_dir) / f"{task}_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target images
    if task == "cls":
        # For classification, we look into subdirectories of test
        image_files = list(Path(data_dir).rglob("*.jpg"))
    else:
        # For segmentation, we look into images folder
        image_files = list((Path(data_dir) / "images").glob("*.jpg"))

    if not image_files:
        print(f"No test images found in {data_dir}")
        return

    print(f"Running inference on {len(image_files)} images...")
    
    # Process in batches for better performance if supported, 
    # but here we iterate for clearer per-image reporting
    for img_path in image_files:
        results = model.predict(
            source=str(img_path), 
            conf=conf, 
            imgsz=imgsz, 
            save=True, 
            project=str(output_dir.parent), 
            name=output_dir.name,
            exist_ok=True
        )

        # Print summary for each image
        for r in results:
            print(f"Image: {img_path.name}", end=" | ")
            if task == "cls":
                top1_idx = r.probs.top1
                top1_conf = r.probs.top1conf.item()
                print(f"Class: {r.names[top1_idx]} ({top1_conf:.2f})")
            else:
                det_count = len(r.boxes)
                mask_count = len(r.masks) if r.masks else 0
                print(f"Detections: {det_count}, Masks: {mask_count}")

    print(f"Inference complete. All results saved to {output_dir}")

def run_integrated_test(cls_weights, seg_weights, data_dir, results_dir, imgsz=320, conf=0.25):
    """Runs both classification and segmentation on all test images and combines them."""
    print(f"Running integrated inference on all images...")
    
    if not os.path.exists(cls_weights) or not os.path.exists(seg_weights):
        print("Error: One or both model weights not found.")
        return

    cls_model = YOLO(cls_weights)
    seg_model = YOLO(seg_weights)

    # Use segmentation test images as target
    image_files = list((Path(data_dir) / "images").glob("*.jpg"))
    if not image_files:
        print(f"No test images found in {data_dir}/images")
        return

    output_dir = Path(results_dir) / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        img_name = img_path.name
        
        # 1. Run Classification
        cls_results = cls_model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)
        top1_idx = cls_results[0].probs.top1
        cls_label = cls_results[0].names[top1_idx]
        cls_conf = cls_results[0].probs.top1conf.item()

        # 2. Run Segmentation
        seg_results = seg_model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)
        
        # 3. Combine Results visually
        annotated_img = seg_results[0].plot()

        # Overlay classification label
        label_text = f"Class: {cls_label} ({cls_conf:.2f})"
        cv2.putText(annotated_img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Save the combined result
        save_path = output_dir / img_name
        cv2.imwrite(str(save_path), annotated_img)
        
        print(f"Processed {img_name}: {cls_label} ({cls_conf:.2f}), {len(seg_results[0].boxes)} masks")

    print(f"Integrated inference complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Brain MRI YOLO11 Inference Test")
    parser.add_argument("--task", type=str, required=True, choices=["cls", "seg", "integrated"],
                        help="Task type: cls, seg, or integrated")
    parser.add_argument("--weights", type=str, help="Path to custom weights (optional)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Paths
    base_path = Path(__file__).resolve().parents[2]
    results_dir = base_path / "results"
    
    # Default weight paths based on current project structure
    cls_default = base_path / "models" / "yolo11m-cls-brain-best.pt"
    seg_default = base_path / "results" / "segmentation" / "weights" / "best.pt"

    if args.task == "cls":
        model_path = args.weights or cls_default
        data_dir = base_path / "brisc2025" / "classification_task" / "test"
        run_test(args.task, str(model_path), str(data_dir), str(results_dir), conf=args.conf)
    elif args.task == "seg":
        model_path = args.weights or seg_default
        data_dir = base_path / "brisc2025" / "segmentation_task" / "test"
        run_test(args.task, str(model_path), str(data_dir), str(results_dir), conf=args.conf)
    elif args.task == "integrated":
        seg_data_dir = base_path / "brisc2025" / "segmentation_task" / "test"
        run_integrated_test(str(cls_default), str(seg_default), str(seg_data_dir), str(results_dir), conf=args.conf)

if __name__ == "__main__":
    main()



# # 1. 분류 모델 테스트
# python src/testing/test.py --task cls --conf 0.3
# # 2. 세그멘테이션 모델 테스트
# python src/testing/test.py --task seg --conf 0.3
# # 3. 통합 테스트 (분류 + 세그멘테이션 합쳐서 보여주기)
# python src/testing/test.py --task integrated --conf 0.3