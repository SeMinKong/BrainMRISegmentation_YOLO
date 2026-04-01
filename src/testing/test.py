import argparse
import logging
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def run_test(
    task: str,
    model_path: str,
    source: str,
    results_dir: str,
    imgsz: int = 320,
    conf: float = 0.7,
) -> None:
    """Load a model and run inference on test images."""
    logger.info("Loading %s model from %s...", task, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model = YOLO(model_path)

    output_dir = Path(results_dir) / f"{task}_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source)
    if source_path.is_file():
        image_files = [source_path]
    elif task == "cls":
        image_files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.png"))
    else:
        img_subdir = source_path / "images"
        if img_subdir.exists() and img_subdir.is_dir():
            image_files = list(img_subdir.glob("*.jpg")) + list(img_subdir.glob("*.png"))
        else:
            image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))

    if not image_files:
        logger.warning("No test images found in %s", source)
        return

    logger.info("Running inference on %d images...", len(image_files))

    for img_path in image_files:
        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=imgsz,
            save=True,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
        )

        for r in results:
            if task == "cls":
                if r.probs is not None:
                    top1_idx = r.probs.top1
                    top1_conf = r.probs.top1conf.item()
                    logger.info("Image: %s | Class: %s (%.2f)", img_path.name, r.names[top1_idx], top1_conf)
                else:
                    logger.info("Image: %s | No classification results.", img_path.name)
            else:
                det_count = len(r.boxes)
                mask_count = len(r.masks) if r.masks else 0
                logger.info("Image: %s | Detections: %d, Masks: %d", img_path.name, det_count, mask_count)

    logger.info("Inference complete. All results saved to %s", output_dir)


def run_integrated_test(
    cls_weights: str,
    seg_weights: str,
    source: str,
    results_dir: str,
    imgsz: int = 320,
    conf: float = 0.7,
) -> None:
    """Run both classification and segmentation on test images and combine them."""
    logger.info("Running integrated inference...")

    if not os.path.exists(cls_weights):
        raise FileNotFoundError(f"Classification weights not found at {cls_weights}")
    if not os.path.exists(seg_weights):
        raise FileNotFoundError(f"Segmentation weights not found at {seg_weights}")

    cls_model = YOLO(cls_weights)
    seg_model = YOLO(seg_weights)

    source_path = Path(source)
    if source_path.is_file():
        image_files = [source_path]
    else:
        img_subdir = source_path / "images"
        if img_subdir.exists() and img_subdir.is_dir():
            image_files = list(img_subdir.glob("*.jpg")) + list(img_subdir.glob("*.png"))
        else:
            image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))

    if not image_files:
        logger.warning("No test images found in %s", source)
        return

    output_dir = Path(results_dir) / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %d images...", len(image_files))

    for img_path in image_files:
        img_name = img_path.name

        # 1. Run Classification
        cls_results = cls_model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)
        if cls_results[0].probs is not None:
            top1_idx = cls_results[0].probs.top1
            cls_label = cls_results[0].names[top1_idx]
            cls_conf = cls_results[0].probs.top1conf.item()
        else:
            cls_label, cls_conf = "Unknown", 0.0

        # 2. Run Segmentation
        seg_results = seg_model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)

        # 3. Combine results visually
        annotated_img = seg_results[0].plot()

        label_text = f"Class: {cls_label} ({cls_conf:.2f})"
        cv2.putText(
            annotated_img,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        save_path = output_dir / img_name
        cv2.imwrite(str(save_path), annotated_img)

        logger.info(
            "Processed %s: %s (%.2f), %d masks",
            img_name,
            cls_label,
            cls_conf,
            len(seg_results[0].boxes),
        )

    logger.info("Integrated inference complete. Results saved to %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Brain MRI YOLO11 Inference Test")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["cls", "seg", "integrated"],
        help="Task type: cls, seg, or integrated",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Path to a single image or a directory of images",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to custom weights (optional)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )

    args = parser.parse_args()

    base_path = Path(__file__).resolve().parents[2]
    results_dir = base_path / "results"

    default_cls_source = base_path / "brisc2025" / "classification_task" / "test"
    default_seg_source = base_path / "brisc2025" / "segmentation_task" / "test"

    cls_default = base_path / "models" / "yolo11m-cls-brain-best.pt"
    seg_default = base_path / "models" / "yolo11m-seg-brain-best.pt"

    if args.task == "cls":
        model_path = args.weights or cls_default
        source = args.source or default_cls_source
        run_test(args.task, str(model_path), str(source), str(results_dir), conf=args.conf)
    elif args.task == "seg":
        model_path = args.weights or seg_default
        source = args.source or default_seg_source
        run_test(args.task, str(model_path), str(source), str(results_dir), conf=args.conf)
    elif args.task == "integrated":
        source = args.source or default_seg_source
        run_integrated_test(
            str(cls_default),
            str(seg_default),
            str(source),
            str(results_dir),
            conf=args.conf,
        )


if __name__ == "__main__":
    main()


# # --- 사용 예시 (Usage Examples) ---
# # 1. 기본 테스트 세트 전체 실행
# # python src/testing/test.py --task cls
# # 2. 특정 이미지 파일 하나만 테스트
# # python src/testing/test.py --task integrated --source rawdata/sample.jpg
# # 3. 특정 폴더 내의 모든 이미지 테스트
# # python src/testing/test.py --task seg --source path/to/my_images/
