# Brain MRI Tumor Analysis using YOLO11 (Classification & Segmentation)

This project is a multi-task learning pipeline that utilizes YOLO11, an object detection model, to classify tumor types and extract their locations and shapes (segmentation) from brain MRI scans.

## Project Overview

Accurate tumor identification and boundary extraction are crucial in medical data analysis. Based on the BRISC 2025 dataset, I implemented the following features:

- 4-Class Brain Tumor Classification: Glioma, Meningioma, Pituitary, and No Tumor classification.
- Instance Segmentation: Extracts pixel-level boundaries to provide size and shape information.
- Integrated Inference Pipeline: Combines results from both classification and segmentation models for intuitive visualization.

## Tech Stack & Environment

- Deep Learning Framework: PyTorch, Ultralytics (YOLO11)
- Computer Vision: OpenCV, NumPy
- Model Architectures:
  - YOLO11m-cls: Dedicated model for brain tumor classification
  - YOLO11m-seg: Dedicated model for instance segmentation
- Hardware: NVIDIA GPU (RTX 5060 Ti 16GB)

## Directory Structure

```text
BrainMRI/
├── src/
│   ├── training/           # Training pipeline (YOLO11 cls/seg)
│   └── testing/            # Integrated inference and validation
├── brisc2025/              # Dataset for Brain MRI (Raw & Masks)
│   ├── classification_task/
│   └── segmentation_task/
├── models/                 # Storage for best-performing weights
├── results/                # Performance metrics and visualization
│   ├── classification/     # Classification metrics (CSV, Confusion Matrix)
│   ├── segmentation/       # Segmentation metrics (mAP, F1-Curve)
│   └── test_results/       # Combined inference visualization (.jpg)
└── README.md
```

## Performance Metrics

These results were measured based on 50 epochs of training.

### 1. Classification
- Top-1 Accuracy: 99.4%
- Validation Loss: 0.030
- Achieved stable and high classification performance across all 4 classes.

### 2. Segmentation
- Mask mAP50: 92.7%
- Box mAP50: 91.7%
- Mask Recall: 89.0%
- Generates masks relatively accurately even in images with unclear tumor boundaries.

## Code Structure & Workflow

### 1. Data Preprocessing
- mask_to_polygons: Automatically converts grayscale mask images into the YOLO segmentation format (polygon coordinates).
- Noise Filtering: Applies morphological operations (closing, opening) to remove small noise from masks and smooth out boundaries.

### 2. Training Pipeline (src/training/train.py)
- Built a CLI interface to train classification and segmentation tasks either independently or simultaneously.
- Automatically saves the best-performing weights in the models directory after training.

### 3. Integrated Testing & Visualization (src/testing/test.py)
- Supports an integrated mode that overlays the predicted class from the classification model and the mask area from the segmentation model onto a single image.

## How to Run

### Environment Setup
```bash
pip install ultralytics opencv-python numpy pyyaml
```

### Model Training
```bash
# Train both classification and segmentation simultaneously
python src/training/train.py --task both

# Train a specific task (e.g., segmentation)
python src/training/train.py --task seg

### Inference
```bash
# Integrated inference (visualizes merged classification + segmentation results)
python src/testing/test.py --task integrated --conf 0.3
```

## Visualization Examples

You can check the prediction result images in the results/test_results directory of the project. 

- Classification result: Class name and confidence score displayed in the top-left corner.
- Segmentation result: Semi-transparent mask and bounding box overlaid on the tumor area.

## Project Highlights

- End-to-End Pipeline: Automated the entire process from data preprocessing to training and integrated inference.
- Latest Model Application: Tuned and optimized YOLO11 for the medical domain.
- Integrated Visualization: Provides intuitive visual outputs beyond simple metric listings.
- Validated Performance: Confirmed algorithm effectiveness by recording 99% classification accuracy and over 92% mAP.