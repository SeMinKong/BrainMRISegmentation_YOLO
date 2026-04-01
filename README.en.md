# Brain MRI Tumor Segmentation & Classification (YOLO11) 🧠

**[한국어 버전](./README.md)**

An end-to-end medical imaging pipeline built with **YOLO11** to perform both **Classification** and **Instance Segmentation** on brain MRI scans. This project provides a unified diagnostic tool that not only identifies the tumor type but also precisely maps its boundaries.

## 🚀 Key Features

- **Multi-class Classification**: Identifies Glioma, Meningioma, Pituitary tumor, and Healthy scans with 99.4% accuracy.
- **High-precision Segmentation**: Pixel-level mask extraction for tumor boundaries (92.7% Mask mAP50).
- **Unified Inference Pipeline**: A single command execution that combines classification and segmentation results into one visualized report.
- **Automated Labeling**: Custom script to convert grayscale mask PNGs into YOLO-compatible polygon labels.

## 🛠 Tech Stack

- **Deep Learning**: PyTorch, Ultralytics YOLO11 (m-cls, m-seg)
- **Computer Vision**: OpenCV, NumPy
- **Language**: Python 3.8+
- **Hardware**: CUDA-enabled NVIDIA GPU recommended

## 🏗 Project Structure

```text
src/
├── training/
│   └── train.py    # Training pipeline (Auto-conversion of masks & labels)
└── testing/
    └── test.py     # Inference (Classification, Segmentation, Integrated)
brisc2025/          # Dataset directory (Categorized by task)
models/             # Best performing weights (.pt)
results/            # Metrics & visualization output
```

## 🧠 Technical Highlights

### 1. Integrated Diagnostic Visualization
Unlike standard pipelines that separate classification and segmentation, this project implements an **Integrated Task**. It runs the classification model to determine the tumor type and the segmentation model to find the location, overlaying both results on a single diagnostic image.

### 2. Custom Mask-to-Polygon Engine
To streamline the YOLO training process, I developed a preprocessing engine that automatically converts medical grayscale masks into precise polygon coordinates, handling noise reduction and morphological closing to ensure high-quality training data.

## 🏁 Quick Start

### Installation
```bash
git clone <repository-url>
cd BrainMRISegmentation_YOLO
pip install ultralytics opencv-python torch
```

### Training
```bash
# Train both classification and segmentation models
python src/training/train.py --task both --epochs 50 --batch 32
```

### Inference
```bash
# Run integrated diagnostic (Classification + Segmentation)
python src/testing/test.py --task integrated --source /path/to/mri_image.jpg
```

## 📊 Performance
- **Classification**: Top-1 Accuracy 99.4%
- **Segmentation**: Mask mAP50 92.7%

> 💡 **Need more details?**
> For advanced CLI configurations, morphological filtering logic, and custom class extensions, please refer to the [Detailed Manual (DETAILS.en.md)](./DETAILS.en.md).

---
Developed for BRISC 2025 - Medical Image AI Challenge.
