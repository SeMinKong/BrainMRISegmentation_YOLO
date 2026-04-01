# YOLO11 기반 뇌 MRI 종양 분류 및 분할

**[English Version](./README.en.md)**

**DEMO**
이 데모는 본인이 WEB UI를 추가하여 이 모델을 사용할 수 있는 상황을 시연한 것입니다.

<video src="https://github.com/user-attachments/assets/9994b0b3-187b-4c12-bfd3-170f6bb8dda5" width="600" controls></video>



**YOLO11** 아키텍처를 활용하여 뇌 MRI 이미지에서 종양을 **분류Classification**하고 **분할Segmentation**하는 엔드-투-엔드 의료 영상 파이프라인입니다. 단순한 진단을 넘어, 종양의 유형 파악과 정확한 경계 추출을 하나의 통합된 시스템으로 구현했습니다.

## 주요 특징

- **4종 종양 정밀 분류**: 신경교종(Glioma), 수막종(Meningioma), 뇌하수체종(Pituitary), 정상(Healthy)을 99.4%의 정확도로 판별합니다.
- **인스턴스 분할(Instance Segmentation)**: 픽셀 레벨의 마스크 추출을 통해 종양의 정확한 경계를 가시화합니다 (Mask mAP50 92.7%).
- **통합 추론 파이프라인(Integrated Task)**: 분류 모델과 분할 모델의 결과를 결합하여, 한 장의 이미지에서 진단명과 위치 정보를 동시에 제공합니다.
- **자동 라벨링 시스템**: 의료용 그레이스케일 마스크(PNG)를 YOLO 학습용 다각형(Polygon) 좌표로 자동 변환하는 전처리 엔진을 포함합니다.

## 기술 스택

- **딥러닝**: PyTorch, Ultralytics YOLO11 (m-cls, m-seg)
- **컴퓨터 비전**: OpenCV, NumPy
- **언어**: Python 3.8+
- **하드웨어**: NVIDIA GPU (CUDA) 환경 권장

## 프로젝트 구조

```text
src/
├── training/
│   └── train.py    # 학습 파이프라인 (마스크 → 레이블 자동 변환 포함)
└── testing/
    └── test.py     # 추론 파이프라인 (분류, 분할, 통합 진단)
brisc2025/          # 데이터셋 디렉토리 (Task별 구조화)
models/             # 최적 성능 가중치 파일 (.pt) 저장소
results/            # 학습 지표 및 추론 시각화 결과물
```

## 핵심 기술 구현 내용

### 1. 통합 진단 시각화(Integrated Diagnostic)
분류와 분할을 별개로 수행하는 일반적인 접근법과 달리, 본 프로젝트는 **통합 추론 태스크**를 구현했습니다. 분류 모델로 종양의 타입을 확정하고, 분할 모델로 위치를 찾아 하나의 리포트 이미지로 병합함으로써 의료진의 빠른 의사결정을 돕습니다.

### 2. 커스텀 마스크-다각형 변환 엔진
의료 영상 특유의 노이즈를 제어하기 위해 형태학적 연산(Morphological Closing/Opening)을 적용한 변환 로직을 설계했습니다. 이를 통해 수동 라벨링 없이도 고품질의 YOLO 학습 데이터를 대량으로 생성할 수 있게 되었습니다.

## 빠른 시작

### 설치 방법
```bash
git clone <repository-url>
cd BrainMRISegmentation_YOLO
pip install ultralytics opencv-python torch
```

### 모델 학습
```bash
# 분류 및 분할 모델 순차적 학습 시작
python src/training/train.py --task both --epochs 50 --batch 32
```

### 추론 및 테스트
```bash
# 통합 진단(분류 + 분할) 결과 확인
python src/testing/test.py --task integrated --source /path/to/mri_image.jpg
```

## 성능 지표
- **분류(Classification)**: Top-1 Accuracy 99.4%
- **분할(Segmentation)**: Mask mAP50 92.7%

>  **더 자세한 정보가 필요하신가요?**
> 상세한 CLI 인자 설정, 마스크-다각형 변환 과정 및 커스텀 클래스 확장법 등은 [상세 매뉴얼(DETAILS.md)](./DETAILS.md)에서 확인하실 수 있습니다.

---
BRISC 2025 의료 영상 AI 챌린지를 위해 개발되었습니다.
