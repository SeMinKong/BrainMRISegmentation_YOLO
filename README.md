# Brain MRI Tumor Segmentation & Classification using YOLO11

> **언어 / Language:** [🇰🇷 한국어](#한국어-문서) | [🇬🇧 English](#english-documentation)

---

<a id="한국어-문서"></a>

# 한국어 문서

## 프로젝트 개요

**Brain MRI Segmentation & Classification**은 **YOLO11** 기반의 엔드-투-엔드 파이프라인으로, 뇌 MRI 스캔 이미지에서 종양을 **분류(Classification)**하고 **분할(Segmentation)**하는 딥러닝 프로젝트입니다. 이 프로젝트는 의료 영상 분석 작업을 자동화하여 의사의 진단을 보조합니다.

### 핵심 기능

- **4가지 종양 클래스 분류**: Glioma (신경교종), Meningioma (수막종), Pituitary (뇌하수체종), No Tumor (정상)
- **인스턴스 분할(Instance Segmentation)**: 종양의 정확한 경계와 픽셀 레벨의 마스크 추출
- **통합 추론 파이프라인**: 분류 및 분할 모델 결과를 시각적으로 결합하여 종합적인 진단 지원
- **높은 정확도**: 분류 Top-1 Accuracy 99.4%, 분할 Mask mAP50 92.7%

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| **딥러닝 프레임워크** | PyTorch, Ultralytics YOLO11 |
| **컴퓨터 비전** | OpenCV, NumPy |
| **모델** | YOLO11m-cls (분류), YOLO11m-seg (분할) |
| **프로그래밍 언어** | Python 3.8+ |
| **하드웨어** | NVIDIA GPU (CUDA 권장), CPU 지원 |

---

## 프로젝트 구조

```
BrainMRISegmentation_YOLO/
├── src/
│   ├── training/
│   │   └── train.py                    # 분류 및 분할 모델 학습 파이프라인
│   └── testing/
│       └── test.py                     # 추론 및 통합 테스트 파이프라인
├── brisc2025/
│   ├── classification_task/
│   │   ├── train/                      # 분류 학습 데이터
│   │   │   ├── glioma/                 # 신경교종 이미지들
│   │   │   ├── meningioma/             # 수막종 이미지들
│   │   │   ├── pituitary/              # 뇌하수체종 이미지들
│   │   │   └── no_tumor/               # 정상 이미지들
│   │   └── test/                       # 분류 테스트 데이터 (동일 구조)
│   └── segmentation_task/
│       ├── train/                      # 분할 학습 데이터
│       │   ├── images/                 # 원본 MRI 이미지 (.jpg, .png)
│       │   ├── masks/                  # 그레이스케일 마스크 PNG (자동 변환됨)
│       │   └── labels/                 # YOLO 다각형 레이블 .txt (자동 생성됨)
│       └── test/                       # 분할 테스트 데이터 (동일 구조)
├── models/                             # 학습된 최고 성능 모델 가중치 저장
├── results/
│   ├── classification/                 # 분류 모델 학습 지표 및 가중치
│   ├── segmentation/                   # 분할 모델 학습 지표 및 가중치
│   ├── cls_inference/                  # 분류 추론 결과
│   ├── seg_inference/                  # 분할 추론 결과
│   └── test_results/                   # 통합 추론 시각화 결과
└── README.md                           # 이 파일
```

---

## 성능 지표 (50 에포크 기준)

### 분류 (Classification)

| 지표 | 값 |
|------|-----|
| Top-1 Accuracy | 99.4% |
| Validation Loss | 0.030 |
| 클래스별 성능 | 모든 4가지 클래스에서 안정적이고 높은 성능 |

**특징**: 각 종양 타입을 높은 신뢰도로 구분하며, False Positive/Negative 최소화

### 분할 (Segmentation)

| 지표 | 값 |
|------|-----|
| Mask mAP50 | 92.7% |
| Box mAP50 | 91.7% |
| Mask Recall | 89.0% |
| 특징 | 불명확한 경계의 이미지에서도 정확한 마스크 생성 |

**특징**: 종양의 정확한 경계를 포착하여 의료 영상 분석에 필요한 높은 정밀도 달성

---

# 사용자 가이드

## 1. 환경 설정 및 설치

### 요구사항

- **Python**: 3.8 이상
- **운영체제**: Linux, macOS, Windows
- **메모리**: 최소 8GB RAM (GPU 사용 시 권장)
- **저장공간**: 최소 10GB (모델 및 데이터셋용)

### 1-1. 필수 패키지 설치

#### 옵션 1: 기본 설치 (CPU)

```bash
# 기본 의존성 설치
pip install --upgrade pip
pip install ultralytics opencv-python numpy pyyaml
```

#### 옵션 2: GPU 지원 설치 (권장)

```bash
# PyTorch CUDA 지원 버전 설치 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 그 외 필수 패키지
pip install ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0 pyyaml>=6.0
```

#### 옵션 3: 가상 환경 활용 (권장)

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (Linux/Mac)
source venv/bin/activate

# 또는 Windows
venv\Scripts\activate

# 패키지 설치
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy pyyaml
```

### 1-2. GPU 설정 확인

```bash
# NVIDIA GPU 사용 가능 여부 확인
nvidia-smi

# 또는 Python에서 확인
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 2. 데이터셋 준비

### 2-1. 분류(Classification) 작업용 데이터셋 구조

분류 작업에서는 **클래스별 폴더 구조**를 사용합니다. 각 폴더명이 클래스 라벨이 됩니다.

```
classification_task/
├── train/
│   ├── glioma/           # 신경교종 이미지들 (예: img1.jpg, img2.png, ...)
│   ├── meningioma/       # 수막종 이미지들
│   ├── pituitary/        # 뇌하수체종 이미지들
│   └── no_tumor/         # 정상 이미지들
└── test/
    ├── glioma/           # 테스트 이미지 (동일 클래스 구조)
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

**요구사항**:
- 이미지 형식: JPG, PNG, JPEG
- 각 클래스당 최소 20-30개 이미지 권장 (학습 안정성)
- 이미지 크기: 자동 조정됨 (권장: 320x320 이상)

### 2-2. 분할(Segmentation) 작업용 데이터셋 구조

분할 작업에서는 **이미지 + 마스크 + 라벨** 구조를 사용합니다.

```
segmentation_task/
├── train/
│   ├── images/           # 원본 MRI 이미지들 (JPG, PNG)
│   │   ├── img_001.jpg
│   │   ├── img_002.png
│   │   └── ...
│   ├── masks/            # 그레이스케일 마스크 PNG (자동으로 .txt로 변환됨)
│   │   ├── img_001_0_gl.png      # Glioma 마스크
│   │   ├── img_002_0_me.png      # Meningioma 마스크
│   │   └── ...
│   └── labels/           # YOLO 다각형 레이블 (자동 생성, 수동 편집 가능)
│       ├── img_001_0_gl.txt
│       ├── img_002_0_me.txt
│       └── ...
└── test/
    ├── images/           # 테스트 이미지들
    ├── masks/            # 테스트 마스크들
    └── labels/           # 테스트 레이블들
```

### 2-3. 마스크 파일 명명 규칙

마스크 PNG 파일명은 다음 형식을 따라야 합니다:

```
{image_id}_{category}_{index}_{class_key}.png
```

**각 부분 설명**:
- `image_id`: 이미지 고유 ID (예: `img_001`, `sample_42`)
- `category`: 카테고리 번호 (예: `0`, `1`)
- `index`: 순서 인덱스 (예: `0`, `1`)
- `class_key`: 종양 타입 약자

**class_key 매핑**:
| class_key | 종양 타입 | 설명 |
|-----------|---------|------|
| `gl` | glioma | 신경교종 (YOLO ID: 0) |
| `me` | meningioma | 수막종 (YOLO ID: 1) |
| `no` | no_tumor | 정상 (YOLO ID: 2) |
| `pi` | pituitary | 뇌하수체종 (YOLO ID: 3) |

**마스크 파일명 예시**:
```
img_001_0_gl.png   → Glioma 마스크
img_002_0_me.png   → Meningioma 마스크
img_003_0_no.png   → No tumor 마스크
img_004_0_pi.png   → Pituitary 마스크
```

**마스크 특징**:
- **파일 형식**: PNG (그레이스케일)
- **픽셀값**: 0 (배경), 255 (종양 영역)
- **크기**: 원본 이미지와 동일 크기
- **자동 변환**: 학습 시 자동으로 YOLO 다각형 형식(.txt)으로 변환됨

---

## 3. 모델 학습

### 3-1. 분류 모델만 학습

```bash
cd /path/to/BrainMRISegmentation_YOLO
python src/training/train.py --task cls --epochs 50 --batch 32
```

**동작**:
1. `brisc2025/classification_task/` 데이터 로드
2. YOLO11m-cls 모델 초기화
3. 50 에포크동안 학습
4. 최고 성능 모델을 `models/yolo11m-cls-brain-best.pt`로 저장

### 3-2. 분할 모델만 학습

```bash
python src/training/train.py --task seg --epochs 50 --batch 32
```

**동작**:
1. `brisc2025/segmentation_task/` 데이터 로드
2. 마스크 PNG → YOLO 다각형 레이블 자동 변환
3. `data.yaml` 자동 생성
4. YOLO11m-seg 모델 초기화
5. 50 에포크동안 학습
6. 최고 성능 모델을 `models/yolo11m-seg-brain-best.pt`로 저장

### 3-3. 분류 및 분할 동시 학습

```bash
python src/training/train.py --task both --epochs 50 --batch 32
```

**동작**: 분류와 분할 모두 순차적으로 수행

---

## 3-4. 학습 CLI 인수 상세 설명

| 인수 | 타입 | 기본값 | 범위 | 설명 |
|------|------|--------|------|------|
| `--task` | str | `both` | `cls`, `seg`, `both` | 수행할 작업: 분류(cls), 분할(seg), 또는 둘 다(both) |
| `--epochs` | int | 50 | 1 ~ 1000 | 학습 에포크 수 (전체 데이터셋을 몇 번 반복할지) |
| `--batch` | int | 32 | 1 ~ 512 | 배치 크기 (한 번에 처리할 이미지 개수). GPU 메모리가 부족하면 감소 |
| `--imgsz` | int | 320 | 32 ~ 1024 | 입력 이미지 크기 (픽셀). 더 크면 정확도 향상, 더 오래 걸림 |
| `--device` | int | 0 | 0, 1, 2, ... | GPU 장치 인덱스. 0=첫 번째 GPU, 1=두 번째 GPU, ... |

---

## 3-5. 학습 예시 모음

### 기본 학습 (권장 설정)
```bash
# 분류, 50 에포크, 배치 32, 이미지 320x320, GPU 0
python src/training/train.py --task cls --epochs 50 --batch 32 --device 0
```

### 고정확도 학습 (느리지만 더 정확)
```bash
# 이미지 크기 640, 100 에포크, 작은 배치
python src/training/train.py --task seg --epochs 100 --batch 16 --imgsz 640 --device 0
```

### 빠른 학습 (탐색 목적)
```bash
# 이미지 크기 256, 20 에포크, 큰 배치
python src/training/train.py --task both --epochs 20 --batch 64 --imgsz 256 --device 0
```

### 제한된 GPU 메모리 환경
```bash
# 배치 크기를 16으로 감소
python src/training/train.py --task both --epochs 50 --batch 16 --device 0
```

### 두 번째 GPU 사용
```bash
# GPU 1 지정
python src/training/train.py --task both --epochs 50 --batch 32 --device 1
```

### CPU 학습 (GPU 없을 때)
```bash
# device를 cpu로 지정 (매우 느림, 테스트용)
python src/training/train.py --task cls --epochs 5 --batch 8 --device cpu
```

---

## 4. 추론 및 테스트

### 4-1. 분류만 수행

```bash
python src/testing/test.py --task cls
```

**동작**:
- 기본 분류 테스트 데이터 사용: `brisc2025/classification_task/test/`
- 저장된 분류 모델(`models/yolo11m-cls-brain-best.pt`) 로드
- 모든 이미지에 대해 추론 수행
- 결과 저장: `results/cls_inference/`

**출력 예시**:
```
Image: img_001.jpg | Class: glioma (0.98)
Image: img_002.jpg | Class: no_tumor (0.97)
Image: img_003.jpg | Class: meningioma (0.95)
```

### 4-2. 분할만 수행

```bash
python src/testing/test.py --task seg
```

**동작**:
- 기본 분할 테스트 데이터 사용: `brisc2025/segmentation_task/test/`
- 저장된 분할 모델(`models/yolo11m-seg-brain-best.pt`) 로드
- 모든 이미지에 대해 추론 수행
- 마스크 및 바운딩 박스 시각화
- 결과 저장: `results/seg_inference/`

**출력 예시**:
```
Image: img_001.jpg | Detections: 2, Masks: 2
Image: img_002.jpg | Detections: 1, Masks: 1
```

### 4-3. 통합 추론 (분류 + 분할)

```bash
python src/testing/test.py --task integrated
```

**동작**:
1. 각 이미지에 대해 분류 모델 실행 → 종양 타입 결정
2. 같은 이미지에 대해 분할 모델 실행 → 종양 위치 및 마스크 추출
3. 분류 결과(클래스명, 신뢰도)와 분할 결과(마스크, 바운딩 박스)를 하나의 이미지로 결합
4. 최종 시각화 결과 저장: `results/test_results/`

**출력 예시**:
```
Processed img_001.jpg: glioma (0.98), 2 masks
Processed img_002.jpg: meningioma (0.95), 1 masks
```

---

## 4-4. 추론 CLI 인수 상세 설명

| 인수 | 타입 | 기본값 | 필수 | 설명 |
|------|------|--------|------|------|
| `--task` | str | - | **필수** | 추론 작업: `cls` (분류), `seg` (분할), `integrated` (통합) |
| `--source` | str | 기본 테스트 경로 | 선택 | 단일 이미지 파일 경로 또는 폴더 경로 |
| `--weights` | str | 모델 디렉토리의 기본값 | 선택 | 커스텀 모델 가중치 파일 경로 (.pt 파일) |
| `--conf` | float | 0.25 | 선택 | 신뢰도 임계값 (0.0 ~ 1.0). 낮을수록 더 많은 감지 |

---

## 4-5. 추론 예시 모음

### 기본 분류 테스트 (기본 테스트셋)
```bash
python src/testing/test.py --task cls
```

### 특정 폴더의 모든 이미지 분할 추론
```bash
python src/testing/test.py --task seg --source /path/to/images/
```

### 단일 이미지로 통합 추론
```bash
python src/testing/test.py --task integrated --source /path/to/image.jpg
```

### 높은 신뢰도 임계값 (거짓 양성 최소화)
```bash
python src/testing/test.py --task cls --conf 0.7
```

### 낮은 신뢰도 임계값 (민감도 최대화)
```bash
python src/testing/test.py --task seg --conf 0.15
```

### 커스텀 모델 가중치 사용
```bash
python src/testing/test.py --task integrated \
  --weights /custom/path/yolo11m-seg-brain-best.pt \
  --source /custom/images/
```

### 특정 경로의 이미지로 높은 신뢰도 설정
```bash
python src/testing/test.py --task cls \
  --source /data/classification_test/ \
  --conf 0.75
```

---

## 5. 결과 해석

### 5-1. 분류 결과 읽기

#### 로그 출력
```
Image: patient_001.jpg | Class: glioma (0.98)
```

**의미**:
- **Image**: 입력 이미지 파일명
- **Class**: 예측된 종양 타입
- **(0.98)**: 신뢰도 점수 (0.0 ~ 1.0)
  - 1.0 = 100% 확신
  - 0.5 = 50% 확신

#### 시각화 해석
- 결과 이미지 좌상단에 "Class: glioma (0.98)" 표시
- 색상: 초록색 텍스트 (신뢰도 높음)

### 5-2. 분할 결과 읽기

#### 로그 출력
```
Image: patient_001.jpg | Detections: 2, Masks: 2
```

**의미**:
- **Detections**: 감지된 종양 개수 (바운딩 박스)
- **Masks**: 생성된 마스크 개수 (픽셀 레벨 분할)

#### 시각화 해석
- **반투명 마스크**: 종양 영역을 색상으로 표시
- **바운딩 박스**: 종양을 감싸는 사각형 (신뢰도 표시)
- **클래스 라벨**: 각 감지에 대한 종양 타입 표시

### 5-3. 통합 추론 결과 읽기

#### 출력 예시
```
Processed patient_001.jpg: glioma (0.98), 2 masks
```

**의미**:
- 분류: glioma (신뢰도 0.98)
- 분할: 2개 마스크 감지
- 최종 이미지: 분류 결과 텍스트 + 분할 마스크 + 바운딩 박스

---

## 5-4. 신뢰도 임계값 조정 가이드

### 높은 임계값 (0.7 이상)
```bash
python src/testing/test.py --task cls --conf 0.8
```
**특징**:
- 매우 확신하는 결과만 표시
- False Positive (거짓 양성) 최소화
- 민감도(Recall) 감소
- **용도**: 보수적 진단, 임상 의사 검토용

### 중간 임계값 (0.3 ~ 0.7)
```bash
python src/testing/test.py --task cls --conf 0.5
```
**특징**:
- 균형잡힌 정확도와 민감도
- 대부분의 경우에 적합
- **용도**: 일반적인 진단

### 낮은 임계값 (0.1 이하)
```bash
python src/testing/test.py --task cls --conf 0.1
```
**특징**:
- 약한 신호도 포함
- False Negative (거짓 음성) 최소화
- 민감도(Recall) 최대화
- **용도**: 초기 선별 검사, 놓친 케이스 감지

---

# 개발자 가이드

## 1. 선행 조건

### 1-1. 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|-----|------|
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| CUDA | 11.8 | 12.1+ |
| cuDNN | 8.6 | 8.7+ |
| RAM | 8GB | 16GB+ |
| GPU 메모리 | 4GB | 8GB+ |

### 1-2. 설치 검증

```bash
# Python 버전 확인
python --version

# PyTorch CUDA 지원 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# YOLO 설치 확인
python -c "from ultralytics import YOLO; print(YOLO.__version__)"
```

---

## 2. 상세 설치 안내

### 2-1. 전체 설치 과정

```bash
# 1. 리포지토리 클론
git clone <repository-url>
cd BrainMRISegmentation_YOLO

# 2. 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 3. pip 업그레이드
pip install --upgrade pip

# 4. GPU 지원 PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 그 외 의존성 설치
pip install ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0 pyyaml>=6.0

# 6. 설치 검증
python -c "from ultralytics import YOLO; print('YOLO ready')"
```

### 2-2. 트러블슈팅

#### CUDA/cuDNN 설정
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# CUDA 경로 설정 (필요시)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### PyTorch CUDA 호환성
```bash
# 올바른 CUDA 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# 또는
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

---

## 3. 프로젝트 구조 상세 설명

### 3-1. 디렉토리 및 파일 역할

#### `src/training/train.py`

**역할**: 분류 및 분할 모델 학습

**주요 함수**:

| 함수 | 입력 | 출력 | 설명 |
|------|------|------|------|
| `train_classification()` | data_dir, results_dir, model_out_dir, epochs, batch, imgsz, device | 학습된 모델 저장 | YOLO11m-cls 모델 학습 |
| `train_segmentation()` | data_dir, results_dir, model_out_dir, epochs, batch, imgsz, device | 학습된 모델 저장 | YOLO11m-seg 모델 학습 |
| `prepare_segmentation_labels()` | base_dir | - | PNG 마스크 → YOLO 레이블 변환 |
| `create_seg_yaml()` | data_root | yaml_path | YOLO 데이터 설정 파일 생성 |
| `mask_to_polygons()` | mask_path, class_id | polygon_string | 마스크 → 다각형 좌표 변환 |
| `_prepare_yolo_val_dir()` | data_path | - | 검증 디렉토리 준비 (심링크/복사) |
| `_save_best_model()` | results, model_out_dir, filename | - | 최고 성능 모델 저장 |

#### `src/testing/test.py`

**역할**: 학습된 모델로 추론 및 평가

**주요 함수**:

| 함수 | 입력 | 출력 | 설명 |
|------|------|------|------|
| `run_test()` | task, model_path, source, results_dir, imgsz, conf | 시각화 이미지 저장 | 단일 작업(cls/seg) 추론 |
| `run_integrated_test()` | cls_weights, seg_weights, source, results_dir, imgsz, conf | 결합된 시각화 저장 | 분류 + 분할 통합 추론 |

#### `brisc2025/` 디렉토리

- **classification_task/**: 클래스별 폴더 구조 (클래스명이 라벨)
- **segmentation_task/**: images, masks, labels 구조 (다각형 좌표가 라벨)

#### `models/` 디렉토리

학습된 모델 가중치 저장:
- `yolo11m-cls-brain-best.pt`: 분류 모델 (최고 성능)
- `yolo11m-seg-brain-best.pt`: 분할 모델 (최고 성능)

#### `results/` 디렉토리

학습 및 추론 결과:
- `classification/`: 분류 학습 지표 (accuracy, loss curve 등)
- `segmentation/`: 분할 학습 지표 (mAP, recall curve 등)
- `cls_inference/`: 분류 추론 결과 이미지
- `seg_inference/`: 분할 추론 결과 이미지
- `test_results/`: 통합 추론 시각화 결과

---

## 4. CLASS_MAP 설명 및 확장

### 4-1. 기본 CLASS_MAP

`src/training/train.py`에 정의된 CLASS_MAP:

```python
CLASS_MAP: dict[str, int] = {
    "glioma": 0,           # 신경교종 (ID: 0)
    "meningioma": 1,       # 수막종 (ID: 1)
    "no_tumor": 2,         # 정상 (ID: 2)
    "pituitary": 3,        # 뇌하수체종 (ID: 3)
    "gl": 0,               # 약자 (full name과 동일 ID 매핑)
    "me": 1,
    "no": 2,
    "pi": 3,
}
```

**용도**:
- 마스크 파일명에서 `class_key` 추출 후 YOLO 클래스 ID로 변환
- 예: `image_001_0_gl.png` → class_key="gl" → ID=0

### 4-2. 새로운 종양 클래스 추가 방법

새로운 종양 클래스(예: `adenoma`)를 추가하려면 다음 단계를 따르세요:

#### 단계 1: CLASS_MAP 업데이트

`src/training/train.py` 수정:

```python
CLASS_MAP: dict[str, int] = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
    "adenoma": 4,          # 새 클래스 추가
    "gl": 0,
    "me": 1,
    "no": 2,
    "pi": 3,
    "ad": 4,               # 약자도 추가
}
```

#### 단계 2: 데이터셋 구조 업데이트

분류 작업의 폴더 구조:
```
classification_task/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   ├── no_tumor/
│   └── adenoma/          # 새 폴더 추가
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    ├── no_tumor/
    └── adenoma/          # 새 폴더 추가
```

분할 작업의 마스크 파일명:
```
image_001_0_ad.png    # adenoma 마스크 (class_key="ad")
```

#### 단계 3: YAML 파일 업데이트

`src/training/train.py`의 `create_seg_yaml()` 함수 수정:

```python
def create_seg_yaml(data_root: str) -> str:
    data_root_path = Path(data_root).absolute()
    yaml_content: dict = {
        "path": str(data_root_path),
        "train": "train/images",
        "val": "test/images",
        "names": {
            0: "glioma",
            1: "meningioma",
            2: "no_tumor",
            3: "pituitary",
            4: "adenoma",         # 새 클래스 추가
        },
    }
    yaml_path = data_root_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return str(yaml_path)
```

#### 단계 4: 검증

마스크 파일명이 올바른지 확인:
```
# 올바른 형식
image_001_0_ad.png  # adenoma, class_key="ad"

# 잘못된 형식 (작동 안 함)
image_001_0_adenoma.png  # CLASS_MAP에 "adenoma" 키가 있지만, class_key는 파일명의 4번째 부분이므로 class_key="adenoma"가 되어 AD 값만 매핑됨
```

---

## 5. 핵심 함수 상세 설명

### 5-1. mask_to_polygons(mask_path, class_id) → str

**목적**: 그레이스케일 마스크 PNG를 YOLO 다각형 형식으로 변환

**입력**:
- `mask_path` (Path): PNG 마스크 파일 경로
- `class_id` (int): YOLO 클래스 ID (0-3)

**출력**:
- str: YOLO 다각형 형식의 레이블 문자열

**동작 원리**:

```
1. PNG 마스크 읽음 (cv2.IMREAD_GRAYSCALE)
   → 그레이스케일 이미지 (0-255)

2. 이진화(Binary threshold) 적용
   → 1 이상의 픽셀을 255로, 0을 0으로 변환

3. 형태학적 연산(Morphological operations)으로 잡음 제거
   ├─ MORPH_CLOSE: 작은 구멍 채우기 (5x5 커널)
   └─ MORPH_OPEN: 작은 돌기 제거 (5x5 커널)

4. 컨투어(Contour) 추출
   → 종양 영역의 윤곽선 찾음

5. 면적 임계값(h * w * 0.001) 이상의 컨투어만 선택
   → 노이즈 제거

6. 정규화된 좌표(0~1)로 다각형 생성
   → YOLO 형식 (x1, y1, x2, y2, ...)
```

**출력 형식**:
```
{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} ...
```

**예시**:
```
0 0.234567 0.345678 0.456789 0.567890 0.123456 0.654321 ...
```

### 5-2. _prepare_yolo_val_dir(data_path) → None

**목적**: YOLO 학습에 필요한 검증 디렉토리 준비

**입력**:
- `data_path` (Path): 데이터 디렉토리 경로

**동작 원리**:

```
1. val/ 디렉토리 존재 확인
   └─ 있으면: 아무것도 하지 않음

2. val/ 없지만 test/ 있으면:
   ├─ 시도1: test/에서 val/로 심볼릭 링크 생성
   │   (Linux/Mac에서 작동)
   │
   └─ 시도2: 심링크 실패 시 test/를 val/로 복사
       (Windows에서 작동)

3. YOLO의 데이터 로더가 val/ 경로를 찾을 수 있게 함
```

**중요성**: YOLO는 학습 중 검증 데이터를 `val/` 경로에서 찾습니다. 이 함수가 이를 자동으로 설정합니다.

**예시 코드**:
```python
val_path = data_path / "val"
test_path = data_path / "test"

if not val_path.exists() and test_path.exists():
    logger.info("Creating symlink from %s to %s...", test_path, val_path)
    try:
        # 심링크 시도 (Linux/Mac)
        os.symlink(test_path.name, val_path)
    except OSError:
        # 심링크 실패 시 폴더 복사 (Windows)
        shutil.copytree(test_path, val_path, dirs_exist_ok=True)
```

### 5-3. _save_best_model(results, model_out_dir, filename) → None

**목적**: 학습 후 최고 성능 모델 저장

**입력**:
- `results` (Results): YOLO 학습 결과 객체
- `model_out_dir` (str): 모델 출력 디렉토리
- `filename` (str): 저장할 파일명

**동작 원리**:

```
1. YOLO 학습 결과에서 최고 성능 가중치 찾음
   → results.save_dir/weights/best.pt

2. 지정된 디렉토리로 복사
   → model_out_dir/filename

3. 로그 기록
```

**예시**:
```python
# 호출
_save_best_model(results, "models/", "yolo11m-cls-brain-best.pt")

# 결과
# models/yolo11m-cls-brain-best.pt 생성
```

---

## 6. 학습 YAML 설정 상세 설명

분할 모델 학습 시 자동으로 생성되는 `data.yaml` 파일의 구조:

```yaml
path: /absolute/path/to/data
train: train/images
val: test/images
names:
  0: glioma
  1: meningioma
  2: no_tumor
  3: pituitary
```

### 6-1. 각 필드 설명

| 필드 | 값 | 설명 |
|------|-----|------|
| `path` | 절대경로 | 데이터 루트의 절대 경로 (YOLO의 기준점) |
| `train` | `train/images` | 학습 이미지 디렉토리 (path로부터 상대경로) |
| `val` | `test/images` | 검증 이미지 디렉토리 (path로부터 상대경로) |
| `names` | Dict[int, str] | 클래스 ID → 클래스명 매핑 |

### 6-2. 수동 YAML 편집

커스텀 설정이 필요한 경우:

```yaml
path: /home/user/data/segmentation_task
train: train/images
val: test/images
names:
  0: glioma
  1: meningioma
  2: no_tumor
  3: pituitary
  
# 선택사항: 학습 설정
train_params:
  epochs: 100
  imgsz: 640
  batch: 16
```

---

## 7. --device 파라미터 상세 가이드

`--device` 파라미터로 어떤 GPU를 사용할지 지정합니다.

### 7-1. 사용 가능한 GPU 확인

```bash
# NVIDIA GPU 확인
nvidia-smi

# 출력 예시:
# GPU 0: NVIDIA A100 (40GB)
# GPU 1: NVIDIA A100 (40GB)
# GPU 2: NVIDIA RTX 4090 (24GB)
```

### 7-2. 단일 GPU 사용

```bash
# 첫 번째 GPU (GPU 0) 사용 [기본값]
python src/training/train.py --task both --device 0

# 두 번째 GPU (GPU 1) 사용
python src/training/train.py --task both --device 1

# 세 번째 GPU (GPU 2) 사용
python src/training/train.py --task both --device 2
```

### 7-3. 여러 GPU 사용 (다중 GPU 학습)

현재 구현은 단일 GPU를 지원합니다. 다중 GPU 학습을 추가하려면:

```python
# src/training/train.py 수정
device = [0, 1, 2]  # GPU 0, 1, 2 동시 사용

# train_classification() 함수 내
results = model.train(
    ...
    device=device,  # 리스트로 지정
    ...
)
```

### 7-4. CPU 사용 (GPU 없을 때)

```bash
# CPU 사용 (매우 느림, 테스트/디버깅용)
python src/training/train.py --task cls --epochs 5 --batch 8 --device cpu
```

### 7-5. GPU 메모리 최적화

GPU 메모리 부족 시:

```bash
# 배치 크기 감소 (메모리 사용량 감소)
python src/training/train.py --task both --batch 8 --device 0

# 이미지 크기 감소 (메모리 사용량 감소)
python src/training/train.py --task both --imgsz 256 --device 0

# 두 가지 모두 감소
python src/training/train.py --task both --batch 8 --imgsz 256 --device 0
```

---

## 8. 커스텀 모델 가중치 사용

학습된 모델을 추론에 사용:

```bash
# 커스텀 분류 모델
python src/testing/test.py --task cls \
  --weights /path/to/custom-cls-model.pt

# 커스텀 분할 모델
python src/testing/test.py --task seg \
  --weights /path/to/custom-seg-model.pt

# 통합 추론 (커스텀 모델 두 개 필요)
# 주의: --weights는 한 개만 지정 가능
# 두 모델 모두 기본 경로에 있어야 함
python src/testing/test.py --task integrated
```

---

## 9. 로깅 및 디버깅

### 9-1. 로그 레벨 설정

`src/training/train.py` 또는 `src/testing/test.py`의 초기 부분:

```python
logging.basicConfig(
    level=logging.INFO,  # 가능한 값: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

### 9-2. 디버그 정보 출력

```python
import logging
logger = logging.getLogger(__name__)

# 정보 레벨 로그
logger.info("Starting training...")

# 경고 레벨 로그
logger.warning("Low GPU memory detected")

# 에러 레벨 로그
logger.error("Model file not found")
```

---

# English Documentation

<a id="english-documentation"></a>

## Project Overview

**Brain MRI Segmentation & Classification** is an end-to-end pipeline based on **YOLO11** that **classifies** and **segments** tumors in brain MRI scan images. This project automates medical imaging analysis to assist physicians in diagnosis.

### Key Features

- **4-class tumor classification**: Glioma, Meningioma, Pituitary, No Tumor
- **Instance segmentation**: Extract precise tumor boundaries and pixel-level masks
- **Integrated inference pipeline**: Combine classification and segmentation results visually for comprehensive diagnosis
- **High accuracy**: 99.4% classification Top-1 Accuracy, 92.7% segmentation Mask mAP50

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning Framework** | PyTorch, Ultralytics YOLO11 |
| **Computer Vision** | OpenCV, NumPy |
| **Models** | YOLO11m-cls (classification), YOLO11m-seg (segmentation) |
| **Programming Language** | Python 3.8+ |
| **Hardware** | NVIDIA GPU (CUDA recommended), CPU supported |

---

## Project Structure

```
BrainMRISegmentation_YOLO/
├── src/
│   ├── training/
│   │   └── train.py                    # Classification & segmentation training pipeline
│   └── testing/
│       └── test.py                     # Inference & integrated testing pipeline
├── brisc2025/
│   ├── classification_task/
│   │   ├── train/                      # Classification training data
│   │   │   ├── glioma/                 # Glioma images
│   │   │   ├── meningioma/             # Meningioma images
│   │   │   ├── pituitary/              # Pituitary images
│   │   │   └── no_tumor/               # Normal images
│   │   └── test/                       # Classification test data (same structure)
│   └── segmentation_task/
│       ├── train/                      # Segmentation training data
│       │   ├── images/                 # Original MRI images (.jpg, .png)
│       │   ├── masks/                  # Grayscale mask PNGs (auto-converted)
│       │   └── labels/                 # YOLO polygon labels .txt (auto-generated)
│       └── test/                       # Segmentation test data (same structure)
├── models/                             # Trained best-performing model weights
├── results/
│   ├── classification/                 # Classification training metrics & weights
│   ├── segmentation/                   # Segmentation training metrics & weights
│   ├── cls_inference/                  # Classification inference results
│   ├── seg_inference/                  # Segmentation inference results
│   └── test_results/                   # Integrated inference visualization results
└── README.md                           # This file
```

---

## Performance Metrics (50 epochs baseline)

### Classification

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 99.4% |
| Validation Loss | 0.030 |
| Per-class Performance | Consistent and high across all 4 classes |

**Characteristics**: Distinguishes each tumor type with high confidence, minimizes False Positives/Negatives

### Segmentation

| Metric | Value |
|--------|-------|
| Mask mAP50 | 92.7% |
| Box mAP50 | 91.7% |
| Mask Recall | 89.0% |
| Characteristics | Generates accurate masks even on images with unclear boundaries |

**Characteristics**: Captures precise tumor boundaries for high-precision medical imaging analysis

---

# User Guide

## 1. Environment Setup & Installation

### Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, Windows
- **Memory**: Minimum 8GB RAM (recommended for GPU)
- **Storage**: Minimum 10GB (for models and datasets)

### 1-1. Install Required Packages

#### Option 1: Basic Installation (CPU)

```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy pyyaml
```

#### Option 2: GPU Support Installation (Recommended)

```bash
# PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0 pyyaml>=6.0
```

#### Option 3: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Or Windows
venv\Scripts\activate

# Install packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy pyyaml
```

### 1-2. Verify GPU Setup

```bash
# Check NVIDIA GPU availability
nvidia-smi

# Or in Python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 2. Dataset Preparation

### 2-1. Classification Dataset Structure

Classification uses **class-based folder structure**. Folder names become class labels.

```
classification_task/
├── train/
│   ├── glioma/           # Glioma images
│   ├── meningioma/       # Meningioma images
│   ├── pituitary/        # Pituitary images
│   └── no_tumor/         # Normal images
└── test/
    ├── glioma/           # Test images (same class structure)
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

**Requirements**:
- Image formats: JPG, PNG, JPEG
- Minimum 20-30 images per class (training stability)
- Image size: Auto-adjusted (recommended: 320x320+)

### 2-2. Segmentation Dataset Structure

Segmentation uses **image + mask + label** structure.

```
segmentation_task/
├── train/
│   ├── images/           # Original MRI images (JPG, PNG)
│   │   ├── img_001.jpg
│   │   ├── img_002.png
│   │   └── ...
│   ├── masks/            # Grayscale mask PNGs (auto-converted to .txt)
│   │   ├── img_001_0_gl.png      # Glioma mask
│   │   ├── img_002_0_me.png      # Meningioma mask
│   │   └── ...
│   └── labels/           # YOLO polygon labels (auto-generated, editable)
│       ├── img_001_0_gl.txt
│       ├── img_002_0_me.txt
│       └── ...
└── test/
    ├── images/           # Test images
    ├── masks/            # Test masks
    └── labels/           # Test labels
```

### 2-3. Mask File Naming Convention

Mask PNG filenames must follow this format:

```
{image_id}_{category}_{index}_{class_key}.png
```

**Components**:
- `image_id`: Image unique ID (e.g., `img_001`, `sample_42`)
- `category`: Category number (e.g., `0`, `1`)
- `index`: Order index (e.g., `0`, `1`)
- `class_key`: Tumor type abbreviation

**class_key Mapping**:
| class_key | Tumor Type | YOLO ID |
|-----------|-----------|---------|
| `gl` | glioma | 0 |
| `me` | meningioma | 1 |
| `no` | no_tumor | 2 |
| `pi` | pituitary | 3 |

**Example filenames**:
```
img_001_0_gl.png   → Glioma mask
img_002_0_me.png   → Meningioma mask
img_003_0_no.png   → No tumor mask
img_004_0_pi.png   → Pituitary mask
```

**Mask characteristics**:
- **Format**: PNG (grayscale)
- **Pixel values**: 0 (background), 255 (tumor region)
- **Size**: Same as original image
- **Auto-conversion**: Automatically converted to YOLO polygon format (.txt) during training

---

## 3. Model Training

### 3-1. Train Classification Only

```bash
cd /path/to/BrainMRISegmentation_YOLO
python src/training/train.py --task cls --epochs 50 --batch 32
```

**Process**:
1. Load data from `brisc2025/classification_task/`
2. Initialize YOLO11m-cls model
3. Train for 50 epochs
4. Save best model to `models/yolo11m-cls-brain-best.pt`

### 3-2. Train Segmentation Only

```bash
python src/training/train.py --task seg --epochs 50 --batch 32
```

**Process**:
1. Load data from `brisc2025/segmentation_task/`
2. Auto-convert mask PNGs to YOLO polygon labels
3. Auto-generate `data.yaml`
4. Initialize YOLO11m-seg model
5. Train for 50 epochs
6. Save best model to `models/yolo11m-seg-brain-best.pt`

### 3-3. Train Both Classification & Segmentation

```bash
python src/training/train.py --task both --epochs 50 --batch 32
```

**Process**: Execute classification and segmentation sequentially

---

## 3-4. Training CLI Arguments Details

| Argument | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `--task` | str | `both` | `cls`, `seg`, `both` | Task type |
| `--epochs` | int | 50 | 1-1000 | Number of training epochs |
| `--batch` | int | 32 | 1-512 | Batch size. Reduce if GPU memory insufficient |
| `--imgsz` | int | 320 | 32-1024 | Input image size (pixels). Larger = better accuracy, slower |
| `--device` | int | 0 | 0, 1, 2, ... | GPU device index (0=first GPU, 1=second GPU, ...) |

---

## 3-5. Training Examples

### Basic Training (Recommended)
```bash
python src/training/train.py --task cls --epochs 50 --batch 32 --device 0
```

### High-Accuracy Training (Slower)
```bash
python src/training/train.py --task seg --epochs 100 --batch 16 --imgsz 640 --device 0
```

### Fast Training (Exploration)
```bash
python src/training/train.py --task both --epochs 20 --batch 64 --imgsz 256 --device 0
```

### Limited GPU Memory
```bash
python src/training/train.py --task both --epochs 50 --batch 16 --device 0
```

### Using Second GPU
```bash
python src/training/train.py --task both --epochs 50 --batch 32 --device 1
```

### CPU Training (No GPU)
```bash
python src/training/train.py --task cls --epochs 5 --batch 8 --device cpu
```

---

## 4. Inference & Testing

### 4-1. Classification Only

```bash
python src/testing/test.py --task cls
```

**Process**:
- Load classification test data: `brisc2025/classification_task/test/`
- Load trained model: `models/yolo11m-cls-brain-best.pt`
- Run inference on all images
- Save results: `results/cls_inference/`

**Output example**:
```
Image: img_001.jpg | Class: glioma (0.98)
Image: img_002.jpg | Class: no_tumor (0.97)
```

### 4-2. Segmentation Only

```bash
python src/testing/test.py --task seg
```

**Process**:
- Load segmentation test data: `brisc2025/segmentation_task/test/`
- Load trained model: `models/yolo11m-seg-brain-best.pt`
- Run inference on all images
- Visualize masks and bounding boxes
- Save results: `results/seg_inference/`

**Output example**:
```
Image: img_001.jpg | Detections: 2, Masks: 2
Image: img_002.jpg | Detections: 1, Masks: 1
```

### 4-3. Integrated Inference (Classification + Segmentation)

```bash
python src/testing/test.py --task integrated
```

**Process**:
1. Run classification model on each image → determine tumor type
2. Run segmentation model on same image → extract tumor location and mask
3. Combine classification result (class name, confidence) with segmentation result (mask, bounding box)
4. Save combined visualization: `results/test_results/`

**Output example**:
```
Processed img_001.jpg: glioma (0.98), 2 masks
Processed img_002.jpg: meningioma (0.95), 1 masks
```

---

## 4-4. Inference CLI Arguments Details

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--task` | str | - | **Required** | `cls`, `seg`, or `integrated` |
| `--source` | str | Default test path | Optional | Single image file or directory path |
| `--weights` | str | Model directory defaults | Optional | Custom model weights file path (.pt) |
| `--conf` | float | 0.25 | Optional | Confidence threshold (0.0-1.0). Lower = more detections |

---

## 4-5. Inference Examples

### Basic Classification Test
```bash
python src/testing/test.py --task cls
```

### Segment All Images in Folder
```bash
python src/testing/test.py --task seg --source /path/to/images/
```

### Integrated Inference on Single Image
```bash
python src/testing/test.py --task integrated --source /path/to/image.jpg
```

### High Confidence Threshold (Fewer False Positives)
```bash
python src/testing/test.py --task cls --conf 0.7
```

### Low Confidence Threshold (Higher Sensitivity)
```bash
python src/testing/test.py --task seg --conf 0.15
```

### Custom Model Weights
```bash
python src/testing/test.py --task integrated \
  --weights /custom/path/yolo11m-seg-brain-best.pt \
  --source /custom/images/
```

### Custom Path with High Confidence
```bash
python src/testing/test.py --task cls \
  --source /data/classification_test/ \
  --conf 0.75
```

---

## 5. Result Interpretation

### 5-1. Reading Classification Results

#### Log Output
```
Image: patient_001.jpg | Class: glioma (0.98)
```

**Meaning**:
- **Image**: Input image filename
- **Class**: Predicted tumor type
- **(0.98)**: Confidence score (0.0-1.0)
  - 1.0 = 100% confident
  - 0.5 = 50% confident

#### Visualization Interpretation
- Top-left corner shows "Class: glioma (0.98)"
- Green text indicates high confidence

### 5-2. Reading Segmentation Results

#### Log Output
```
Image: patient_001.jpg | Detections: 2, Masks: 2
```

**Meaning**:
- **Detections**: Number of detected tumors (bounding boxes)
- **Masks**: Number of generated masks (pixel-level segmentation)

#### Visualization Interpretation
- **Transparent masks**: Tumor regions colored
- **Bounding boxes**: Rectangles around tumors
- **Class labels**: Tumor type labels on each detection

### 5-3. Reading Integrated Results

#### Output Example
```
Processed patient_001.jpg: glioma (0.98), 2 masks
```

**Meaning**:
- Classification: glioma (confidence 0.98)
- Segmentation: 2 masks detected
- Final image: classification text + segmentation masks + bounding boxes

---

## 5-4. Confidence Threshold Adjustment Guide

### High Threshold (0.7+)
```bash
python src/testing/test.py --task cls --conf 0.8
```
**Characteristics**:
- Only high-confidence results
- Minimizes False Positives
- Lower Sensitivity (Recall)
- **Use case**: Conservative diagnosis, clinical review

### Medium Threshold (0.3-0.7)
```bash
python src/testing/test.py --task cls --conf 0.5
```
**Characteristics**:
- Balanced accuracy and sensitivity
- Suitable for most cases
- **Use case**: General diagnosis

### Low Threshold (0.1 or less)
```bash
python src/testing/test.py --task cls --conf 0.1
```
**Characteristics**:
- Includes weak signals
- Minimizes False Negatives
- Higher Sensitivity (Recall)
- **Use case**: Initial screening, catch missed cases

---

# Developer Guide

## 1. Prerequisites

### 1-1. System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| CUDA | 11.8 | 12.1+ |
| cuDNN | 8.6 | 8.7+ |
| RAM | 8GB | 16GB+ |
| GPU Memory | 4GB | 8GB+ |

### 1-2. Installation Verification

```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO ready')"
```

---

## 2. Detailed Installation

### 2-1. Complete Installation Process

```bash
# 1. Clone repository
git clone <repository-url>
cd BrainMRISegmentation_YOLO

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch with GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install other dependencies
pip install ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0 pyyaml>=6.0

# 6. Verify installation
python -c "from ultralytics import YOLO; print('YOLO ready')"
```

### 2-2. Troubleshooting

#### CUDA/cuDNN Setup
```bash
nvidia-smi
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### PyTorch CUDA Compatibility
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Project Structure Details

### 3-1. Directory & File Roles

#### `src/training/train.py`

**Role**: Train classification and segmentation models

**Main Functions**:

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `train_classification()` | data_dir, results_dir, model_out_dir, epochs, batch, imgsz, device | Saved model | Train YOLO11m-cls |
| `train_segmentation()` | data_dir, results_dir, model_out_dir, epochs, batch, imgsz, device | Saved model | Train YOLO11m-seg |
| `prepare_segmentation_labels()` | base_dir | - | Convert PNG masks to YOLO labels |
| `create_seg_yaml()` | data_root | yaml_path | Generate YOLO config file |
| `mask_to_polygons()` | mask_path, class_id | polygon_string | Convert mask to polygon coordinates |
| `_prepare_yolo_val_dir()` | data_path | - | Prepare validation directory (symlink/copy) |
| `_save_best_model()` | results, model_out_dir, filename | - | Save best-performing model |

#### `src/testing/test.py`

**Role**: Run inference and evaluation with trained models

**Main Functions**:

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `run_test()` | task, model_path, source, results_dir, imgsz, conf | Visualization images | Single task (cls/seg) inference |
| `run_integrated_test()` | cls_weights, seg_weights, source, results_dir, imgsz, conf | Combined visualization | Classification + segmentation inference |

#### `brisc2025/` Directory

- **classification_task/**: Class-based folder structure
- **segmentation_task/**: images, masks, labels structure

#### `models/` Directory

Trained model weights:
- `yolo11m-cls-brain-best.pt`: Classification model
- `yolo11m-seg-brain-best.pt`: Segmentation model

#### `results/` Directory

Training and inference results:
- `classification/`: Training metrics
- `segmentation/`: Training metrics
- `cls_inference/`: Classification results
- `seg_inference/`: Segmentation results
- `test_results/`: Integrated visualization

---

## 4. CLASS_MAP Explanation & Extension

### 4-1. Default CLASS_MAP

Defined in `src/training/train.py`:

```python
CLASS_MAP: dict[str, int] = {
    "glioma": 0,           # Glioma (ID: 0)
    "meningioma": 1,       # Meningioma (ID: 1)
    "no_tumor": 2,         # Normal (ID: 2)
    "pituitary": 3,        # Pituitary (ID: 3)
    "gl": 0,               # Abbreviation
    "me": 1,
    "no": 2,
    "pi": 3,
}
```

**Purpose**:
- Extract `class_key` from mask filename and convert to YOLO class ID
- Example: `image_001_0_gl.png` → class_key="gl" → ID=0

### 4-2. Adding New Tumor Classes

To add a new tumor class (e.g., `adenoma`):

#### Step 1: Update CLASS_MAP

Modify `src/training/train.py`:

```python
CLASS_MAP: dict[str, int] = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
    "adenoma": 4,          # New class
    "gl": 0,
    "me": 1,
    "no": 2,
    "pi": 3,
    "ad": 4,               # Abbreviation
}
```

#### Step 2: Update Dataset Structure

Classification folder structure:
```
classification_task/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   ├── no_tumor/
│   └── adenoma/          # New folder
└── test/
    └── adenoma/          # New folder
```

Segmentation mask filenames:
```
image_001_0_ad.png    # Adenoma mask
```

#### Step 3: Update YAML

Modify `create_seg_yaml()` function:

```python
def create_seg_yaml(data_root: str) -> str:
    yaml_content: dict = {
        ...
        "names": {
            0: "glioma",
            1: "meningioma",
            2: "no_tumor",
            3: "pituitary",
            4: "adenoma",      # New class
        },
    }
```

#### Step 4: Verify

Ensure mask filenames are correct:
```
image_001_0_ad.png  # Correct
image_001_0_adenoma.png  # Wrong (uses full class name as class_key)
```

---

## 5. Core Functions Detailed Explanation

### 5-1. mask_to_polygons(mask_path, class_id) → str

**Purpose**: Convert grayscale mask PNG to YOLO polygon format

**Input**:
- `mask_path` (Path): PNG mask file path
- `class_id` (int): YOLO class ID (0-3)

**Output**:
- str: YOLO polygon format label string

**Process**:

```
1. Read PNG mask (cv2.IMREAD_GRAYSCALE)
   → Grayscale image (0-255)

2. Apply binary threshold
   → Convert pixels ≥1 to 255, rest to 0

3. Morphological operations for noise removal
   ├─ MORPH_CLOSE: Fill small holes (5x5 kernel)
   └─ MORPH_OPEN: Remove small protrusions (5x5 kernel)

4. Extract contours
   → Find tumor region outlines

5. Filter by area threshold (h * w * 0.001)
   → Remove noise

6. Generate normalized polygon coordinates (0-1)
   → YOLO format (x1, y1, x2, y2, ...)
```

**Output Format**:
```
{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} ...
```

**Example**:
```
0 0.234567 0.345678 0.456789 0.567890 0.123456 0.654321 ...
```

### 5-2. _prepare_yolo_val_dir(data_path) → None

**Purpose**: Prepare validation directory for YOLO training

**Input**:
- `data_path` (Path): Data directory path

**Process**:

```
1. Check if val/ exists
   └─ If yes: Do nothing

2. If val/ missing but test/ exists:
   ├─ Attempt 1: Create symlink from test/ to val/
   │   (Works on Linux/Mac)
   │
   └─ Attempt 2: Copy test/ to val/ if symlink fails
       (Works on Windows)

3. Enable YOLO data loader to find val/ path
```

**Importance**: YOLO looks for validation data at `val/` path during training.

**Example code**:
```python
val_path = data_path / "val"
test_path = data_path / "test"

if not val_path.exists() and test_path.exists():
    try:
        os.symlink(test_path.name, val_path)
    except OSError:
        shutil.copytree(test_path, val_path, dirs_exist_ok=True)
```

### 5-3. _save_best_model(results, model_out_dir, filename) → None

**Purpose**: Save best-performing model after training

**Input**:
- `results` (Results): YOLO training results object
- `model_out_dir` (str): Model output directory
- `filename` (str): Output filename

**Process**:

```
1. Find best weights from training results
   → results.save_dir/weights/best.pt

2. Copy to output directory
   → model_out_dir/filename

3. Log completion
```

**Example**:
```python
_save_best_model(results, "models/", "yolo11m-cls-brain-best.pt")
# Creates: models/yolo11m-cls-brain-best.pt
```

---

## 6. Training YAML Config Details

Auto-generated `data.yaml` file structure for segmentation:

```yaml
path: /absolute/path/to/data
train: train/images
val: test/images
names:
  0: glioma
  1: meningioma
  2: no_tumor
  3: pituitary
```

### 6-1. Field Descriptions

| Field | Value | Description |
|-------|-------|-------------|
| `path` | Absolute path | Data root directory (YOLO reference point) |
| `train` | `train/images` | Training images directory (relative to path) |
| `val` | `test/images` | Validation images directory (relative to path) |
| `names` | Dict[int, str] | Class ID → class name mapping |

### 6-2. Manual YAML Editing

For custom configurations:

```yaml
path: /home/user/data/segmentation_task
train: train/images
val: test/images
names:
  0: glioma
  1: meningioma
  2: no_tumor
  3: pituitary
```

---

## 7. --device Parameter Guide

The `--device` parameter specifies which GPU to use.

### 7-1. Check Available GPUs

```bash
nvidia-smi

# Example output:
# GPU 0: NVIDIA A100 (40GB)
# GPU 1: NVIDIA A100 (40GB)
# GPU 2: NVIDIA RTX 4090 (24GB)
```

### 7-2. Single GPU Usage

```bash
# Use first GPU (GPU 0) [default]
python src/training/train.py --task both --device 0

# Use second GPU (GPU 1)
python src/training/train.py --task both --device 1

# Use third GPU (GPU 2)
python src/training/train.py --task both --device 2
```

### 7-3. Multiple GPU Training

Current implementation supports single GPU. For multi-GPU training:

```python
# Modify src/training/train.py
device = [0, 1, 2]  # Use GPUs 0, 1, 2 simultaneously

results = model.train(
    ...
    device=device,  # Pass as list
    ...
)
```

### 7-4. CPU Usage (No GPU)

```bash
python src/training/train.py --task cls --epochs 5 --batch 8 --device cpu
```

### 7-5. GPU Memory Optimization

When GPU memory is insufficient:

```bash
# Reduce batch size (decreases memory usage)
python src/training/train.py --task both --batch 8 --device 0

# Reduce image size (decreases memory usage)
python src/training/train.py --task both --imgsz 256 --device 0

# Both reductions
python src/training/train.py --task both --batch 8 --imgsz 256 --device 0
```

---

## 8. Using Custom Model Weights

Use trained models for inference:

```bash
# Custom classification model
python src/testing/test.py --task cls \
  --weights /path/to/custom-cls-model.pt

# Custom segmentation model
python src/testing/test.py --task seg \
  --weights /path/to/custom-seg-model.pt

# Integrated (requires both models in default paths)
python src/testing/test.py --task integrated
```

---

## 9. Logging & Debugging

### 9-1. Log Level Settings

In `src/training/train.py` or `src/testing/test.py`:

```python
logging.basicConfig(
    level=logging.INFO,  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

### 9-2. Debug Information

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Starting training...")
logger.warning("Low GPU memory detected")
logger.error("Model file not found")
```

---

## License

This project uses YOLO11 from Ultralytics (licensed under AGPL-3.0).

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**Last Updated**: 2026-04-01
**Python Version**: 3.8+
**YOLO Version**: 11
