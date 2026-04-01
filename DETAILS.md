# Brain MRI Segmentation & Classification 기술 상세

## 1. 학습(Training) CLI 인수 상세 (`train.py`)
| 인수 | 기본값 | 범위 | 설명 |
|------|--------|------|------|
| `--task` | `both` | `cls`, `seg`, `both` | 수행할 작업 지정 |
| `--epochs` | 50 | 1 ~ 1000 | 학습 에포크 수 |
| `--batch` | 32 | 1 ~ 512 | 배치 크기. 메모리 부족 시 감소 |
| `--imgsz` | 320 | 32 ~ 1024 | 입력 이미지 크기. 클수록 정확도 향상되나 느려짐 |
| `--device` | 0 | 0, 1, cpu | 사용할 GPU 인덱스 또는 CPU |

## 2. 추론(Testing) 신뢰도 임계값 가이드
추론 시 `--conf` 옵션으로 감도(Sensitivity)를 조절할 수 있습니다.
- **높은 임계값 (0.7 이상)**: `python test.py --task cls --conf 0.8` (보수적 진단, 거짓 양성 최소화)
- **중간 임계값 (0.3 ~ 0.7)**: 일반적인 균형 잡힌 정확도
- **낮은 임계값 (0.1 이하)**: `python test.py --task seg --conf 0.1` (약한 신호 포함, 거짓 음성 최소화)

## 3. 새로운 종양 클래스 추가 (`CLASS_MAP`)
새로운 질환(예: `adenoma`)을 추가하려면 `train.py` 내의 맵을 수정해야 합니다.
```python
CLASS_MAP: dict[str, int] = {
    "glioma": 0, "meningioma": 1, "no_tumor": 2, "pituitary": 3,
    "adenoma": 4,  # 새 클래스 추가
    "gl": 0, "me": 1, "no": 2, "pi": 3,
    "ad": 4        # 파일명 분석용 약자 매핑
}
```
또한 데이터셋 폴더 구조와 자동 생성되는 `yaml` 설정 파일 내의 `names` 딕셔너리도 동일하게 확장해주어야 합니다.

## 4. 마스크-다각형 변환 핵심 엔진 (`mask_to_polygons`)
수동 라벨링을 대체하는 의료용 마스크 자동 변환 로직입니다.
1. **이진화(Binary threshold)** 적용: 픽셀 값 > 0 을 255로 치환.
2. **형태학적 연산(Morphological Filtering)**:
   - `MORPH_CLOSE`: 5x5 커널을 사용해 종양 내부의 작은 구멍(노이즈)을 채웁니다.
   - `MORPH_OPEN`: 5x5 커널로 주변에 흩어진 미세한 돌기나 픽셀 찌꺼기를 제거합니다.
3. **컨투어(Contour) 추출**: 면적 임계값(`w * h * 0.001`) 이상의 주요 객체 윤곽선만 추출하여 YOLO 포맷의 정규화(0~1)된 Polygon 좌표계로 반환합니다.
