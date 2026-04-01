# Brain MRI Segmentation Technical Details

## 1. Training CLI Arguments (`train.py`)
| Argument | Default | Range | Description |
|----------|---------|-------|-------------|
| `--task` | `both` | `cls`, `seg`, `both` | Task to perform |
| `--epochs` | 50 | 1-1000 | Total training epochs |
| `--batch` | 32 | 1-512 | Batch size. Reduce if OOM |
| `--imgsz` | 320 | 32-1024 | Image resolution |
| `--device` | 0 | 0, 1, cpu | GPU index or CPU |

## 2. Confidence Threshold Guide
Adjust the sensitivity using the `--conf` flag during inference.
- **High (0.7+)**: `python test.py --task cls --conf 0.8` (Conservative diagnosis, minimizes false positives.)
- **Medium (0.3 - 0.7)**: Balanced accuracy.
- **Low (< 0.1)**: `python test.py --task seg --conf 0.1` (Maximizes sensitivity to catch weak signals, minimizes false negatives.)

## 3. Extending Tumor Classes (`CLASS_MAP`)
To add a new class like `adenoma`, update `train.py`:
```python
CLASS_MAP: dict[str, int] = {
    # existing classes...
    "adenoma": 4,
    "ad": 4
}
```
You must also update the folder structures and the `names` dictionary in the generated `data.yaml`.

## 4. Mask-to-Polygon Engine (`mask_to_polygons`)
1. **Binarization**: Converts all positive pixel values to 255.
2. **Morphological Filtering**:
   - `MORPH_CLOSE` (5x5): Fills small holes inside the tumor mask.
   - `MORPH_OPEN` (5x5): Removes small scattered noise pixels around the boundaries.
3. **Contour Extraction**: Filters out tiny objects (area < `w * h * 0.001`) to prevent noise artifacts, and converts the remaining boundaries into YOLO-normalized (0-1) polygon coordinates.
