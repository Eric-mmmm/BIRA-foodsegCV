# Ensemble check (CPU, no Jetson needed)

Checks what the ensemble in `models/ensemble/` actually outputs, using FoodSeg103 ground truth.
`ens_onnx.py` is a numpy/onnxruntime copy of `bench_photo.py` (same preprocessing and thresholds).

```bash
pip install onnxruntime pyarrow opencv-python-headless pillow numpy
git lfs pull --include="src/foodcv/models/ensemble/*.onnx"
# data: data/validation-00000-of-00001.parquet and id2label.json from
# https://huggingface.co/datasets/EduardoPacheco/FoodSeg103  (use the parquet; datasets-server masks are lossy JPEG)
cd src/foodcv/eval/ensemble_check
python run_eval.py --parquet val.parquet --out out --nonfood-dir some_non_food_jpgs/   # ~0.9 s/photo on 2 CPU cores
python analyze.py --out out --id2label id2label.json
```

## Result of the run on 2026-11-19 (500-photo sample of the 2135 test photos, B1 on all 2135)
- B1: accepts 97.8% of food photos, 0/100 non-food photos (Imagenette) accepted.
- DeepLab covers 8 foods only (banana, broccoli, carrot, cauliflower, pasta, rice, bread, potato).
  IoU: broccoli .77, carrot .63, rice .58, pasta .52, banana .48, potato .45, cauliflower .44, bread .43.
  About 75% of food pixels belong to other foods, and 89% of those are labelled background.
- Dish classifiers (6 Italian dishes + other_food) give confident wrong labels on non-pasta photos
  (asparagus -> risotto, granola bars -> lasagna, strawberry tart -> gnocchi). ~10% of photos with no
  pasta/noodles/rice get an Italian label.
- `dish_accepted` is always True for ConvNeXt (threshold 0.0, no per-class thresholds); carbonara and
  risotto have threshold 0.0 for EfficientNet.
- Output is one image-level label + one class-index mask, not a list of `Detection`s (no per-item
  centroid/bbox/score).
- Caveats: FP32 CPU, not TensorRT FP16; training data of the models unknown; web photos, not ZED Mini view.
