# BIRA-foodsegCV

Computer vision model for HEKA-BIRA, a robotic arm that helps people eat.

Given a camera image of a plate, the model must:

1. **Identify** each food item (classification)
2. **Segment** it (find its borders as a pixel mask)
3. **Locate** it (output coordinates the arm can target)

## Team and ownership

| Area | Owner | Folder |
|---|---|---|
| Ensemble pipeline (3 pretrained CV models) | @polyhassan | `src/foodcv/models/ensemble/` |
| Custom model built from scratch | @Eric-mmmm | `src/foodcv/models/custom/` |
| Standardized testing / evaluation pipeline | Harlise | `src/foodcv/eval/` |

## The interface

Every model, whoever wrote it, must expose the same interface so the evaluation pipeline can test all of them identically. See `src/foodcv/interface.py`.

- Input: an RGB image (`numpy` array, H x W x 3)
- Output: a list of detections, each with `label`, `score`, `mask`, `bbox` and `centroid_px`

Changes to the interface need a review from all three of us.

## Repository structure

```
src/foodcv/     # source code (data, models, eval, interface)
configs/        # YAML configs (hyperparameters, paths)
splits/         # train/val/test image-ID lists (shared by all models)
results/        # one JSON per evaluation run
scripts/        # train.py, evaluate.py
tests/          # tests and smoke tests
docs/           # notes, dataset documentation
```

## Setup

```bash
git clone https://github.com/<owner>/BIRA-foodsegCV.git
cd BIRA-foodsegCV
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Datasets and model weights are **not stored in git**.

- Dataset: TODO (name, source, license, download link)
- Annotation format: TODO (e.g. COCO)
- Where to put it: `data/` (gitignored)
- Model weights: TODO (shared drive / Hugging Face link)

## Usage

```bash
# Train the custom model
python scripts/train.py --config configs/custom.yaml

# Evaluate any model on the shared test split
python scripts/evaluate.py --config configs/eval.yaml --model custom
```

TODO: update once the scripts exist.

## Evaluation

All models are tested on the same splits (`splits/`) with the same metrics:

- Segmentation quality: mIoU / mask AP
- Classification accuracy
- Coordinate error (pixels, or mm once calibrated)
- Inference speed (ms per image)

## Results

| Model | mIoU | Centroid error | Speed (ms) | Commit |
|---|---|---|---|---|
| Ensemble | TODO | TODO | TODO | TODO |
| Custom | TODO | TODO | TODO | TODO |

## Workflow

- `main` is protected: all changes go through a pull request with at least one review.
- One issue per task. Branch names follow `feature/<area>-<short-description>`.
- Link the issue in the PR and report metrics if the PR changes a model.
- Keep hyperparameters in `configs/`, not hardcoded.
- Never commit datasets, weights, or any patient/personal data.

## Status

Work in progress. See the Issues and Projects tabs for current tasks.
