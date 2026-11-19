"""Run the exported FoodSeg103 ONNX on one RGB photo using NVIDIA CUDA."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("photo", type=Path)
parser.add_argument("output", type=Path, help="PNG where each pixel stores the predicted class ID")
args = parser.parse_args()
if args.output.suffix.lower() != ".png":
    parser.error("Choose a .png output to preserve class IDs.")
if args.output.exists():
    parser.error("Choose a new output filename.")
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
session = ort.InferenceSession(str(here / "foodseg103.onnx"), providers=[("CUDAExecutionProvider", {"use_tf32": "0"})])
if session.get_providers()[0] != "CUDAExecutionProvider":
    raise RuntimeError("This demo needs ONNX Runtime with CUDA.")
input_node = session.get_inputs()[0]
_, _, height, width = input_node.shape
photo = Image.open(args.photo).convert("RGB")
rgb = np.asarray(photo.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
tensor = np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None])
logits, = session.run(["logits"], {input_node.name: tensor})
mask = Image.fromarray(np.argmax(logits[0], axis=0).astype(np.uint8)).resize(photo.size, Image.Resampling.NEAREST)
mask.save(args.output)
labels = json.loads((here / "id2label.json").read_text())
ids, counts = np.unique(np.asarray(mask), return_counts=True)
for idx in np.argsort(-counts):
    print(f"{labels[str(ids[idx])].strip()}: {counts[idx] / sum(counts):.1%}")
