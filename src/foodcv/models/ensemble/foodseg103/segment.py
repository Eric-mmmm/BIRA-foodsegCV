"""Segment one RGB photo with the FoodSeg103 TensorRT engine."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bench_photo import Engine, LOGGER, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("foodseg103_mask.png"))
    args = parser.parse_args()
    names = json.loads((HERE / "id2label.json").read_text())
    with Image.open(args.photo) as source:
        image = source.convert("RGB")
    runtime = trt.Runtime(LOGGER)
    engine = Engine(runtime, "foodseg103")
    with torch.inference_mode():
        scores = engine(tensor(image, [256, 256]))
        if tuple(scores.shape) != (1, 104, 256, 256):
            raise ValueError(f"Unexpected shape {scores.shape}")
        ids = scores.argmax(1)[0].byte().cpu().numpy()
    mask = Image.fromarray(ids).resize(image.size, Image.Resampling.NEAREST)
    mask.save(args.output, format="PNG")
    values, counts = np.unique(np.asarray(mask), return_counts=True)
    print(json.dumps([dict(id=int(v), label=names[str(v)].strip(), pixels=int(n))
                      for v, n in zip(values, counts)], ensure_ascii=False))


if __name__ == "__main__":
    main()
