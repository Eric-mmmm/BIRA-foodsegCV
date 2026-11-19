"""Benchmark one decoded RGB photo with TensorRT 10 on Jetson."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--engine", type=Path, default=Path("foodseg103_fp16.engine"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--mask", type=Path, default=Path("foodseg103_mask.png"))
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1:
        parser.error("--runs and --warmup must be positive")
    import tensorrt as trt
    import torch

    with Image.open(args.photo) as image:
        photo = image.convert("RGB")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Rebuild the engine on this Jetson with its installed TensorRT.")
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream()
    shape = tuple(engine.get_tensor_shape("logits"))
    if tuple(engine.get_tensor_shape("image")) != (1, 3, 448, 448) or shape != (1, 104, 448, 448):
        raise ValueError("Expected the 448-pixel BIRA DINOv2 engine.")
    if any(engine.get_tensor_dtype(name) != trt.float32 for name in ("image", "logits")):
        raise ValueError("Expected FP32 input and output bindings.")
    output = torch.empty(shape, device="cuda", dtype=torch.float32)
    context.set_tensor_address("logits", output.data_ptr())
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    times = []
    for i in range(args.warmup + args.runs):
        start = time.perf_counter()
        rgb = np.asarray(photo.resize((448, 448), Image.Resampling.BILINEAR), np.float32) / 255
        batch = np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None])
        tensor = torch.from_numpy(batch).cuda()
        context.set_tensor_address("image", tensor.data_ptr())
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed.")
        ids = output.argmax(1)[0].byte().cpu().numpy()
        mask = Image.fromarray(ids).resize(photo.size, Image.Resampling.NEAREST)
        if i >= args.warmup:
            times.append((time.perf_counter() - start) * 1000)
    mask.save(args.mask)
    print(json.dumps(dict(device=torch.cuda.get_device_name(), tensorrt=trt.__version__,
                          photo_size=photo.size, runs=args.runs, warmup=args.warmup,
                          photo_to_mask_p50_ms=float(np.median(times)),
                          photo_to_mask_p95_ms=float(np.percentile(times, 95)),
                          mask=str(args.mask)), indent=2))


if __name__ == "__main__":
    main()
