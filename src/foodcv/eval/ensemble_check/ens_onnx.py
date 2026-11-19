"""CPU port of models/ensemble/bench_photo.py (numpy + PIL + onnxruntime, no torch/TensorRT).

Same resize/crop/normalisation, thresholds and region filter as bench_photo.py, so the
outputs can be checked on any machine. Differences: FP32 on CPU, and logits are
upsampled with cv2 bilinear instead of torch bilinear (equivalent, half-pixel centres).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ENSEMBLE_DIR = Path(__file__).resolve().parents[2] / "models" / "ensemble"
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
NAMES = ["b1", "efficientnet", "convnext", "deeplab"]


def _resize(img, size):
    if isinstance(size, int):  # short side -> size, like torchvision
        w, h = img.size
        s, l = (w, h) if w <= h else (h, w)
        ns, nl = size, int(size * l / s)
        nw, nh = (ns, nl) if w <= h else (nl, ns)
    else:
        nh, nw = size
    return img.resize((nw, nh), Image.BILINEAR)


def tensor(img, size, crop=None):
    img = _resize(img, size)
    if crop:
        w, h = img.size
        t, l = int(round((h - crop) / 2.)), int(round((w - crop) / 2.))
        img = img.crop((l, t, l + crop, t + crop))
    a = (np.asarray(img, np.float32) / 255. - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class Ensemble:
    def __init__(self, model_dir=ENSEMBLE_DIR, threads=4):
        model_dir = Path(model_dir)
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        self.sess = {n: ort.InferenceSession(str(model_dir / f"{n}.onnx"), o,
                                             providers=["CPUExecutionProvider"]) for n in NAMES}
        self.cfg = {n: json.loads((model_dir / f"{n}.json").read_text()) for n in NAMES}
        self.food_threshold = self.cfg["b1"]["threshold"]
        self.min_frac = self.cfg["deeplab"]["region"]["minimum_area_fraction"]

    def _run(self, name, x):
        return self.sess[name].run(None, {"images": x})[0]

    def food(self, img):
        """B1 food-presence probability."""
        return float(softmax(self._run("b1", tensor(img, 256, 224)))[0, 1])

    def dish(self, img, which="efficientnet"):
        """-> (label, score, accepted, all_probs) with the same acceptance rule as bench_photo.py."""
        spec = self.cfg[which]["dish"]
        p = softmax(self._run(which, tensor(img, 365, 320)))[0]
        i = int(p.argmax())
        label, score = spec["classes"][i], float(p[i])
        thr = spec.get("per_class_thresholds", {}).get(label, spec["acceptance_threshold"])
        return label, score, score >= thr, p

    def seg(self, img):
        """DeepLab class-index mask (0=background, 1..8 = deeplab.json order) at image size."""
        logits = self._run("deeplab", tensor(img, [640, 640]))[0]
        w, h = img.size
        up = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in logits])
        mask = up.argmax(0).astype(np.uint8)
        counts = np.bincount(mask.ravel(), minlength=9)
        for k in range(1, 9):
            if counts[k] < mask.size * self.min_frac:
                mask[mask == k] = 0
        return mask
