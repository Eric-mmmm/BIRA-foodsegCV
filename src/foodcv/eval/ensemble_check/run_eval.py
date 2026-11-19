"""Run the ensemble on FoodSeg103 (+ optional non-food photos) and save raw results.

Needs: pip install onnxruntime pyarrow opencv-python-headless pillow numpy
Data:  FoodSeg103 test parquet (lossless masks) + id2label.json from
       https://huggingface.co/datasets/EduardoPacheco/FoodSeg103  (data/validation-*.parquet)
       Do NOT use the datasets-server masks: they are lossy JPEG and corrupt class ids.
Usage: python run_eval.py --parquet val.parquet --out out/ [--nonfood-dir imgs/] [--n-target 300 --n-other 200]
"""
import argparse, glob, io, json, random, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from ens_onnx import Ensemble

# FoodSeg103 id -> deeplab class index (deeplab.json order)
TARGET = {29: 1, 87: 2, 84: 3, 72: 4, 64: 5, 66: 6, 58: 7, 70: 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nonfood-dir")
    ap.add_argument("--n-target", type=int, default=300, help="photos containing a DeepLab food")
    ap.add_argument("--n-other", type=int, default=200, help="photos containing none of them")
    ap.add_argument("--skip-b1-all", action="store_true", help="skip B1 on the whole split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    t = pq.read_table(a.parquet).to_pydict()
    n = len(t["id"])
    img_of = lambda i: Image.open(io.BytesIO(t["image"][i]["bytes"])).convert("RGB")
    lab_of = lambda i: np.array(Image.open(io.BytesIO(t["label"][i]["bytes"])))
    e = Ensemble()
    t0 = time.time()

    # phase 1: B1 on everything (cheap)
    b1 = {"food": [], "nonfood": []}
    if not a.skip_b1_all:
        b1["food"] = [(i, e.food(img_of(i))) for i in range(n)]
    if a.nonfood_dir:
        for f in sorted(glob.glob(str(Path(a.nonfood_dir) / "*.jpg"))):
            b1["nonfood"].append((f, e.food(Image.open(f).convert("RGB"))))
    json.dump(b1, open(out / "b1.json", "w"))
    print("phase 1 done", round(time.time() - t0), "s", flush=True)

    # phase 2: full ensemble on a subset; confusion matrix pred(9) x gt(104)
    rng = random.Random(a.seed)
    has = [i for i in range(n) if any(c in TARGET for c in t["classes_on_image"][i])]
    hs = set(has)
    other = [i for i in range(n) if i not in hs]
    sel = rng.sample(has, min(a.n_target, len(has))) + rng.sample(other, min(a.n_other, len(other)))
    conf = np.zeros((9, 104), np.int64)
    recs = []
    for k, i in enumerate(sel):
        img, gt = img_of(i), lab_of(i)
        np.add.at(conf, (e.seg(img).ravel(), gt.ravel()), 1)
        r = {"i": i, "gt": [int(c) for c in t["classes_on_image"][i]], "food": e.food(img)}
        for w in ("efficientnet", "convnext"):
            lab, sc, acc, _ = e.dish(img, w)
            r[w] = [lab, sc, bool(acc)]
        recs.append(r)
        if k % 25 == 0:
            print(k, "/", len(sel), round(time.time() - t0), "s", flush=True)
    np.save(out / "conf.npy", conf)
    json.dump(recs, open(out / "recs.json", "w"))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
