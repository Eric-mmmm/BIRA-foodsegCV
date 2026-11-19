"""Print the summary tables from run_eval.py output.  Usage: python analyze.py --out out/ --id2label id2label.json"""
import argparse, collections, json
from pathlib import Path

import numpy as np

NAMES = ["background", "banana", "broccoli", "carrot", "cauliflower", "pasta", "rice", "bread", "potato"]
GT = {1: [29], 2: [87], 3: [84], 4: [72], 5: [64], 6: [66], 7: [58], 8: [70]}
ITALIAN = {"macaroni_and_cheese", "spaghetti_bolognese", "spaghetti_carbonara", "lasagna", "gnocchi", "risotto"}
THR = 0.68392014503479  # b1.json threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--id2label", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    id2 = {int(k): v for k, v in json.load(open(a.id2label)).items()}

    b1 = json.load(open(out / "b1.json"))
    f = np.array([s for _, s in b1["food"]]); nf = np.array([s for _, s in b1["nonfood"]])
    if len(f): print(f"B1: food photos accepted {np.mean(f >= THR):.1%} of {len(f)}")
    if len(nf): print(f"B1: non-food photos wrongly accepted {np.sum(nf >= THR)} / {len(nf)}")

    conf = np.load(out / "conf.npy")
    print("\nDeepLab IoU vs FoodSeg103 ground truth (pixel level, over the sample)")
    for k in range(1, 9):
        g = GT[k]; tp = conf[k, g].sum(); fp = conf[k].sum() - tp; fn = conf[:, g].sum() - tp
        print(f"  {NAMES[k]:12s} IoU {tp/(tp+fp+fn):.3f}  recall {tp/(tp+fn):.3f}  precision {tp/(tp+fp+1e-9):.3f}")
    tgt = [c for v in GT.values() for c in v]
    oth = [c for c in range(1, 104) if c not in tgt]
    food_px = conf[:, 1:].sum()
    print(f"\n  food pixels in the 8 supported foods: {conf[:, tgt].sum()/food_px:.1%}; in all other foods: {conf[:, oth].sum()/food_px:.1%}")
    print(f"  other-food pixels labelled background: {conf[0, oth].sum()/conf[:, oth].sum():.1%}")
    print(f"  background pixels labelled food: {conf[1:, 0].sum()/conf[:, 0].sum():.2%}")

    recs = json.load(open(out / "recs.json"))
    def grp(r):
        g = set(r["gt"])
        return ("pasta" if 64 in g else "noodles" if 65 in g else "rice (no pasta)" if 66 in g else "no pasta/noodles/rice")
    G = collections.defaultdict(list)
    for r in recs: G[grp(r)].append(r)
    for w in ("efficientnet", "convnext"):
        print(f"\nDish classifier: {w}")
        for g, rs in G.items():
            ital = sum(r[w][0] in ITALIAN for r in rs); accI = sum(r[w][0] in ITALIAN and r[w][2] for r in rs)
            print(f"  {g:24s} n={len(rs):3d}  Italian label {ital/len(rs):5.1%}  accepted {accI/len(rs):5.1%}  any-label accepted {np.mean([r[w][2] for r in rs]):5.1%}")


if __name__ == "__main__":
    main()
