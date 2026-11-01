"""ap() must give the same number as pycocotools, and image weights must behave like
duplicated images. Run with: pytest tests/test_coco_eval.py"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pycocotools")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coco_eval import ap, collect  # noqa: E402

CATS = [3, 5, 7]


def synthetic(n_img, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_img):
        gts, dets = [], []
        for _ in range(rng.integers(0, 12)):
            c = int(rng.choice(CATS))
            s = float(rng.choice([12, 25, 40, 80, 150])) * rng.uniform(0.8, 1.2)
            box = [*rng.uniform(0, 500, 2), s, s * rng.uniform(0.5, 1.5)]
            gts.append((c, box))
            if rng.random() < 0.8:
                jitter = list(np.array(box) + rng.normal(0, box[2] * 0.08, 4))
                dets.append((c if rng.random() < 0.9 else 5, jitter, float(rng.random())))
        for _ in range(rng.integers(0, 5)):
            s = float(rng.uniform(10, 100))
            dets.append((int(rng.choice(CATS)), [*rng.uniform(0, 500, 2), s, s], float(rng.random() * 0.6)))
        data.append((gts, dets))
    return data


def coco_eval(data, order):
    images, anns, dets = [], [], []
    for new_id, i in enumerate(order, 1):
        gts, ds = data[i]
        images.append(dict(id=new_id, file_name=f"{new_id}.jpg", width=640, height=640))
        for c, b in gts:
            anns.append(dict(id=len(anns) + 1, image_id=new_id, category_id=c, bbox=b, area=b[2] * b[3], iscrowd=0))
        for c, b, s in ds:
            dets.append(dict(image_id=new_id, category_id=c, bbox=b, score=s))
    gt = COCO()
    gt.dataset = dict(images=images, annotations=anns, categories=[dict(id=c, name=str(c)) for c in CATS])
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.params.maxDets = [1, 100, 500]
        E.evaluate()
        E.accumulate()
    return E


def reference(E, k, a):
    p = E.eval["precision"][:, :, k, a, 2]
    return p[p > -1].mean() if (p > -1).any() else np.nan


def test_same_as_pycocotools():
    data = synthetic(60)
    E = coco_eval(data, range(60))
    for k in range(len(CATS)):
        for a in range(4):
            ref, mine = reference(E, k, a), ap(collect(E, k, a))
            assert (np.isnan(ref) and np.isnan(mine)) or abs(ref - mine) < 1e-12


def test_weights_equal_duplicated_images():
    data = synthetic(60, seed=1)
    E = coco_eval(data, range(60))
    order = np.random.default_rng(2).integers(0, 60, 60)
    E_dup = coco_eval(data, order)
    weights = np.bincount(order, minlength=60)
    for k in range(len(CATS)):
        for a in range(4):
            ref, mine = reference(E_dup, k, a), ap(collect(E, k, a), weights)
            assert (np.isnan(ref) and np.isnan(mine)) or abs(ref - mine) < 1e-12
