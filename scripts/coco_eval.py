"""COCO average precision per class and object size, with image weights for the bootstrap.

The functions repeat COCOeval.accumulate for one class and one area range. With all
image weights equal to 1 the result is the COCO AP; a bootstrap resample of the val
images is a weight vector (how often each image was drawn).
"""

import numpy as np


def collect(E, k, a, max_det=None):
    """Detections of one class (index k) and one area range (index a) from a COCOeval run,
    sorted by score over all images, as COCOeval.accumulate does it."""
    p = E.params
    n_img, n_area = len(p.imgIds), len(p.areaRng)
    max_det = max_det or p.maxDets[-1]
    rows = E.evalImgs[k * n_area * n_img + a * n_img: k * n_area * n_img + (a + 1) * n_img]
    scores, img, match, ignore = [], [], [], []
    npig = np.zeros(n_img)
    for i, e in enumerate(rows):
        if e is None:
            continue
        s = np.asarray(e['dtScores'][:max_det])
        scores.append(s)
        img.append(np.full(len(s), i))
        match.append(e['dtMatches'][:, :max_det])
        ignore.append(e['dtIgnore'][:, :max_det])
        npig[i] = np.count_nonzero(np.asarray(e['gtIgnore']) == 0)
    n_thr = len(p.iouThrs)
    if scores:
        scores = np.concatenate(scores)
        order = np.argsort(-scores, kind='mergesort')
        img = np.concatenate(img)[order]
        match = np.concatenate(match, axis=1)[:, order].astype(bool)
        ignore = np.concatenate(ignore, axis=1)[:, order].astype(bool)
    else:
        img = np.zeros(0, int)
        match = ignore = np.zeros((n_thr, 0), bool)
    return dict(img=img, tp=match & ~ignore, fp=~match & ~ignore, npig=npig,
                rec_thrs=np.asarray(p.recThrs))


def ap(part, weights=None, thr=None):
    """COCO AP (mean over IoU thresholds and 101 recall points) for one class and area range.
    weights[i] = how often image i is in the sample (all ones = the normal COCO result).
    thr = index of one IoU threshold (0 = IoU 0.5) instead of the mean over all ten."""
    n_img = len(part['npig'])
    w_img = np.ones(n_img) if weights is None else np.asarray(weights, float)
    npig = float((w_img * part['npig']).sum())
    if npig == 0:
        return np.nan
    if part['img'].size == 0:
        return 0.0
    w = w_img[part['img']]
    rows = slice(None) if thr is None else slice(thr, thr + 1)
    tp = np.cumsum(part['tp'][rows] * w, axis=1)
    fp = np.cumsum(part['fp'][rows] * w, axis=1)
    rc = tp / npig
    pr = tp / (tp + fp + np.spacing(1))
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
    out = []
    for t in range(rc.shape[0]):
        idx = np.searchsorted(rc[t], part['rec_thrs'], side='left')
        ok = idx < rc.shape[1]
        q = np.zeros(len(idx))
        q[ok] = pr[t, idx[ok]]
        out.append(q.mean())
    return float(np.mean(out))


def bootstrap(parts, n_img, n_boot=1000, seed=0):
    """AP of every part on n_boot resamples of the images (drawn with replacement)."""
    rng = np.random.default_rng(seed)
    res = {key: np.empty(n_boot) for key in parts}
    for b in range(n_boot):
        wts = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
        for key, part in parts.items():
            res[key][b] = ap(part, wts)
    return res


def interval(samples):
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))
