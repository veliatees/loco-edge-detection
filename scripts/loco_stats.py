"""Count what is really inside the LOCO annotation files.

Usage:
    python scripts/loco_stats.py path/to/loco/rgb

Prints images and boxes per subset, boxes per class, and how many boxes
are small, medium or large (COCO size limits: 32x32 and 96x96 pixels).
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

# LOCO's own split, as named in the annotation files
SUBSETS = {
    "loco-sub1-v1-val.json": "val",
    "loco-sub2-v1-train.json": "train",
    "loco-sub3-v1-train.json": "train",
    "loco-sub4-v1-val.json": "val",
    "loco-sub5-v1-train.json": "train",
}


def size_bucket(w, h):
    area = w * h
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ann_dir", type=Path, help="the rgb/ folder of the LOCO repository")
    args = parser.parse_args()

    per_class = collections.Counter()
    per_split = {"train": collections.Counter(), "val": collections.Counter()}
    sizes = collections.defaultdict(collections.Counter)
    images = collections.Counter()
    resolutions = collections.Counter()
    boxes_per_image = []

    for file_name, split in SUBSETS.items():
        data = json.loads((args.ann_dir / file_name).read_text())
        names = {c["id"]: c["name"] for c in data["categories"]}
        images[split] += len(data["images"])
        count = collections.Counter(a["image_id"] for a in data["annotations"])
        boxes_per_image += [count[im["id"]] for im in data["images"]]
        resolutions.update((im["width"], im["height"]) for im in data["images"])

        subset = collections.Counter()
        for a in data["annotations"]:
            name = names[a["category_id"]]
            _, _, w, h = a["bbox"]
            per_class[name] += 1
            per_split[split][name] += 1
            sizes[name][size_bucket(w, h)] += 1
            subset[name] += 1
        print(f"{file_name:26s} {split:5s} {len(data['images']):5d} images "
              f"{len(data['annotations']):6d} boxes  {dict(subset)}")

    total = sum(per_class.values())
    print(f"\nimages: train {images['train']}, val {images['val']}, "
          f"total {sum(images.values())}")
    print(f"boxes: {total}")
    print(f"boxes per image: median {statistics.median(boxes_per_image)}, "
          f"max {max(boxes_per_image)}")
    print("most common resolutions:", resolutions.most_common(5))

    print(f"\n{'class':20s} {'boxes':>7s} {'share':>6s} {'small':>6s} {'medium':>7s} "
          f"{'large':>6s} {'train':>7s} {'val':>6s}")
    for name, n in per_class.most_common():
        s = sizes[name]
        print(f"{name:20s} {n:7d} {n / total:6.1%} {s['small'] / n:6.1%} "
              f"{s['medium'] / n:7.1%} {s['large'] / n:6.1%} "
              f"{per_split['train'][name]:7d} {per_split['val'][name]:6d}")
    small = sum(s["small"] for s in sizes.values())
    print(f"\nsmall boxes overall: {small / total:.1%}")


if __name__ == "__main__":
    main()
