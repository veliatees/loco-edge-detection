"""Convert the LOCO annotations (COCO format) to the YOLO format.

Usage:
    python scripts/coco_to_yolo.py --ann-dir path/to/loco/rgb \
        --images-root path/to/downloaded/images --out data/loco_yolo

Result:
    data/loco_yolo/images/train/*.jpg   (links or copies of the original images)
    data/loco_yolo/labels/train/*.txt   (one line per box: class x_center y_center w h, 0..1)
    data/loco_yolo/images/val/..., data/loco_yolo/labels/val/...
    data/loco_yolo/loco.yaml            (dataset file for Ultralytics)

The images are found by file name anywhere below --images-root, so the
folder layout of the download does not matter.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

SUBSETS = {
    "loco-sub1-v1-val.json": "val",
    "loco-sub2-v1-train.json": "train",
    "loco-sub3-v1-train.json": "train",
    "loco-sub4-v1-val.json": "val",
    "loco-sub5-v1-train.json": "train",
}

# LOCO category id -> YOLO class index
CLASSES = {3: "small_load_carrier", 5: "forklift", 7: "pallet", 10: "stillage", 11: "pallet_truck"}
INDEX = {cat_id: i for i, cat_id in enumerate(CLASSES)}


def yolo_line(bbox, width, height, cls):
    """COCO box [x, y, w, h] in pixels -> YOLO line, clipped to the image."""
    x, y, w, h = bbox
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    xc = (x0 + x1) / 2 / width
    yc = (y0 + y1) / 2 / height
    return f"{cls} {xc:.6f} {yc:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"


def place_image(src, dst, copy):
    if dst.exists() or dst.is_symlink():
        return
    if not copy:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:  # e.g. Windows without symlink rights
            pass
    shutil.copy2(src, dst)


def convert(ann_dir, images_root, out, copy=False):
    index = {}
    if images_root is not None:
        for p in images_root.rglob("*.jpg"):
            index[p.name] = p

    summary = {}
    for file_name, split in SUBSETS.items():
        data = json.loads((ann_dir / file_name).read_text())
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        lines = {im["id"]: [] for im in data["images"]}
        info = {im["id"]: im for im in data["images"]}
        skipped = 0
        for a in data["annotations"]:
            im = info[a["image_id"]]
            line = yolo_line(a["bbox"], im["width"], im["height"], INDEX[a["category_id"]])
            if line is None:
                skipped += 1
            else:
                lines[a["image_id"]].append(line)

        missing = 0
        for image_id, rows in lines.items():
            name = info[image_id]["file_name"]
            (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(rows) + "\n")
            if images_root is not None:
                if name in index:
                    place_image(index[name], img_dir / name, copy)
                else:
                    missing += 1

        s = summary.setdefault(split, {"images": 0, "boxes": 0, "skipped": 0, "missing": 0})
        s["images"] += len(lines)
        s["boxes"] += sum(len(r) for r in lines.values())
        s["skipped"] += skipped
        s["missing"] += missing

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES.values()))
    (out / "loco.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=Path, required=True, help="the rgb/ folder of the LOCO repository")
    parser.add_argument("--images-root", type=Path, help="folder with the downloaded LOCO images")
    parser.add_argument("--out", type=Path, default=Path("data/loco_yolo"))
    parser.add_argument("--copy", action="store_true", help="copy images instead of linking them")
    args = parser.parse_args()

    summary = convert(args.ann_dir, args.images_root, args.out, args.copy)
    for split, s in summary.items():
        print(f"{split}: {s['images']} images, {s['boxes']} boxes, "
              f"{s['skipped']} empty boxes skipped, {s['missing']} images not found")
    print(f"dataset file: {args.out / 'loco.yaml'}")


if __name__ == "__main__":
    main()
