import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from coco_to_yolo import SUBSETS, convert, yolo_line  # noqa: E402


def test_yolo_line_normalises_box():
    # 100x50 box at (100, 50) in a 1000x500 image, class 2
    assert yolo_line([100, 50, 100, 50], 1000, 500, 2) == "2 0.150000 0.150000 0.100000 0.100000"


def test_yolo_line_clips_box_to_image():
    # box runs over the right and bottom edge
    line = yolo_line([900, 400, 200, 200], 1000, 500, 0)
    _, xc, yc, w, h = line.split()
    assert float(xc) + float(w) / 2 <= 1.0
    assert float(yc) + float(h) / 2 <= 1.0


def test_yolo_line_drops_empty_box():
    assert yolo_line([10, 10, 0, 5], 100, 100, 1) is None


def test_convert_writes_labels_and_yaml(tmp_path):
    ann_dir = tmp_path / "rgb"
    ann_dir.mkdir()
    categories = [{"id": 3, "name": "small_load_carrier"}, {"id": 7, "name": "pallet"}]
    for i, file_name in enumerate(SUBSETS):
        data = {
            "images": [{"id": i, "file_name": f"img{i}.jpg", "width": 200, "height": 100}],
            "categories": categories,
            "annotations": [
                {"id": 1, "image_id": i, "category_id": 7, "bbox": [0, 0, 20, 10]},
                {"id": 2, "image_id": i, "category_id": 3, "bbox": [100, 50, 50, 25]},
            ],
        }
        (ann_dir / file_name).write_text(json.dumps(data))

    out = tmp_path / "yolo"
    summary = convert(ann_dir, None, out)

    assert summary["train"]["images"] == 3 and summary["val"]["images"] == 2
    assert summary["train"]["boxes"] == 6 and summary["val"]["boxes"] == 4
    lines = (out / "labels" / "val" / "img0.txt").read_text().split()
    assert lines[0] == "2"  # pallet is class index 2
    assert "names:" in (out / "loco.yaml").read_text()
