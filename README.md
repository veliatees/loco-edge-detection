# Warehouse object detection on small hardware

Which warehouse objects does a detector stop seeing first when it is made small and fast enough for a cheap device?

Robots and cameras in warehouses rarely have a large GPU. To run a detector on a CPU or an edge board, the model has to become smaller: fewer parameters, lower input resolution, 8-bit weights. Each of these steps costs some accuracy, but the cost is not the same for every object. An average score hides this. This project measures it per class.

I train two detectors on real warehouse images, a small CNN (YOLO11n) and a transformer-based detector (RT-DETR), find where they fail, and then make them smaller step by step. After each step I measure what it cost, for each object class and each object size.

## Data

LOCO (Logistics Objects in Context) from the Chair of Materials Handling, Material Flow, Logistics at TU Munich. The images were taken in the production and logistics areas of industrial partners. License: CC0.

- Annotations (COCO format): https://github.com/tum-fml/loco
- Annotated images: https://go.mytum.de/239870

The images are not part of this repository.

What the annotation files actually contain (`python scripts/loco_stats.py loco/rgb`):

| Class | Boxes | Share | Smaller than 32x32 px | Train | Val |
|---|---:|---:|---:|---:|---:|
| pallet | 120,445 | 79.5% | 28.0% | 68,148 | 52,297 |
| small_load_carrier | 22,151 | 14.6% | 30.3% | 13,240 | 8,911 |
| stillage | 5,407 | 3.6% | 3.1% | 3,400 | 2,007 |
| pallet_truck | 2,827 | 1.9% | 2.9% | 1,674 | 1,153 |
| forklift | 598 | 0.4% | 0.3% | 474 | 124 |

- 5,097 images and 151,428 boxes. The LOCO README says 5,593 images and 152,421 annotations, but the JSON files contain fewer.
- Train: subsets 2, 3 and 5 (2,820 images). Val: subsets 1 and 4 (2,277 images). This is the split given by the file names.
- A median image has 20 boxes; the busiest one has 345. The default detection limit in Ultralytics is 300 per image, so all runs here use `max_det=500`.
- 26.9% of all boxes are smaller than 32x32 pixels.
- Subset 4 has no small load carriers, so the val score for that class comes from subset 1 only.
- The images come in five resolutions, from 640x480 to 1920x1080.

What this means before training anything: the forklift score will be unstable, because there are only 124 forklifts in val. When the input resolution goes down, pallets and small load carriers should suffer first, because almost a third of them are already tiny.

Published results to compare with: the LOCO paper (Mayershofer et al., ICMLA 2020) reports 41% mAP. A replication study reports 64.54% mAP and names forklift and pallet truck as the weakest classes (https://arxiv.org/abs/2209.13499). Their evaluation settings are not exactly the same as mine, so these numbers are a rough reference, not a direct comparison.

## Plan

**Week 1: baseline.** Convert the labels and train YOLO11n (2.6 million parameters) at 960 px on a Kaggle GPU.
Done when: mAP50-95 per class on val is in this README and the run is saved.

**Week 2: second model and error analysis.** Train RT-DETR-l (33 million parameters, transformer-based, no NMS step) on the same split. Evaluate both models with my own script (pycocotools): AP per class, AP per object size, and a grid of typical failures.
Done when: one table, one figure and three written findings.

**Week 3: make it small.** Export to ONNX. Quantize to INT8 with ONNX Runtime. Compare input sizes 640 and 960. Measure latency on a laptop CPU (and on a Raspberry Pi if available). Run the ONNX model from C++ with ONNX Runtime. This builds on [pytorch-to-cpp-inference](https://github.com/veliatees/pytorch-to-cpp-inference), a smaller project where I export a PyTorch network's weights and load them in C++.
Done when: the results table below has latency, size and AP per class for every version.

**Week 4: ship.** FastAPI endpoint and Dockerfile, pytest for the converter and the evaluation, GitHub Actions, a short write-up.
Done when: `docker run` serves predictions and CI is green.

Current step: week 2. Both models are trained (results below); next are a YOLO11n run at 640 px, two more seeds, and the per-size evaluation with pycocotools.

## Results

The table is filled in as runs finish. AP values are AP50-95 on the val split. CPU latency and size are measured in week 3 on the exported ONNX models, so all versions are measured the same way.

| Model | Weights | Input | mAP50-95 | pallet | small_load_carrier | stillage | pallet_truck | forklift | CPU ms/image | Size MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | FP32 | 960 | 0.164 | 0.267 | 0.159 | 0.251 | 0.058 | 0.085 | | |
| YOLO11n | INT8 | 960 | | | | | | | | |
| YOLO11n | INT8 | 640 | | | | | | | | |
| RT-DETR-l | FP32 | 640 | 0.215 | 0.262 | 0.195 | 0.372 | 0.111 | 0.135 | | |

### Week 1: what the baseline shows

YOLO11n, 960 px, best epoch 35 of 65 (early stopping with patience 30), 1.6 hours on one Kaggle T4 GPU. Per-class table: `results/yolo11n_fp32_960.csv`, training log: `results/train_log.csv`.

| Class | Val boxes | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| pallet | 52,297 | 0.588 | 0.646 | 0.637 | 0.267 |
| stillage | 2,007 | 0.624 | 0.419 | 0.461 | 0.251 |
| small_load_carrier | 8,911 | 0.571 | 0.369 | 0.391 | 0.159 |
| forklift | 124 | 0.211 | 0.258 | 0.176 | 0.085 |
| pallet_truck | 1,153 | 0.456 | 0.147 | 0.178 | 0.058 |
| all | 64,492 | 0.490 | 0.368 | 0.369 | 0.164 |

1. **Overall level.** mAP50 is 0.369. For reference, the LOCO paper reports 22.1% mAP50 for YOLOv4-tiny and 41.0% for YOLOv4, and the replication study reports 49.98% (YOLOv4-tiny) and 64.54% (YOLOv4), evaluated on subset 1 only with subset 5 added to training. The splits differ, so this is a rough comparison.
2. **Pallet truck is the weakest class, not forklift.** It has 1,153 val boxes but only 0.147 recall. The confusion matrix (`results/confusion_matrix_normalized.png`) shows that 27% of forklifts are predicted as pallet trucks and 19% of pallet trucks as forklifts. The replication study describes the same confusion and notes that manual and electric pallet trucks share one class.
3. **Most false detections are pallets.** 81% of the detections that match no labelled object are predicted as pallet. Open question for week 2: how many of them are pallets that were not labelled?
4. **The model stops transferring early.** After epoch 35 the training loss kept falling, but val mAP50-95 only moved between 0.14 and 0.16. The val subsets come from other recording sessions than the train subsets, so the model gets better at the training scenes without getting better at new ones. The jumps of about 0.01 from one epoch to the next are also why the comparisons later need several seeds and confidence intervals.

### Week 2: RT-DETR-l against YOLO11n

RT-DETR-l, 640 px, best epoch 14 of 34 (early stopping with patience 20), 2.3 hours on one Kaggle T4 GPU. Files: `results/rtdetr_l_640/`.

| Class | YOLO11n 960 px AP50-95 | RT-DETR-l 640 px AP50-95 | RT-DETR-l recall |
|---|---:|---:|---:|
| pallet | 0.267 | 0.262 | 0.630 |
| stillage | 0.251 | 0.372 | 0.581 |
| small_load_carrier | 0.159 | 0.195 | 0.414 |
| forklift | 0.085 | 0.135 | 0.323 |
| pallet_truck | 0.058 | 0.111 | 0.233 |
| all | 0.164 | 0.215 | 0.436 |

1. **RT-DETR-l is better overall, but not on pallets.** mAP50 rises from 0.369 to 0.448. The gains are in stillages, pallet trucks and forklifts. Pallets stay at the same AP, and the confusion matrix shows more missed pallets (13% instead of 8%). This fits the lower input size, since many pallets are tiny. The comparison changes two things at once, though: the model (33 million instead of 2.6 million parameters) and the input size (640 instead of 960 px). The next run, YOLO11n at 640 px (`notebooks/03_yolo11n_640_and_seeds.ipynb`), separates the two.
2. **The forklift and pallet truck confusion becomes one-sided.** RT-DETR-l predicts 43% of the forklifts as pallet trucks (YOLO11n: 27%), but only 7% of the pallet trucks as forklifts (YOLO11n: 19%). The training split has 1,674 pallet trucks and 474 forklifts. One possible reason is that the model leans towards the more frequent of two similar classes; this needs a closer look at the images.
3. **The early peak is stronger.** Val mAP50-95 peaked at epoch 14 (0.215) and then fell to 0.17-0.18 while the training loss kept falling.

## Repository layout

```
scripts/coco_to_yolo.py        LOCO annotations (COCO JSON) -> YOLO label files and dataset YAML
scripts/loco_stats.py          counts boxes per class, split and size
notebooks/01_baseline_yolo11n.ipynb   week 1, runs on Kaggle
notebooks/02_rtdetr.ipynb             week 2, runs on Kaggle
notebooks/03_yolo11n_640_and_seeds.ipynb  YOLO11n at 640 px and two more seeds at 960 px
tests/                         pytest for the converter
results/                       per-class tables, training logs and plots (YOLO11n week 1 at the top level, other runs in subfolders)
docs/research_plan.md          question, related work and hypotheses, written before the results
```

## Run

**Training on Kaggle:** upload the LOCO images as a private Kaggle dataset (a zip is fine, the notebook unpacks archives). Import a notebook from `notebooks/`, set Accelerator to GPU and Internet to on, add the image dataset as input, and use Save Version > Save & Run All. The notebook clones the annotations, converts them, trains the model and writes a per-class table to `out/`. Kaggle asks for phone verification before it allows GPU and internet.

**Locally:**

```bash
pip install -r requirements.txt
git clone https://github.com/tum-fml/loco
python scripts/loco_stats.py loco/rgb
python scripts/coco_to_yolo.py --ann-dir loco/rgb --images-root path/to/images --out data/loco_yolo
yolo detect train model=yolo11n.pt data=data/loco_yolo/loco.yaml imgsz=960 epochs=100 batch=16 max_det=500
pytest
```

On Windows, add `--copy` to the converter, because creating symlinks needs extra rights there.

## License

Ultralytics is licensed under AGPL-3.0, so this repository uses AGPL-3.0 as well. The LOCO data is CC0.
