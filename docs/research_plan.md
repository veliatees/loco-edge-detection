# Research plan

## Question

When a detector is compressed for edge hardware (INT8 weights, lower input resolution), is the accuracy loss spread evenly, or does it fall on rare classes and small objects?

## What is already known

- Hooker et al., "Characterising Bias in Compressed Models" (https://arxiv.org/abs/2010.03058): pruning and quantization barely change overall accuracy, but the errors concentrate on a small set of examples, and these come disproportionately from underrepresented groups. The study is image classification on CelebA.
- Hooker et al., "What Do Compressed Deep Neural Networks Forget?" (https://arxiv.org/abs/1911.05248): introduces Compression Identified Exemplars, the examples where a compressed model and the original model disagree.
- Cygert and Czyzewski, "Robustness in Compressed Neural Networks for Object Detection" (https://arxiv.org/abs/2102.05509): pruning Faster R-CNN on driving datasets hurts rare classes (train, truck, bus) the most. Quantization is left as future work.

The open part for this project: post-training INT8 quantization and lower input resolution of current real-time detectors, one CNN and one transformer-based model, on an imbalanced industrial dataset, measured per class and per object size.

## Hypotheses (written before the results)

- H1: INT8 and the smaller input size reduce AP most for the rarest class (forklift, 0.4% of boxes) and for small objects (under 32x32 px), while mAP over all classes changes little.
- H2: RT-DETR loses more than YOLO11n under static INT8 quantization. ONNX Runtime recommends static quantization for CNNs and dynamic quantization for transformers.
- H3: The calibration set matters. A calibration set with more images of rare classes reduces the INT8 loss for those classes, compared with a random calibration set of the same size.

## Method

- Metrics with pycocotools: AP per class and per object size (small, medium, large), plus precision and recall at a fixed confidence threshold.
- Uncertainty: 95% bootstrap intervals over val images (1,000 resamples). Forklift has only 124 val boxes, so differences there need intervals.
- Training noise: YOLO11n trained with 3 seeds, so a drop from compression can be compared with the normal spread between runs.
- Disagreement analysis: images where the INT8 model's detections differ most from the FP32 model (the idea of Compression Identified Exemplars, applied to detection), with a grid of examples.

## Output

Tables and figures in this repository and a short technical report.
