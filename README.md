# ESSA-Net

This repository contains the two-stream RGB/infrared detection model, training entry point, and evaluation code for ESSA-Net. The implementation builds on Ultralytics YOLOv8 and TwoStream-YOLOv8. The dataset and trained model weights are not included.

## Setup

Use Python 3.8 and a CUDA-enabled PyTorch installation appropriate for your system. The experiment environment used PyTorch 2.3.1, torchvision 0.18.1, SpikingJelly 0.0.0.0.14 and timm 1.0.19.

```bash
pip install -e .
```

SpikingJelly is required to reproduce the LIF neuron behavior; the code contains a fallback with different dynamics when it is absent.

## Dataset

Prepare paired DroneVehicle images with matching filenames and YOLO-format labels:

```text
DroneVehicle/
  images/{train,val,test}/  # visible RGB images
  image/{train,val,test}/   # infrared images
  labels/{train,val,test}/  # shared YOLO labels
```

Copy `configs/dronevehicle.example.yaml`, replace `/path/to/DroneVehicle` with an absolute path, and verify every RGB/IR image pair and label. The data are not distributed here.

## Training

```bash
python train.py --data /path/to/dronevehicle.yaml --device 0
```

This initializes the full model from its architecture YAML. Defaults follow the project's training entry point: batch 12, 260 epochs, image size 640, copy-paste 0.1 and close-mosaic 5. Checkpoints are written under `runs/`.

## Evaluation

```bash
python test.py --weights /path/to/best.pt --data /path/to/dronevehicle.yaml --split test --batch 12 --imgsz 640 --device 0
```

Use `--split val` for validation. Checkpoints must be produced by this implementation or supplied separately.

## License and attribution

The underlying Ultralytics code is distributed under AGPL-3.0; see `LICENSE`. TwoStream-YOLOv8: https://github.com/mujianyu/TwoStream_Yolov8 .
