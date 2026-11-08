# ESSA-Net

PyTorch training and evaluation code for ESSA-Net, a two-stream RGB/infrared object detector. The model combines RIFusion, Enhanced Spiking Self-Attention (ESSA), and Adaptive-Threshold Focal Loss (ATFL). The implementation builds on Ultralytics YOLOv8 and TwoStream-YOLOv8.

The repository provides the full-model implementation and commands for training and evaluation. DroneVehicle images, annotations and trained checkpoints are not included.

## Repository structure

```text
.
├── configs/
│   ├── essa_net.yaml                  # Full model architecture
│   └── dronevehicle.example.yaml      # Paired dataset paths and classes
├── ultralytics/                       # Modified two-stream YOLO implementation
├── train.py                           # Training entry point
├── test.py                            # Evaluation entry point
├── pyproject.toml                     # Package and dependencies
└── LICENSE                            # AGPL-3.0 license
```

The principal model and loss implementations are in `ultralytics/nn/modules/block.py` and `ultralytics/utils/loss.py`; dual-stream routing is in `ultralytics/nn/tasks.py`.

## Setup

Use Python 3.8 and a CUDA-enabled PyTorch installation appropriate for your system. The experiment environment used PyTorch 2.3.1, torchvision 0.18.1, SpikingJelly 0.0.0.0.14 and timm 1.0.19.

```bash
pip install -e .
```

SpikingJelly is required to reproduce the LIF neuron behavior; the code contains a fallback with different dynamics when it is absent.

## Data preparation

Prepare paired DroneVehicle images with matching filenames and YOLO-format labels:

```text
DroneVehicle/
  images/{train,val,test}/  # visible RGB images
  image/{train,val,test}/   # infrared images
  labels/{train,val,test}/  # shared YOLO labels
```

The five classes are `car`, `truck`, `bus`, `van`, and `freight_car`. RGB and IR pairs must have matching filenames and image dimensions, and labels must use normalized YOLO bounding-box format. Obtain and prepare the dataset separately; it is not distributed here.

Copy `configs/dronevehicle.example.yaml`, replace `/path/to/DroneVehicle` with an absolute path, and verify all six RGB and IR paths. This implementation uses the directory names `images` for RGB and `image` for IR. Make sure `test` and `test_ir` point to the test split when evaluating it.

## Training

```bash
python train.py --data /path/to/dronevehicle.yaml --device 0
```

This initializes the full model from its architecture YAML. Defaults follow the project's training entry point: batch 12, 260 epochs, image size 640, copy-paste 0.1 and close-mosaic 5. Checkpoints are written under `runs/`. Use `python train.py --help` for the available command-line options.

## Evaluation

```bash
python test.py --weights /path/to/best.pt --data /path/to/dronevehicle.yaml --split test --batch 12 --imgsz 640 --device 0
```

Use `--split val` for validation. Checkpoints must be produced by this implementation or supplied separately. Use `python test.py --help` for the available command-line options.

## Model weights

Pretrained weights are not included in the current release. Evaluation requires a compatible checkpoint supplied separately or obtained through training.

## License and attribution

The underlying Ultralytics code is distributed under AGPL-3.0; see `LICENSE`. TwoStream-YOLOv8: https://github.com/mujianyu/TwoStream_Yolov8 .
