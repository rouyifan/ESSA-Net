# ESSA-Net

## ESSA-Net: Virtual-Time Spiking Attention for Robust RGB–Infrared Object Detection

**Zihan Yuan, Rouyi Fan, Zhihua Liu, Yuhong Zhang, Chunhua Zhu**

**Publication:** [venue, year, paper link and DOI to be added upon publication]

[![Python](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue)](LICENSE)

ESSA-Net (Enhanced Spiking Self-Attention Network) is a two-stream detector for paired visible and infrared imagery. It addresses modality-dependent challenges such as poor visible-light contrast and infrared background clutter through cross-modal feature conditioning, virtual-step spiking attention, and difficulty-aware classification. The virtual time steps are internal computations on static paired images, not video frames.

This repository contains the full-model implementation and training and evaluation entry points. Dataset images, annotations and trained checkpoints are not included.

## Overall Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Overall architecture of ESSA-Net" width="95%">
</p>

RGB and infrared images are processed by separate feature-extraction branches. RIFusion exchanges information between the branches, and corresponding deeper features are combined and refined with ESSA before multi-scale detection.

## Core Contributions

### RIFusion

RIFusion concatenates paired features, applies channel recalibration, and mixes and shuffles channels to support RGB–IR interaction. Its output retains the concatenated channel count and is split back into two branches.

<p align="center">
  <img src="assets/rifusion.png" alt="RIFusion feature interaction mechanism" width="95%">
</p>

### ESSA

ESSA applies leaky integrate-and-fire (LIF) encoding over virtual steps and forms attention from temporally averaged spike representations. It is designed to filter fused features before detection.

<p align="center">
  <img src="assets/essa.png" alt="ESSA virtual-step spiking attention illustration from the manuscript" width="95%">
</p>

**Implementation note:** The current manuscript figure depicts Q/K/V projections after spike encoding, whereas the released code projects Q/K/V before applying LIF encoding to each branch. The illustration requires revision to match the implementation.

### ATFL

ATFL uses confidence-dependent modulation in the classification loss to adjust the contribution of samples with different prediction difficulties. The current manuscript and implementation call this **Adaptive-Threshold Focal Loss**; the alternative name “Adaptive Task Focusing Loss” requires confirmation before publication. A standalone ATFL figure is not currently available.

## Main Results on Three Datasets

The following are values reported in the manuscript. Metrics are percentages; mAP50 is mean AP at IoU 0.50, and mAP50–95 is mean AP over IoU thresholds 0.50–0.95. The baseline comparison below is reported only for DroneVehicle; corresponding baseline results for the other two datasets are not specified in the manuscript tables. Published comparison results may use different evaluation protocols.

| Dataset | Model | Input | mAP50 (%) | mAP50–95 (%) | Protocol in manuscript |
|:--|:--|:--|--:|--:|:--|
| DroneVehicle | Two-stream baseline | RGB + IR | 81.2 | — | Test set; runtime comparison |
| DroneVehicle | **ESSA-Net** | RGB + IR | **83.1** | — | Test set |
| LLVIP | **ESSA-Net** | RGB + IR | **98.7** | **72.3** | Paper's reported comparison protocol |
| M3FD | **ESSA-Net** | RGB + IR | **82.7** | — | Paper's reported comparison protocol |

The manuscript notes inconsistent M3FD protocols across prior studies. Exact split manifests and trained checkpoints for these results are not included in this release; see the publication for the reported comparison settings.

## Ablation Studies

The component ablation below is reported on the DroneVehicle **validation** set. Values are percentages; R is recall. These results are distinct from the test-set comparison above. An enabled component is marked ✓.

| RIFusion | ESSA | ATFL | mAP50 (%) | mAP50–95 (%) | R (%) |
|:--:|:--:|:--:|--:|--:|--:|
| — | — | — | 82.8 | 62.3 | 77.9 |
| — | ✓ | — | 84.7 | 63.7 | 79.5 |
| ✓ | — | — | 83.3 | 63.1 | 78.3 |
| — | — | ✓ | 83.4 | 62.2 | 79.4 |
| ✓ | ✓ | — | 85.3 | 64.5 | 80.2 |
| — | ✓ | ✓ | 84.5 | 63.8 | 79.7 |
| ✓ | — | ✓ | 85.6 | 64.2 | 80.3 |
| ✓ | ✓ | ✓ | **86.6** | **65.6** | **81.4** |

The full configuration achieves the highest reported mAP50 in this table; partial combinations are not uniformly additive.

### Virtual time steps

Detection scores below are from the manuscript's DroneVehicle validation-set sensitivity table. T=6 is the reported main setting.

| Virtual steps T | mAP50 (%) | mAP50–95 (%) | Forward latency (ms) |
|--:|--:|--:|--:|
| 1 | 85.8 | 64.8 | 16.2 |
| 2 | 85.3 | 64.3 | 17.8 |
| 4 | 85.8 | 64.7 | 22.1 |
| 6 | **86.6** | **65.6** | 26.7 |
| 8 | 84.9 | 63.9 | 30.8 |

Latency was measured on an NVIDIA RTX 4090 using batch 1, FP16, 640 × 512 input, and 16 CPU threads. It measures synchronized model forward time and excludes image loading, preprocessing, host-to-device transfer, and NMS. Timing and detection scores are separate evaluations.

### Encoding controls

The manuscript compares ESSA with two static encoding controls on DroneVehicle validation:

| Encoding | mAP50 (%) | mAP50–95 (%) |
|:--|--:|--:|
| Single-step LIF (T=1) | 85.8 | 64.8 |
| Static sigmoid gating | 85.9 | 64.6 |
| **Multi-step LIF (T=6)** | **86.6** | **65.6** |

The ablation configurations and their checkpoints are not provided as separate runnable configurations in this release.

## Installation and Environment Preparation

The tested environment used Python 3.8.19, PyTorch 2.3.1, torchvision 0.18.1, CUDA runtime 11.8, SpikingJelly 0.0.0.0.14, and timm 1.0.19. Install the PyTorch build appropriate for your system, then install this repository:

```bash
git clone https://github.com/rouyifan/ESSA-Net.git
cd ESSA-Net
python -m pip install -r requirements.txt
```

`requirements.txt` installs the local package and dependencies declared in `pyproject.toml`. SpikingJelly is required to reproduce the LIF behavior: the fallback used when it is absent implements different dynamics. The repository includes a modified Ultralytics package; run commands from this checkout rather than substituting an unrelated installed version.

## Dataset Preparation

Obtain the datasets separately and comply with their respective terms of use:

| Dataset | Task | Dataset source |
|:--|:--|:--|
| DroneVehicle | Aerial vehicle detection | [VisDrone/DroneVehicle](https://github.com/VisDrone/DroneVehicle) |
| LLVIP | Low-light pedestrian detection | [BUPT LLVIP](https://github.com/bupt-ai-cz/LLVIP) |
| M3FD | Multi-class RGB–IR detection | [TarDAL / M3FD](https://github.com/JinyuanLiu-CV/TarDAL) |

For the supplied DroneVehicle configuration, prepare aligned RGB/IR image pairs and YOLO-format detection labels:

```text
DroneVehicle/
├── images/{train,val,test}/   # visible images
├── image/{train,val,test}/    # infrared images
└── labels/{train,val,test}/   # normalized class, x_center, y_center, width, height
```

RGB and IR images must have matching filenames and dimensions. Each label file must have the same stem as its paired images. Convert the source annotations to the required YOLO format and follow the relevant dataset split and preprocessing protocol; conversion scripts and exact manuscript split manifests for all three datasets are not included. DroneVehicle uses the class order `car`, `truck`, `bus`, `van`, `freight_car` in the supplied configuration. LLVIP and M3FD require separate dataset YAML files with the correct paths and class names; their processed data and evaluation configurations are not supplied.

Copy `configs/dronevehicle.example.yaml` and replace `/path/to/DroneVehicle` with the absolute path to the prepared data. This implementation expects the directory names `images` and `image`. Verify that `test` and `test_ir` both refer to the test split before evaluation.

## Training and Testing

### Training

```bash
python train.py --data /path/to/dronevehicle.yaml --device 0
```

The entry point constructs the full model from `configs/essa_net.yaml`. Its defaults are 260 epochs, batch size 12, image size 640, `copy_paste=0.1`, and `close_mosaic=5`; these are the released script defaults, not a guarantee that they reproduce every manuscript table. Checkpoints are saved under `runs/`.

Available overrides include `--model`, `--epochs`, `--batch`, `--imgsz`, and `--device`. Run `python train.py --help` for details.

### Testing

```bash
python test.py --weights /path/to/best.pt --data /path/to/dronevehicle.yaml \
  --split test --batch 12 --imgsz 640 --device 0
```

Use `--split val` for validation. The released `--imgsz 640` is the code's evaluation setting; the separate latency benchmark above uses 640 × 512 inputs. Run `python test.py --help` for the available options.

### Checkpoints

Pretrained model weights are not currently distributed. Train a model or supply a compatible checkpoint to `--weights` for testing. A checkpoint link and a verified reproduction recipe will be added when available.

## License

The project includes code derived from Ultralytics YOLO and is distributed under the [GNU Affero General Public License v3.0](LICENSE). Third-party packages and datasets are subject to their own licenses and terms; consult their respective repositories before use or redistribution.

## Acknowledgements

This implementation builds on [Ultralytics](https://github.com/ultralytics/ultralytics) and [TwoStream-YOLOv8](https://github.com/mujianyu/TwoStream_Yolov8). The spiking components use [SpikingJelly](https://github.com/fangwei123456/spikingjelly). We thank the authors of DroneVehicle, LLVIP, and M3FD for making their resources available.

## Citation

The manuscript's publication details are not yet available. Replace the following placeholders with the final venue, year, and DOI when the paper is published:

```bibtex
@article{yuan_essa_net,
  title   = {ESSA-Net: Virtual-Time Spiking Attention for Robust RGB--Infrared Object Detection},
  author  = {Yuan, Zihan and Fan, Rouyi and Liu, Zhihua and Zhang, Yuhong and Zhu, Chunhua},
  journal = {[VENUE TO BE ADDED]},
  year    = {[YEAR TO BE ADDED]},
  doi     = {[DOI TO BE ADDED]}
}
```
