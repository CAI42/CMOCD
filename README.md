# CMOCD: Cross-View Multimodal Object-Level Change Detection for Small Vehicles

Official implementation of the paper "Cross-View Visible-Thermal Object-Level Change Detection for Small Vehicles" (IEEE JSTARS 2026).
This repository provides the CVOCD dataset (first cross-view visible-thermal object-level change detection dataset for small vehicles) and the CMOCD framework (cross-view multimodal object-level change detection method) based on YOLOv11.

🔍 Overview
Core Contributions

1. CVOCD Dataset: First public dataset for cross-view visible-thermal object-level change detection, focusing on small vehicles (cars, buses, trucks) with 2723 image pairs and 155k+ annotated objects.

2. CMOCD Framework: Integrates two key modules to address cross-view misalignment and multimodal feature discrepancy:

  - CFFA Module: Coarse-to-fine feature alignment for spatial consistency.

  - VTFF Module: Decoupled channel-spatial attention fusion for multimodal complementarity.

📊 Dataset (CVOCD)
Dataset Details
  -Image Pairs: 2723 (visible-thermal bitemporal pairs)
  -Image Resolution: Thermal: 1280×1024; Visible: 4000×3000
  -Annotated Objects: 155,277 vehicles (91,363 unchanged; 63,914 changed)
  -Split Ratio: Train: 1906 pairs; Val: 409 pairs; Test: 408 pairs
  -Illumination Scenarios: Nearly no difference (1480 pairs), Slight difference (708 pairs), Significant difference (535 pairs)
  -Collection Sites: Changchun, Luoyang, Fuyang (China) – covers urban/rural traffic scenes
Dataset Links:[Quark Netdisk] [Baidu Netdisk]
Dataset Structure:
  CVOCD/
    ├── train/
    │   ├── infrared/          # Thermal images (temporal 1)
    │   │   ├── images/
    │   │   │   ├── CC_01_02_0007.JPG
    │   │   │   └── ...
    │   │   └── labels/        # YOLOv5 format annotations located on infrared images(bounding box + change/unchanged type)
    │   │   │   ├── CC_01_02_0007.txt
    │   │   │   └── ...
    │   ├── visible/          # Visible images (temporal 2)
    │   │   ├── images/
    │   │   │   ├── CC_01_02_0007.JPG
    │   │   │   └── ...
    │   │   └── labels/        # YOLOv5 format annotations located on infrared images(bounding box + change/unchanged type)
    │   │   │   ├── CC_01_02_0007.txt
    │   │   │   └── ...
    ├── val/                  # Same structure as train
    └── test/                 # Same structure as train

🚀 Installation
Prerequisites
- Python 3.8+
- PyTorch 1.18+
- CUDA 11.6+ (for GPU acceleration)
- Other dependencies:
  pip install -r requirements.txt

Clone the Repository
  git clone https://github.com/CAI42/CMOCD.git
  cd CMOCD

📝 Training
python train.py

📝 Evaluation
python detect-6C.py

📚 Citation
If you use the CVOCD dataset or CMOCD framework in your research, please cite our paper:
@article{cai2026cross,
  title={Cross-View Visible-Thermal Object-Level Change Detection for Small Vehicles},
  author={Cai, Luyang and Sun, He and Yang, Hao and Zhao, Zhuxin and Ni, Li and Gao, Lianru},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={19},
  pages={4443--4456},
  year={2026},
  publisher={IEEE},
  doi={10.1109/JSTARS.2026.3653040}
}


📋 License
- Dataset: The CVOCD dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Commercial use is prohibited.

🤝 Contact
For questions about the dataset or code, please contact:
- Luyang Cai: cailuyang21@mails.ucas.ac.cn
- He Sun (corresponding author): sunhe@aircas.ac.cn

Acknowledgements
This work was supported by the National Natural Science Foundation of China (Grant 62571514, 62301534, 42325104) and the Science and Disruptive Technology Program, AIRCAS (Grant 2025-AIRCAS-SDPT-17). We thank the team for data collection and annotation support.


