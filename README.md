YOLO-SemBEV
YOLO-SemBEV is a real-time 3D Bird's-Eye-View (BEV) occupancy and object detection framework for autonomous driving. It operates from a single front-facing camera and a single front-facing radar, achieving an end-to-end latency of 21 ms (47+ FPS) on a single NVIDIA RTX A6000 GPU.

Paper: YOLO-SemBEV: Semantic-Guided BEV Occupancy Detection using Single Camera and Radar (2026)
Code: This repository | Weights: Download from Google Drive

Highlights

Recall-first design — foreground F1 of 0.684, Recall of 0.779 at 50 m range

21 ms end-to-end inference — fastest in our comparison

Single camera + single radar — no surround-view rig required

Two variants: YOLO-SemBEV (YOLOv8 backbone) and R-SemBEV (ResNet-101 backbone)

Evaluated on the nuScenes dataset

![Architecture Overview]
(assets/Model_Architecture.png)
YOLO-SemBEV consists of four key components:

YOLOv8 Backbone + PAN Neck — extracts multi-scale image features (P3/P4/P5) from the front camera

Semantically Painted Radar Encoder — projects radar point cloud into image feature space using a 2D segmentation prior, then encodes into BEV

Sparse Semantic Proposal Encoder — lifts 2D segmentation masks into 3D BEV space as spatial anchors for foreground objects

Multi-scale BEV Neck + Detection Head — fuses image BEV, radar BEV, and proposal BEV at three scales (128×128, 64×64, 32×32) for occupancy and 3D object detection

The 2D segmentation prior is provided by a frozen ResNet-UNet model pre-trained on Cityscapes.


Results
Foreground Occupancy (nuScenes val, 50 m range)
Model	Backbone	FG IoU	Pr	Rec	F1	mAP	Latency
YOLO-SemBEV	YOLOv8	0.258	0.609	0.779	0.684	0.328	21 ms
R-SemBEV	ResNet-101	0.231	0.518	0.784	0.624	0.306	26 ms
Inference Latency vs. Baselines
Method	Input	Latency
MonoScene	1C	~870 ms
SurroundOcc	6C	~350 ms
CRN	6C+6R	~47 ms
RCBEVDet	6C+6R	~48 ms
YOLO-SemBEV	1C+1R	21 ms
Repository Structure
text
YOLO-SemBEV/
├── train.py                                  # Main training script
├── src/
│   ├── model_bev_seg12.py                    # YOLO-SemBEV full model (YOLOv8 backbone)
│   ├── model_bev_seg12_resnetbackbone.py     # R-SemBEV variant (ResNet-101 backbone)
│   ├── loss_bev_seg12.py                     # Loss functions (occupancy + detection)
│   └── dataset_loader_seg_large.py           # nuScenes dataloader with radar painting
├── requirements.txt
└── README.md
Installation
bash
# Clone the repository
git clone https://github.com/<your-username>/YOLO-SemBEV.git
cd YOLO-SemBEV

# Create conda environment
conda create -n yolosembev python=3.9 -y
conda activate yolosembev

# Install dependencies
pip install -r requirements.txt
Requirements
Python 3.9+

PyTorch >= 2.0

torchvision

nuscenes-devkit

numpy, opencv-python, tqdm, scipy

Data Preparation
Download the nuScenes dataset (Full dataset v1.0) and organise it as follows:

text
data/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-trainval/
Update the dataset root path in train.py:

python
NUSCENES_ROOT = "/path/to/data/nuscenes"
Pretrained Weights
Model	Backbone	Weights
YOLO-SemBEV	YOLOv8	Download
R-SemBEV	ResNet-101	Download
Segmentation Prior	ResNet-UNet	Download
Place downloaded weights in a weights/ directory:

text
weights/
├── yolosembev.pth
├── rsembev.pth
└── seg_prior.pth
Training
Training proceeds in two stages. Update paths and hyperparameters at the top of train.py before running.

Stage 1 — Warm-up on stratified subset (13,000 samples, 7 epochs):

bash
python train.py --stage 1 \
                --data_root /path/to/data/nuscenes \
                --seg_weights weights/seg_prior.pth \
                --output_dir checkpoints/stage1
Stage 2 — Full training (34,000 samples, 7 epochs):

bash
python train.py --stage 2 \
                --data_root /path/to/data/nuscenes \
                --seg_weights weights/seg_prior.pth \
                --resume checkpoints/stage1/last.pth \
                --output_dir checkpoints/stage2
To train the ResNet-101 variant (R-SemBEV), add --backbone resnet101.

Evaluation
bash
python train.py --eval \
                --data_root /path/to/data/nuscenes \
                --weights weights/yolosembev.pth \
                --seg_weights weights/seg_prior.pth
Citation
If you find this work useful, please cite:

text
@article{yolosembev2026,
  title     = {YOLO-SemBEV: Semantic-Guided BEV Occupancy Detection
               using Single Camera and Radar},
  author    = {<Authors>},
  journal   = {<Venue>},
  year      = {2026}
}
License
This project is released under the MIT License.
