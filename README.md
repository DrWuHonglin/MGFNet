## MGFNet
MGFNet: A Multiscale Gated Fusion Network For Multimodal Semantic Segmentation [[Paper](https://doi.org/10.1007/s00371-025-03912-x)]

<p align="center">
  <img src="https://github.com/DrWuHonglin/MGFNet/blob/main/images/framework.png" width="900" height="450">
</p>

This repository contains the official implementation of MGFNet, a novel network for multimodal semantic segmentation.

- Achieves efficient and precise multimodal remote sensing semantic segmentation
- MGFNet: A dual-stream multimodal semantic segmentation network with a multilevel fusion strategy.
- Introduces the MGF module for extracting multiscale complementary features and adaptively weighting modalities.
- CMI & CMME Modules: The CMI module enables rich cross-modal interactions and long-range dependency modeling, while the CMME module enhances multiscale feature integration for improved segmentation.
  
## Results

1. MGFNet achieves competitive results on the following datasets:
- Vaihingen: 84.18% mIoU
- Potsdam  : 85.87% mIoU
2. We provide visualizations of our results on the Vaihingen and Potsdam datasets:
<p align="center">
  <img src="https://github.com/DrWuHonglin/MGFNet/blob/main/images/vaihingen.png" width="800" height="450">
</p>
<p align="center">
  <img src="https://github.com/DrWuHonglin/MGFNet/blob/main/images/potsdam.png" width="800" height="450">
</p>
Qualitative performance comparisons on the Vaihingen and Potsdam test set. (a) RGB images, (b) DSM, (c) Ground truth, (d) ABCNet, (e) TransUNet, (f) UNetFormer, (g)
MAResU-Net, (h) CMTFNet, (i) vFuseNet, (j) SA-GATE, (k) ESANet, (l) CMGFNet, (m) CMFNet, (n) SGFNet, (o) AsymFormer, and (p) proposed MGFNet. The red boxes are added to all subfigures to highlight the differences.

## Installation
1. Requirements
   
- Python 3.10.15	
- CUDA 12.1
- torch==1.13.0+cu117
- torchvision==0.14.0+cu117
- tqdm==4.66.4
- numpy==1.23.5
- pandas==2.0.1
- ipython==8.12.3

## Demo
To quickly test the MGFNet with randomly generated tensors, you can run the demo.py file. This allows you to verify the model functionality without requiring a dataset.
1. Ensure that the required dependencies are installed:
```
pip install -r requirements.txt
```
2. Run the demo script:
```
python demo.py
```

## Datasets
All datasets including ISPRS Potsdam, ISPRS Vaihingen can be downloaded [here](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets).
