# DA6401 Assignment 2 - Multitask Vision Pipeline

**GitHub Repo Link:** https://github.com/NagasaiManda/DA6401-PA2

**Report Link:** https://wandb.ai/ee23b042-indian-institute-of-technology-madras/DA6401_Assignment2/reports/Building-a-Complete-Visual-Perception-Pipeline--VmlldzoxNjQ5MTAzOQ?accessToken=now7ko1kbzrbybnt83miqa73vf1vk7uumvxsrtq5f07u5ros96j2ekt92myc2719

## Project Summary
This project implements a multitask visual perception pipeline on the Oxford-IIIT Pet dataset, combining:
- **Classification**: Pet breed classification (37 classes)
- **Localization**: Bounding box prediction (center-x, center-y, width, height)
- **Segmentation**: Pet trimap segmentation (foreground, background, unknown)

Implemented modules include:
- VGG11 encoder backbone with BatchNorm and CustomDropout
- Classification head
- Localization head (regression)
- U-Net style segmentation decoder with skip connections
- Losses: CrossEntropyLoss, IoULoss, DiceLoss, CEDiceLoss
- Multitask model with shared backbone

## Repository Structure
- `assignment/models/`: core model implementations (VGG11, segmentation, classification, localization)
- `assignment/data/`: dataset loading and preprocessing (Oxford-IIIT Pet LazyDataset)
- `assignment/losses/`: loss functions (IoU, Dice, combined losses)
- `assignment/train.py`: training entry point for all three tasks
- `assignment/inference.py`: evaluation/inference entry point

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```


## Notes
- Input images are resized to 224×224 and normalized using ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Localization outputs are in pixel space (not normalized) relative to the 224×224 input canvas
- Segmentation produces trimap masks: 0=background, 1=foreground, 2=boundary
- Training uses data augmentation (horizontal flip, color jitter, RandomResizedCrop) for robustness
- Validation and test sets use only resize and normalization
- Models use shared VGG11 backbone to reduce redundancy

## References
- VGG11 Architecture: https://arxiv.org/abs/1409.1556
- Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/


