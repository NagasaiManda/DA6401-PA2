"""Inference and evaluation
"""


"""Inference script for Multi-Task Perception Model
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from data.pets_data import OxfordIIITPetLazyDataset
from models.multitask import MultiTaskPerceptionModel 

def calculate_iou(pred_boxes, target_boxes):
    pred_boxes[:, 2:] = pred_boxes[:, 2:].clamp(min=1e-6)

    # Convert CXCYWH -> XYXY
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)

    union = pred_area + tgt_area - inter_area + 1e-6
    iou = inter_area / union
    
    return iou

def calculate_dice(preds, masks, num_classes=3):
    preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    masks_onehot = F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (preds_onehot * masks_onehot).sum(dim=(2, 3))
    cardinality = preds_onehot.sum(dim=(2, 3)) + masks_onehot.sum(dim=(2, 3))
    
    dice = (2. * intersection + 1e-6) / (cardinality + 1e-6)
    return dice.mean()

def run_inference(root_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    test_dataset = OxfordIIITPetLazyDataset(root_dir=root_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    print(f"Test Dataset size: {len(test_dataset)}")

    print("Loading Multi-Task Model and pre-trained weights...")
    model = MultiTaskPerceptionModel()
    model.to(device)
    model.eval()

    # Classification
    all_class_preds, all_class_labels = [], []
    total_class_loss = 0.0
    
    # Localization
    total_loc_mse = 0.0
    total_loc_iou = 0.0
    total_loc_acc = 0.0 # Bounding box accuracy (IoU > 0.5)
    
    # Segmentation
    total_seg_loss = 0.0
    total_seg_dice = 0.0
    total_seg_pixel_acc = 0.0
    
    total_samples = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Batches"):
            images = batch["image"].float().to(device)
            
            # Ground Truths
            labels = batch["label"].long().to(device)
            bboxes = batch["bbox"].float().to(device)
            masks = batch["mask"].long().to(device)

            batch_size = images.size(0)
            total_samples += batch_size

            # Forward Pass
            outputs = model(images)
            
            # Extract Outputs
            class_logits = outputs['classification']
            loc_preds = outputs['localization']
            seg_logits = outputs['segmentation']

            class_loss = F.cross_entropy(class_logits, labels)
            total_class_loss += class_loss.item() * batch_size
            
            class_preds = torch.argmax(class_logits, dim=1)
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_labels.extend(labels.cpu().numpy())

            loc_mse = F.mse_loss(loc_preds, bboxes, reduction='mean')
            total_loc_mse += loc_mse.item() * batch_size
            
            ious = calculate_iou(loc_preds.clone(), bboxes.clone())
            total_loc_iou += ious.sum().item()
            total_loc_acc += (ious >= 0.5).float().sum().item()

            seg_loss = F.cross_entropy(seg_logits, masks)
            total_seg_loss += seg_loss.item() * batch_size
            
            seg_preds = torch.argmax(seg_logits, dim=1)
            
            pixel_acc = (seg_preds == masks).float().mean()
            total_seg_pixel_acc += pixel_acc.item() * batch_size
            
            dice_score = calculate_dice(seg_preds, masks, num_classes=seg_logits.shape[1])
            total_seg_dice += dice_score.item() * batch_size

    # Classification
    avg_class_loss = total_class_loss / total_samples
    class_accuracy = (np.array(all_class_preds) == np.array(all_class_labels)).mean()
    class_f1_macro = f1_score(all_class_labels, all_class_preds, average='macro')
    class_f1_weighted = f1_score(all_class_labels, all_class_preds, average='weighted')

    # Localization
    avg_loc_mse = total_loc_mse / total_samples
    avg_loc_iou = total_loc_iou / total_samples
    avg_loc_acc = total_loc_acc / total_samples # % of boxes with IoU >= 0.5

    # Segmentation
    avg_seg_loss = total_seg_loss / total_samples
    avg_seg_dice = total_seg_dice / total_samples
    avg_seg_pixel_acc = total_seg_pixel_acc / total_samples

    print("\n" + "="*50)
    print(" MULTI-TASK INFERENCE RESULTS ".center(50, "="))
    print("="*50)
    
    print("\n[ Classification ]")
    print(f"Accuracy:      {class_accuracy:.4f}")
    print(f"F1 (Macro):    {class_f1_macro:.4f}")
    print(f"F1 (Weighted): {class_f1_weighted:.4f}")
    print(f"CE Loss:       {avg_class_loss:.4f}")

    print("\n[ Localization ]")
    print(f"Accuracy (IoU >= 0.5): {avg_loc_acc:.4f}")
    print(f"Mean IoU:              {avg_loc_iou:.4f}")
    print(f"MSE Loss:              {avg_loc_mse:.4f}")

    print("\n[ Segmentation ]")
    print(f"Dice Score:            {avg_seg_dice:.4f}")
    print(f"Pixel-wise Accuracy:   {avg_seg_pixel_acc:.4f}")
    print(f"CE Loss:               {avg_seg_loss:.4f}")
    print("="*50)

if __name__ == "__main__":
    # change dataset dir here
    DATASET_DIR = "/kaggle/input/datasets/nagasaimanda/oxford-pets/dataset"
    run_inference(root_dir=DATASET_DIR)