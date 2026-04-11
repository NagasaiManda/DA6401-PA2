"""Custom IoU loss 
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        # TODO: validate reduction in {"none", "mean", "sum"}.

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        pred_boxes, target_boxes: [B, 4] in (cx, cy, w, h)
        """

        # --- 🔒 Ensure valid widths/heights ---
        pred_w = pred_boxes[:, 2].clamp(min=1e-6)
        pred_h = pred_boxes[:, 3].clamp(min=1e-6)
        tgt_w  = target_boxes[:, 2].clamp(min=1e-6)
        tgt_h  = target_boxes[:, 3].clamp(min=1e-6)
        
        # --- Convert CXCYWH → XYXY ---
        pred_x1 = pred_boxes[:, 0] - pred_w / 2
        pred_y1 = pred_boxes[:, 1] - pred_h / 2
        pred_x2 = pred_boxes[:, 0] + pred_w / 2
        pred_y2 = pred_boxes[:, 1] + pred_h / 2
        
        tgt_x1 = target_boxes[:, 0] - tgt_w / 2
        tgt_y1 = target_boxes[:, 1] - tgt_h / 2
        tgt_x2 = target_boxes[:, 0] + tgt_w / 2
        tgt_y2 = target_boxes[:, 1] + tgt_h / 2
        
        # --- Intersection ---
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # --- Areas ---
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        
        # --- IoU ---
        union = pred_area + tgt_area - inter_area + self.eps
        iou = inter_area / union
        
        loss = 1 - iou
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss