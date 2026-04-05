"""Training entrypoint
"""

import os

import torch

from models import VGG11Classifier
from models import CustomDropout
from torch import optim
from data.pets_data import OxfordIIITPetRawDataset
from torch.utils.data import DataLoader
from torch import nn
from models import VGG11Localizer
from losses import IoULoss 

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].float()
            labels = batch["label"].long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate_localizer(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].float()
            target_boxes = batch["bbox"].float()  # [B, 4]

            pred_boxes = model(images)
            loss = criterion(pred_boxes, target_boxes)

            total_loss += loss.item() * images.size(0)

            # --- IoU computation (same logic as loss, reused) ---
            pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
            pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
            pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
            pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

            tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
            tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
            tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
            tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

            inter_x1 = torch.max(pred_x1, tgt_x1)
            inter_y1 = torch.max(pred_y1, tgt_y1)
            inter_x2 = torch.min(pred_x2, tgt_x2)
            inter_y2 = torch.min(pred_y2, tgt_y2)

            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h

            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)

            union = pred_area + tgt_area - inter_area + 1e-6
            iou = inter_area / union

            total_iou += iou.sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_iou = total_iou / total_samples

    return avg_loss, avg_iou


def train_classifier(num_epochs,continue_train=False):
    print("Loading datasets...")
    train_dataset = OxfordIIITPetRawDataset(root_dir="../dataset/", mode="train")
    val_dataset = OxfordIIITPetRawDataset(root_dir="../dataset/", mode="val")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True)
    print("Datasets loaded.")
    model = VGG11Classifier(num_classes=37)
    best_model_path = "best_model_classifier.pt"
    if continue_train:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded weights from {best_model_path} for continued training.")
    

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for batch in train_loader:
            images = batch["image"].float()
            labels = batch["label"].long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_samples += images.size(0)


        train_loss = train_loss_sum / train_samples
        train_acc = train_correct / train_samples
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)


        print(
            f"Epoch {epoch + 1}: "
            f"train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, "
            f"best_val_acc={best_val_acc:.4f}"
        )



def train_localizer(num_epochs,continue_train=False):
    print("Loading datasets...")
    train_dataset = OxfordIIITPetRawDataset(root_dir="../dataset/", mode="train")
    val_dataset = OxfordIIITPetRawDataset(root_dir="../dataset/", mode="val")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, persistent_workers=True)
    print("Datasets loaded.")
    model = VGG11Localizer(num_classes=37)
    best_model_path = "best_model_localizer.pt"
    if continue_train:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded weights from {best_model_path} for continued training.")
    

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = IoULoss()
    best_val_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for batch in train_loader:
            images = batch["image"].float()
            labels = batch["bbox"].float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

        
        train_loss, train_iou = evaluate_localizer(model, train_loader, criterion)
        val_loss, val_iou = evaluate_localizer(model, val_loader, criterion)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)


        print(
            f"Epoch {epoch + 1}: "
            f"train_iou={train_iou:.4f}, train_loss={train_loss:.4f}, "
            f"val_iou={val_iou:.4f}, val_loss={val_loss:.4f}, "
            f"best_val_iou={best_val_iou:.4f}"
        )

if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())        
    torch.set_num_interop_threads(os.cpu_count())
    train_classifier(10,continue_train=True)
