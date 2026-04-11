"""Training entrypoint
"""

import os

import torch

from models import VGG11Classifier
from models import CustomDropout
from torch import optim
from data.pets_data import OxfordIIITPetLazyDataset
from torch.utils.data import DataLoader
from torch import nn
from models import VGG11Localizer, VGG11UNet
from losses import IoULoss, DiceLoss, CEDiceLoss
import numpy as np
from sklearn.metrics import f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].float().to(device)
            labels = batch["label"].long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy




def train_classifier(num_epochs,continue_train=False):
    print("Loading datasets...")
    train_dataset = OxfordIIITPetLazyDataset(root_dir="/kaggle/input/datasets/nagasaimanda/oxford-pets/dataset", mode="train")
    val_dataset = OxfordIIITPetLazyDataset(root_dir="/kaggle/input/datasets/nagasaimanda/oxford-pets/dataset", mode="val")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4,  persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True )
    print("Datasets loaded.")
    model = VGG11Classifier(num_classes=37)
    model = model.to(device)
    best_model_path = "best_model_classifier_transforms.pt"
    if continue_train:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded weights from {best_model_path} for continued training.")
    

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3, 
    )
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for batch in train_loader:
            images = (batch["image"].float()).to(device)
            labels = (batch["label"].long()).to(device)

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
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)


        print(
            f"Epoch {epoch + 1}: "
            f"train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, "
            f"best_val_acc={best_val_acc:.4f}"
        )
    np.save("train_loss_withBN_transforms.npy", np.array(train_losses))
    np.save("val_loss_withBN_transforms.npy", np.array(val_losses))

def test_classifier(model_path, root_dir):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load Test Dataset
    # Ensure root_dir matches your training setup
    test_dataset = OxfordIIITPetLazyDataset(root_dir=root_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    print(f"Test Dataset size: {len(test_dataset)}")

    # 3. Load Model
    # Note: Ensure VGG11Classifier is defined in your script environment
    model = VGG11Classifier(num_classes=37)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Move to CPU and convert to numpy for sklearn
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # 'macro' calculates metrics for each label, and finds their unweighted mean. 
    # 'weighted' accounts for label imbalance by calculating the mean of labels weighted by support.
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print("\n" + "="*30)
    print("TEST RESULTS")
    print("="*30)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print("="*30)

    # 5. Detailed Per-Class Report
    # Useful to see which pet breeds the model struggles with most
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds))

def train_localizer(num_epochs,continue_train=False):
    print("Loading datasets...")
    train_dataset = OxfordIIITPetLazyDataset(root_dir="/kaggle/input/datasets/nagasaimanda/oxford-pets/dataset", mode="train")
    val_dataset = OxfordIIITPetLazyDataset(root_dir="/kaggle/input/datasets/nagasaimanda/oxford-pets/dataset", mode="val")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)
    print("Datasets loaded.")
    model = VGG11Localizer()
    best_model_path = "best_model_localizer.pt"
    if continue_train:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded weights from {best_model_path} for continued training.")
    classifier = VGG11Classifier()
    classifier.load_state_dict(torch.load("/kaggle/input/models/nagasai100805/classifier/pytorch/default/1/best_model_classifier_transforms (2).pt"))
    model.vgg11_enc.load_state_dict(classifier.VGG11enc.state_dict())
    model.to(device)
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    model.vgg11_enc.eval()
    criterion = IoULoss()
    best_val_iou = 0.0
    for param in model.vgg11_enc.parameters():
        param.requires_grad = False
    print("Started trainind")
    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for batch in train_loader:
            images = batch["image"].float().to(device)
            labels = batch["bbox"].float().to(device)
            
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
        
def evaluate_localizer(model, dataloader, criterion):
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].float().to(device)
            target_boxes = batch["bbox"].float().to(device)  # [B, 4]

            pred_boxes = model(images)

            # 🔒 Ensure valid widths/heights (VERY IMPORTANT)
            pred_boxes[:, 2:] = pred_boxes[:, 2:].clamp(min=1e-6)

            loss = criterion(pred_boxes, target_boxes)
            total_loss += loss.item() * images.size(0)

            # --- Convert CXCYWH → XYXY ---
            pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
            pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
            pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
            pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

            tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
            tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
            tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
            tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

            # --- Intersection ---
            inter_x1 = torch.max(pred_x1, tgt_x1)
            inter_y1 = torch.max(pred_y1, tgt_y1)
            inter_x2 = torch.min(pred_x2, tgt_x2)
            inter_y2 = torch.min(pred_y2, tgt_y2)

            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h

            # --- Areas (clamped for safety) ---
            pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
            tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)

            union = pred_area + tgt_area - inter_area + 1e-6
            iou = inter_area / union

            total_iou += iou.sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_iou = total_iou / total_samples

    return avg_loss, avg_iou

def train_segmenter(num_epochs, continue_train=False):
    print("Loading datasets...")

    train_dataset = OxfordIIITPetLazyDataset(
        root_dir="/kaggle/input/datasets/nagasaimanda/oxford-pets/dataset",
        mode="train"
    )
    val_dataset = OxfordIIITPetLazyDataset(
        root_dir="/kaggle/input/datasets/nagasaimanda/oxford-pets/dataset",
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=4, persistent_workers=True
    )

    print("Datasets loaded.")

    model = VGG11UNet(num_classes=3)

    best_model_path = "best_model_segmenter.pt"

    if continue_train:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded weights from {best_model_path}")

    # ✅ Load pretrained encoder (same as localization)
    classifier = VGG11Classifier()
    classifier.load_state_dict(torch.load(
        "/kaggle/input/models/nagasai100805/classifier/pytorch/default/1/best_model_classifier_transforms (2).pt"
    ))

    model.encoder.load_state_dict(classifier.VGG11enc.state_dict())

    model.to(device)

    # Freeze encoder (optional but recommended initially)
    for param in model.encoder.parameters():
        param.requires_grad = False
        model.encoder.eval()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    criterion = CEDiceLoss()

    best_val_iou = 0.0

    print("Started training...")

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            images = batch["image"].float().to(device)
            masks = batch["mask"].long().to(device)  # ✅ segmentation mask

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

        train_loss, train_iou = evaluate_segmenter(model, train_loader, criterion)
        val_loss, val_iou = evaluate_segmenter(model, val_loader, criterion)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch+1}: "
            f"train_iou={train_iou:.4f}, train_loss={train_loss:.4f}, "
            f"val_iou={val_iou:.4f}, val_loss={val_loss:.4f}, "
            f"best_val_iou={best_val_iou:.4f}")
    
def evaluate_segmenter(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].float().to(device)
            masks = batch["mask"].long().to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)

            # IoU (mean over classes, excluding background optional)
            intersection = ((preds == masks) & (masks > 0)).sum(dim=(1, 2)).float()
            union = ((preds > 0) | (masks > 0)).sum(dim=(1, 2)).float() + 1e-6

            iou = intersection / union

            total_iou += iou.sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_iou / total_samples
if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())        
    torch.set_num_interop_threads(os.cpu_count())
    train_classifier(10,continue_train=True)
