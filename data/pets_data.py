import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

from torchvision.transforms import v2
from torchvision import tv_tensors

class OxfordIIITPetLazyDataset(Dataset):
    NUM_CLASSES = 37

    def __init__(self, root_dir: str, mode: str = "train", image_size=(224, 224)):
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size

        self.images_dir = os.path.join(root_dir, "images", "images")
        self.annotations_dir = os.path.join(root_dir, "annotations", "annotations")
        self.masks_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")

        self.samples = self._load_split_samples()

        # Augmentation pipeline
        if self.mode == "train":
           self.transform = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=self.image_size, scale=(0.85, 1.0), antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

        else:
            # For Val/Test, Just resize and normalize
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(self.image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
            ])

    # Check if mask exists
    def _mask_path(self, image_id: str):
        p1 = os.path.join(self.masks_dir, f"{image_id}.png")
        p2 = os.path.join(self.masks_dir, f"._{image_id}.png")
        return p1 if os.path.exists(p1) else p2

    # helper to get path mappings
    def _split_file_path(self):
        split_files = {"train": "trainval.txt", "val": "trainval.txt", "test": "test.txt"}
        return os.path.join(self.annotations_dir, split_files[self.mode])


    # Splitting train and val
    def _partition_trainval_samples(self, samples, val_ratio=0.2):
        random.seed(42) 
        random.shuffle(samples)
        split = int(len(samples) * (1 - val_ratio))
        return samples[:split], samples[split:]

    # Loading samples
    def _load_split_samples(self):
        samples = []
        with open(self._split_file_path(), "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                parts = line.strip().split()
                image_id, class_id = parts[0], parts[1]
                samples.append({
                    "image_path": os.path.join(self.images_dir, f"{image_id}.jpg"),
                    "mask_path": self._mask_path(image_id),
                    "xml_path": os.path.join(self.xmls_dir, f"{image_id}.xml"),
                    "label": int(class_id) - 1,
                })
        if self.mode in {"train", "val"}:
            train, val = self._partition_trainval_samples(samples)
            return train if self.mode == "train" else val
        return samples

    # Loading bbox coords 
    def _load_bbox(self, xml_path, orig_w, orig_h):
        if not os.path.exists(xml_path):
            return [0.0, 0.0, float(orig_w), float(orig_h)]
        root = ET.parse(xml_path).getroot()
        box = root.find(".//bndbox")
        if box is None:
            return [0.0, 0.0, float(orig_w), float(orig_h)]
        return [
            float(box.find("xmin").text),
            float(box.find("ymin").text),
            float(box.find("xmax").text),
            float(box.find("ymax").text)
        ]

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
    
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"])
        orig_w, orig_h = image.size
        
        img_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)
        
        bbox_raw = self._load_bbox(sample["xml_path"], orig_w, orig_h)
        boxes_tv = tv_tensors.BoundingBoxes(
            [bbox_raw], 
            format="XYXY", 
            canvas_size=(orig_h, orig_w)
        )
    
        img_tv, mask_tv, boxes_tv = self.transform(img_tv, mask_tv, boxes_tv)
    
        mask_tv = mask_tv.to(torch.long).squeeze(0)
        mask_tv = mask_tv - 1
        label = torch.tensor(sample["label"], dtype=torch.long)
    
        bbox = boxes_tv.squeeze(0)  # (4,)
        xmin, ymin, xmax, ymax = bbox
    
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w  = xmax - xmin
        h  = ymax - ymin

        # Convert "XYXY" fromat to "CxCyWH" format    
        bbox_cxcywh = torch.stack([cx, cy, w, h]).to(torch.float32)
    
        return {
            "image": img_tv,
            "mask": mask_tv,
            "label": label,
            "bbox": bbox_cxcywh, 
        }