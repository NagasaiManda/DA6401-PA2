import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

# New imports for V2 transforms
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

        # Define the augmentation pipeline
        if self.mode == "train":
            self.transform = v2.Compose([
                v2.ToImage(),                            # Convert to tensor-based image
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.2),
                v2.RandomRotation(degrees=15),
                # RandomResizedCrop handles the "output is always 224x224" requirement
                v2.RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0), antialias=True),
                v2.ColorJitter(brightness=0.3, contrast=0.2),
                v2.GaussianNoise(mean=0, sigma=0.05),    # Add random noise
                v2.ToDtype(torch.float32, scale=True),   # Scale to [0, 1]
            ])
        else:
            # For Val/Test: Just resize and normalize
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(self.image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def _mask_path(self, image_id: str):
        p1 = os.path.join(self.masks_dir, f"{image_id}.png")
        p2 = os.path.join(self.masks_dir, f"._{image_id}.png")
        return p1 if os.path.exists(p1) else p2

    def _split_file_path(self):
        split_files = {"train": "trainval.txt", "val": "trainval.txt", "test": "test.txt"}
        return os.path.join(self.annotations_dir, split_files[self.mode])

    def _partition_trainval_samples(self, samples, val_ratio=0.2):
        # Seed shuffle for reproducibility if needed, or use random.shuffle
        random.seed(42) 
        random.shuffle(samples)
        split = int(len(samples) * (1 - val_ratio))
        return samples[:split], samples[split:]

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

        # 1. Load raw data
        image = Image.open(sample["image_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"])
        orig_w, orig_h = image.size
        
        # 2. Wrap targets in Torchvision V2 Tensors
        # This is the "magic" that allows transforms to update Bboxes and Masks
        img_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)
        
        bbox_raw = self._load_bbox(sample["xml_path"], orig_w, orig_h)
        # Format is [xmin, ymin, xmax, ymax] (XYXY)
        boxes_tv = tv_tensors.BoundingBoxes(
            [bbox_raw], 
            format="XYXY", 
            canvas_size=(orig_h, orig_w)
        )

        # 3. Apply the transform pipeline to all simultaneously
        # Note: Bbox scaling, rotation, and cropping are handled automatically here
        img_tv, mask_tv, boxes_tv = self.transform(img_tv, mask_tv, boxes_tv)

        # 4. Final Formatting
        # Convert mask to long for CrossEntropy and squeeze out channel dim
        mask_tv = mask_tv.to(torch.long).squeeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "image": img_tv,
            "mask": mask_tv,
            "label": label,
            "bbox": boxes_tv.squeeze(0)
        }