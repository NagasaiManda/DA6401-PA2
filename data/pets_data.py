"""Oxford-IIIT Pet dataset with raw image, mask, label, bbox, and split info."""

import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPetRawDataset(Dataset):
    """Loads channel-first images, class-index labels, masks, and bboxes."""

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

    def _mask_path(self, image_id: str) -> str:
        candidates = [
            os.path.join(self.masks_dir, f"{image_id}.png"),
            os.path.join(self.masks_dir, f"._{image_id}.png"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"No trimap found for {image_id}")

    def _split_file_path(self) -> str:
        split_files = {
            "train": "trainval.txt",
            "val": "trainval.txt",
            "test": "test.txt",
        }
        if self.mode not in split_files:
            raise ValueError("mode must be 'train', 'val' or 'test'")
        return os.path.join(self.annotations_dir, split_files[self.mode])

    def _partition_trainval_samples(self, samples, val_ratio=0.2, random_state=42):
        indices = list(range(len(samples)))
        rng = random.Random(random_state)
        rng.shuffle(indices)

        val_size = int(len(indices) * val_ratio)
        val_indices = set(indices[:val_size])

        train_samples = [samples[i] for i in range(len(samples)) if i not in val_indices]
        val_samples = [samples[i] for i in range(len(samples)) if i in val_indices]
        return train_samples, val_samples

    def _load_split_samples(self):
        split_file = self._split_file_path()
        samples = []

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                image_id, class_id, species, breed_id = line.split()
                xml_path = os.path.join(self.xmls_dir, f"{image_id}.xml")
                if self.mode in {"train", "val"} and not os.path.exists(xml_path):
                    continue

                samples.append(
                    {
                        "image_id": image_id,
                        "image_path": os.path.join(self.images_dir, f"{image_id}.jpg"),
                        "mask_path": self._mask_path(image_id),
                        "xml_path": xml_path,
                        "class_id": int(class_id) - 1,
                        "species": int(species),
                        "breed_id": int(breed_id),
                    }
                )

        if self.mode in {"train", "val"}:
            train_samples, val_samples = self._partition_trainval_samples(samples)
            return train_samples if self.mode == "train" else val_samples

        return samples

    def _load_bbox(self, xml_path: str, image_shape) -> tuple[np.ndarray, bool]:
        height, width = image_shape[:2]
        if not os.path.exists(xml_path):
            return np.array([0.0, 0.0, float(width), float(height)], dtype=np.float32), False

        root = ET.parse(xml_path).getroot()
        bndbox = root.find(".//bndbox")
        if bndbox is None:
            return np.array([0.0, 0.0, float(width), float(height)], dtype=np.float32), False

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32), True

    def _resize_sample(self, image, mask, bbox):
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.image_size

        image = Image.fromarray(image).resize((target_width, target_height), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((target_width, target_height), Image.NEAREST)

        bbox = bbox.astype(np.float32).copy()
        bbox[[0, 2]] *= target_width / original_width
        bbox[[1, 3]] *= target_height / original_height

        return np.array(image), np.array(mask), bbox

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = np.array(Image.open(sample["image_path"]).convert("RGB"))
        mask = np.array(Image.open(sample["mask_path"]))
        bbox, has_bbox = self._load_bbox(sample["xml_path"], image.shape)
        image, mask, bbox = self._resize_sample(image, mask, bbox)
        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        label = torch.tensor(sample["class_id"], dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "bbox": bbox
            # "split": self.mode,
            # "class_id": sample["class_id"],
            # "image_id": sample["image_id"],
            # "species": sample["species"],
            # "breed_id": sample["breed_id"],
            # "has_bbox": has_bbox,
        }
