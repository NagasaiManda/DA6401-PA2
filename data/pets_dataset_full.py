"""Oxford-IIIT Pet dataset with image, mask, one-hot label, and bbox."""

import os
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class transforms:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.valid_transforms = [
            self.random_rotation,
            self.horizontal_flip,
            self.vertical_flip,
            self.random_crop,
            self.random_brightness,
            self.random_contrast,
            self.add_noise,
        ]

    def random_rotation(self, img, mask, bbox):
        angle = random.randint(-30, 30)

        pil_img = Image.fromarray(img)
        pil_mask = Image.fromarray(mask)

        height, width = img.shape[:2]
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        pil_mask = pil_mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        bbox = self._rotate_bbox(bbox, angle, width, height)
        return np.array(pil_img), np.array(pil_mask), bbox

    def horizontal_flip(self, img, mask, bbox):
        if random.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

            width = img.shape[1]
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([width - xmax, ymin, width - xmin, ymax], dtype=np.float32)

        return img, mask, bbox

    def vertical_flip(self, img, mask, bbox):
        if random.random() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)

            height = img.shape[0]
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([xmin, height - ymax, xmax, height - ymin], dtype=np.float32)

        return img, mask, bbox

    def random_crop(self, img, mask, bbox, crop_size=(200, 200)):
        h, w = img.shape[:2]
        ch, cw = crop_size

        if h <= ch or w <= cw:
            return self.resize(img, mask, bbox)

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)

        img = img[top : top + ch, left : left + cw]
        mask = mask[top : top + ch, left : left + cw]

        xmin, ymin, xmax, ymax = bbox
        xmin -= left
        xmax -= left
        ymin -= top
        ymax -= top

        xmin = np.clip(xmin, 0, cw)
        xmax = np.clip(xmax, 0, cw)
        ymin = np.clip(ymin, 0, ch)
        ymax = np.clip(ymax, 0, ch)

        if xmax <= xmin or ymax <= ymin:
            bbox = np.array([0.0, 0.0, float(cw), float(ch)], dtype=np.float32)
        else:
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

        return self.resize(img, mask, bbox)

    def resize(self, img, mask, bbox, size=None):
        if size is None:
            size = self.size

        height, width = img.shape[:2]
        target_width, target_height = size

        pil_img = Image.fromarray(img)
        pil_mask = Image.fromarray(mask)

        pil_img = pil_img.resize(size, Image.BILINEAR)
        pil_mask = pil_mask.resize(size, Image.NEAREST)

        bbox = bbox.astype(np.float32).copy()
        bbox[[0, 2]] *= target_width / width
        bbox[[1, 3]] *= target_height / height

        bbox[0] = np.clip(bbox[0], 0, target_width)
        bbox[2] = np.clip(bbox[2], 0, target_width)
        bbox[1] = np.clip(bbox[1], 0, target_height)
        bbox[3] = np.clip(bbox[3], 0, target_height)

        return np.array(pil_img), np.array(pil_mask), bbox

    def random_brightness(self, img, mask, bbox):
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
        return np.array(pil_img), mask, bbox

    def random_contrast(self, img, mask, bbox):
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
        return np.array(pil_img), mask, bbox

    def add_noise(self, img, mask, bbox):
        noise = np.random.normal(0, 10, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8), mask, bbox

    def normalize(self, img):
        return (img / 255.0).astype(np.float32)

    def _rotate_bbox(self, bbox, angle, width, height):
        radians = np.deg2rad(angle)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)

        cx = width / 2.0
        cy = height / 2.0

        xmin, ymin, xmax, ymax = bbox
        corners = np.array(
            [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ],
            dtype=np.float32,
        )

        translated = corners - np.array([cx, cy], dtype=np.float32)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        rotated = translated @ rotation.T
        rotated += np.array([cx, cy], dtype=np.float32)

        xmin_new = np.clip(np.min(rotated[:, 0]), 0, width)
        xmax_new = np.clip(np.max(rotated[:, 0]), 0, width)
        ymin_new = np.clip(np.min(rotated[:, 1]), 0, height)
        ymax_new = np.clip(np.max(rotated[:, 1]), 0, height)

        if xmax_new <= xmin_new or ymax_new <= ymin_new:
            return np.array([0.0, 0.0, float(width), float(height)], dtype=np.float32)

        return np.array([xmin_new, ymin_new, xmax_new, ymax_new], dtype=np.float32)

    def __call__(self, img, mask, bbox):
        img, mask, bbox = self.resize(img, mask, bbox)

        num_transforms = random.randint(0, 2)
        for _ in range(num_transforms):
            transform = random.choice(self.valid_transforms)
            img, mask, bbox = transform(img, mask, bbox)

        img, mask, bbox = self.resize(img, mask, bbox)
        img = self.normalize(img)

        return img, mask, bbox.astype(np.float32)


class OxfordIIITPetFullDataset(Dataset):
    """Oxford-IIIT Pet dataset that returns all targets together."""

    NUM_CLASSES = 37

    def __init__(self, root_dir: str, mode: str = "train"):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transforms(size=(224, 224))

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
                samples.append(
                    {
                        "image_id": image_id,
                        "image_path": os.path.join(self.images_dir, f"{image_id}.jpg"),
                        "mask_path": self._mask_path(image_id),
                        "xml_path": os.path.join(self.xmls_dir, f"{image_id}.xml"),
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

    def _one_hot(self, class_index: int) -> np.ndarray:
        label = np.zeros(self.NUM_CLASSES, dtype=np.float32)
        label[class_index] = 1.0
        return label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = np.array(Image.open(sample["image_path"]).convert("RGB"))
        mask = np.array(Image.open(sample["mask_path"]))
        bbox, has_bbox = self._load_bbox(sample["xml_path"], img.shape)

        img, mask, bbox = self.transform(img, mask, bbox)
        label = self._one_hot(sample["class_id"])

        return {
            "image": img,
            "mask": mask,
            "label": label,
            "bbox": bbox,
            "class_id": sample["class_id"],
            "image_id": sample["image_id"],
            "species": sample["species"],
            "breed_id": sample["breed_id"],
            "has_bbox": has_bbox,
        }
