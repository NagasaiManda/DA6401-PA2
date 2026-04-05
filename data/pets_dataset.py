"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np

class transforms:
    def __init__(self):
        self.valid_transforms = [self.random_rotation, self.horizontal_flip, self.vertical_flip,
                                self.random_crop, self.random_brightness, self.random_contrast, 
                                self.add_noise]

    def random_rotation(self, img, mask):
        angle = random.randint(-30, 30)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        return np.array(img), np.array(mask)

    def horizontal_flip(self, img, mask):
        if random.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask

    def vertical_flip(self, img, mask):
        if random.random() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        return img, mask

    def random_crop(self, img, mask, crop_size=(200, 200)):
        h, w = img.shape[:2]
        ch, cw = crop_size


        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)

        img = img[top:top+ch, left:left+cw]
        mask = mask[top:top+ch, left:left+cw]

        return self.resize(img, mask)

    # Helper function to be used in crop and resizing input images
    def resize(self, img, mask, size=(224, 224)):
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        img = img.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        return np.array(img), np.array(mask)

    def random_brightness(self, img, mask):
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        return np.array(img), mask

    def random_contrast(self, img, mask):
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        return np.array(img), mask

    def add_noise(self, img, mask):
        noise = np.random.normal(0, 10, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8), mask

    def normalize(self, img):
        img = (img / 255.0).astype(np.float32)
        return img

    def __call__(self, img, mask):
        img, mask = self.resize(img, mask)
        num_transforms = random.randint(0, 2)
        for _ in range(num_transforms):
            transform = random.choice(self.valid_transforms)
            img, mask = transform(img, mask)
        img = self.normalize(img)

        return img, mask








class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = root_dir
        self.split = split 
        self.transform = transforms()
        self.images_dir = os.path.join(root_dir, "images", "images")
        self.masks_dir = os.path.join(root_dir,"annotations", "annotations", "trimaps")
        self.test_size = 0.2
        self.random_state = 42
        self.x = []
        self.y = []
        for img_name in os.listdir(self.images_dir):
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, "._"+ img_name.split(".")[0] + ".png")
            self.x.append(img_path)
            self.y.append(mask_path)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size, random_state=self.random_state)
    
    def __len__(self):
        if self.split == "train":
            return len(self.x_train)
        else:
            return len(self.x_test)
    
    def __getitem__(self, idx):
        if self.split == "train":
            img_path = self.x_train[idx]
            mask_path = self.y_train[idx]
        else:
            img_path = self.x_test[idx]
            mask_path = self.y_test[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        img, mask = self.transform(img, mask)
        return img, mask
        