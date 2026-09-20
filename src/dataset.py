import numpy as np
import os
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

from config import Config as c

class AeroscapesDataset(Dataset):
    def __init__(self, root, split = "train", augment = True):
        self.root = root
        self.split = split
        self.image_size = c.IMAGE_SIZE
        self.augment = augment and (split == "train")

        if split == "train":
            split_file = "train.txt"
        elif split == "val":
            split_file = "test.txt"
        elif split == "test":
            split_file = "val.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        split_path = os.path.join(root, "ImageSets", split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")
        
        with open(split_path) as f:
            self.names = [line.strip() for line in f if line.strip()]
        
        print(f"[Dataset] {split}: {len(self.names)} images (augment={self.augment})")
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        
        img_path  = os.path.join(self.root, "JPEGImages", name + ".jpg")
        mask_path = os.path.join(self.root, "SegmentationClass", name + ".png")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        H, W = self.image_size

        image = image.resize((W, H), Image.BILINEAR)
        mask  = mask.resize((W, H), Image.NEAREST)
        
        if self.augment:
            image, mask = self._augment(image, mask)
    
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
        
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask = mask.clamp(0, c.NUM_CLASSES - 1)

        return image, mask
    
    def _augment(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)
            mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST,  fill=0)

        if random.random() > 0.5:
            H, W = self.image_size
            scale = random.uniform(0.75, 1.25)
            
            new_H = int(H * scale)
            new_W = int(W * scale)
            
            image = image.resize((new_W, new_H), Image.BILINEAR)
            mask  = mask.resize((new_W, new_H), Image.NEAREST)
            
            top  = random.randint(0, max(0, new_H - H))
            left = random.randint(0, max(0, new_W - W))
            
            crop_H = min(H, new_H)
            crop_W = min(W, new_W)
            
            image = TF.crop(image, top, left, crop_H, crop_W)
            mask  = TF.crop(mask,  top, left, crop_H, crop_W)
            
            if scale < 1.0:
                image = image.resize((W, H), Image.BILINEAR)
                mask  = mask.resize((W, H), Image.NEAREST)
                
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.6, 1.4))
            
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.6, 1.4))
            
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.6, 1.4))
            
        if random.random() > 0.5:
            image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))

        return image, mask
    
    
def create_splits(root, train_ratio = 0.8):
    image_sets = os.path.join(root, "ImageSets")
    
    trn_path   = os.path.join(image_sets, "trn.txt")
    train_out  = os.path.join(image_sets, "train.txt")
    val_out   = os.path.join(image_sets, "test.txt")
    
    if os.path.exists(train_out) and os.path.exists(val_out):
        return
    
    if not os.path.exists(trn_path):
        raise FileNotFoundError(f"Missing file: {trn_path}")
    
    with open(trn_path) as f:
        names = [line.strip() for line in f if line.strip()]
    
    train_names, val_names = train_test_split(names, train_size=train_ratio)
    
    with open(train_out, "w") as f:
            for x in train_names:
                f.write(x + "\n")
    
    with open(val_out, "w") as f:
            for x in val_names:
                f.write(x + "\n")
     
def compute_class_weights(dataset, num_samples=500):
    """
    Estimate inverse frequency class weights.
    """
    counts = np.zeros(c.NUM_CLASSES, dtype=np.float64)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
            _, mask = dataset[idx]
            for i in range(c.NUM_CLASSES):
                counts[i] += (mask == i).sum().item()
                
    counts = np.maximum(counts, 1.0) # Just to avoid division by zero
    w = 1.0 / counts
    w = w / w.sum() * c.NUM_CLASSES
    
    print("[Dataset] Class weights:")
    for name, i, cnt in zip(c.CLASS_NAMES, w, counts):
        print(f"    {name} w = {i:.3f}  (pixels {cnt})")
    
    return torch.FloatTensor(w)


def get_dataloaders(root):
    create_splits(root)
    train_dataset = AeroscapesDataset(root, split="train", augment=True)
    val_dataset = AeroscapesDataset(root, split="val", augment=False)
    test_dataset = AeroscapesDataset(root, split="test", augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=c.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    class_weights = compute_class_weights(train_dataset)
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(c.DATA_ROOT)
    
    images, masks = next(iter(train_loader))