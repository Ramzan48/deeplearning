import torch
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchmetrics import JaccardIndex

from config import Config as c
from dataset import get_dataloaders
from swin_segformer import SwinSegFormer

PALETTE = np.array([
    [  0,   0,   0],
    [220,  20,  60],
    [  0, 200,   0],
    [ 30, 144, 255],
    [255, 215,   0],
    [255,  69,   0],
    [148, 103, 189],
    [150,  75,   0],
    [128, 128, 128],
    [ 34, 139,  34],
    [169, 169, 169],
    [135, 206, 235],
], dtype=np.uint8)

VIS_NAMES = ["315006_008", "200004_001", "213002_008", "002000_015"]

def mask_to_rgb(mask):
    return PALETTE[mask.clip(0, len(PALETTE) - 1)]

def denormalise(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return np.clip(img, 0, 1)
        
        

def main():
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    _, _, test_loader, _ = get_dataloaders(c.DATA_ROOT)
    model = SwinSegFormer().to(device)
    checkpt = os.path.join(c.CHECKPOINT_DIR, "best.pth")
    
    state = torch.load(checkpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')} (best val mIoU: {state.get('best_miou', 0)*100:.2f}%)")
    
    model.eval()
    
    miou_metric = JaccardIndex("multiclass", num_classes=c.NUM_CLASSES, average="macro").to(device)
    per_class_miou = JaccardIndex("multiclass", num_classes=c.NUM_CLASSES, average="none").to(device)
    
    correct = 0
    total = 0
    
    n_total    = len(test_loader.dataset)
    sample_every = max(1, n_total // 20)
    candidates = []
    img_counter = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images).argmax(dim=1)
            miou_metric.update(preds, masks)
            per_class_miou.update(preds, masks)
            correct += (preds == masks).sum().item()
            total   += masks.numel()
            
            for i in range(images.shape[0]):
                if img_counter % sample_every == 0:
                    candidates.append((
                        denormalise(images[i]),
                        mask_to_rgb(masks[i].cpu().numpy()),
                        mask_to_rgb(preds[i].cpu().numpy()),
                    ))
                img_counter += 1
            
            
    miou = miou_metric.compute().item()
    per_class = per_class_miou.compute().cpu().numpy()
    pixel_acc = correct / total
    
    w = max(len(n) for n in c.CLASS_NAMES)
    print()
    print(f"{'mIoU':<{w}} {miou*100:6.2f}%")
    print(f"{'Pixel acc.':<{w}} {pixel_acc*100:6.2f}%")
    print()
    for i in np.argsort(per_class)[::-1]:
        print(f"  {c.CLASS_NAMES[i]:<{w}} {per_class[i]*100:6.2f}%")
    print()

    split_file = os.path.join(c.DATA_ROOT, "ImageSets", "val.txt")
    with open(split_file) as f:
        all_names = [l.strip() for l in f if l.strip()]
 
    selected = []
    with torch.no_grad():
        for name in VIS_NAMES:
            idx = all_names.index(name)
            img_tensor, mask_tensor = test_loader.dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            pred = model(img_tensor).argmax(dim=1)[0].cpu().numpy()
            selected.append((
                denormalise(img_tensor[0]),
                mask_to_rgb(mask_tensor.numpy()),
                mask_to_rgb(pred),
            ))
 
    os.makedirs("visualisations", exist_ok=True)
    fig, axes = plt.subplots(3, len(selected), figsize=(len(selected) * 4, 9))
    fig.suptitle("Swin-SegFormer — Test set predictions", fontsize=13)
 
    for col, (img, gt, pred) in enumerate(selected):
        for row, (data, label) in enumerate(zip(
            [img, gt, pred], ["Input", "Ground truth", "Prediction"]
        )):
            axes[row, col].imshow(data)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=10)
 
    patches = [mpatches.Patch(color=PALETTE[i] / 255.0, label=c.CLASS_NAMES[i])
               for i in range(c.NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.05))
 
    plt.tight_layout()
    save_path = "visualisations/test_grid.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()