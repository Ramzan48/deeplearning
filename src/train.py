import wandb
import os
import time
import torch
import torch.nn as nn
import segmentation_models_pytorch.losses as seg_losses
from torchmetrics import JaccardIndex

from dataset import get_dataloaders
from swin_segformer import SwinSegFormer
from config import Config as c

def train_epoch(model, loader, ce_loss, dice_loss, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    for batch_i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            ce = ce_loss(logits, masks)
            dice = dice_loss(logits, masks)
            loss = ce + dice
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if(batch_i + 1) % 10 == 0:
            avg = total_loss / (batch_i + 1)
            print(f"Epoch {epoch} : step {batch_i+1}/{len(loader)} : loss {avg:.4f}  (CE {ce.item():.4f}  Dice {dice.item():.4f})")    
            #wandb.log({"trainer/loss": avg, "trainer/ce": ce.item(), "trainer/dice": dice.item()})
    
    return total_loss / len(loader)
        
def validate(model, loader, ce_loss, dice_loss, miou_metric, device):
    model.eval()
    miou_metric.reset()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                ce = ce_loss(logits, masks)
                dice = dice_loss(logits, masks)
                loss = ce + dice
                
            total_loss += loss.item()
            
            miou_metric.update(logits.argmax(dim=1), masks)
            
    return total_loss / len(loader), miou_metric.compute().item()
    

def main():
    os.makedirs(c.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    wandb.init(
        project="aeroscapes",
        resume="allow",
        config={
            "image_size": c.IMAGE_SIZE,
            "num_classes": c.NUM_CLASSES,
            "batch_size": c.BATCH_SIZE,
            "num_epochs": c.NUM_EPOCHS,
            "encoder_lr": c.ENCODER_LR,
            "decoder_lr": c.DECODER_LR,
            "weight_decay": c.WEIGHT_DECAY,
            "grad_clip": c.GRAD_CLIP,
            "hidden_dim": c.HIDDEN_DIM,
            "window_size": c.WINDOW_SIZE,
            "dropout": c.DROPOUT,
            "drop_path_rate": c.DROP_PATH_RATE,
            "decoder_dim": c.DECODER_DIM,
        },
    )
    train_loader, val_loader, _, class_weights = get_dataloaders(c.DATA_ROOT)
    
    model = SwinSegFormer().to(device)
    wandb.watch(model, log="gradients", log_freq=100)
    
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    dice_loss = seg_losses.DiceLoss(mode = "multiclass", from_logits= True, ignore_index= 0)
    miou_metric = JaccardIndex(task = "multiclass", num_classes= c.NUM_CLASSES, average = "macro").to(device)
    
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": c.ENCODER_LR},
        {"params": model.decoder.parameters(), "lr": c.DECODER_LR},
    ], weight_decay=c.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.NUM_EPOCHS, eta_min=1e-6)
    
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    best_path   = os.path.join(c.CHECKPOINT_DIR, "best.pth")
    last_path   = os.path.join(c.CHECKPOINT_DIR, "last.pth")
    
    start_epoch = 1
    best_miou   = 0.0
    
    if os.path.exists(last_path):
        print(f"Resuming from {last_path}")
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint["best_miou"]
        print(f"Epoch {start_epoch}, best mIoU: {best_miou*100:.2f}%")
    
    print(f"Starting {c.NUM_EPOCHS} epochs")
    
    for epoch in range(start_epoch, c.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, ce_loss, dice_loss, optimizer, device, epoch, scaler)
        val_loss, miou = validate(model, val_loader, ce_loss, dice_loss, miou_metric, device)
        
        scheduler.step()
        elapsed = time.time() - t0
        lr_enc  = optimizer.param_groups[0]["lr"]
        lr_dec  = optimizer.param_groups[1]["lr"]
        
        print(f"Epoch {epoch:3d}/{c.NUM_EPOCHS}  ({elapsed:.0f}s)")
        print(f"  train loss : {train_loss:.4f}")
        print(f"  val loss : {val_loss:.4f}   mIoU : {miou*100:.2f}%")
        
        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/miou": miou * 100,
            "lr/encoder": lr_enc,
            "lr/decoder": lr_dec,
        })
        
        if miou > best_miou:
            best_miou = miou
            torch.save({"epoch": epoch, "best_miou": best_miou, "model": model.state_dict()}, best_path)
            print(f"New best mIoU: {best_miou*100:.2f}%  saved")
            wandb.summary["best_miou"]  = best_miou * 100
            wandb.summary["best_epoch"] = epoch
            
        torch.save({
            "epoch": epoch, "best_miou": best_miou,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, last_path)
        
    print(f"Training done. Best mIoU: {best_miou*100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()