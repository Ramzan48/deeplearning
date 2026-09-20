import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    def __init__(self, in_channels, decoder_dim, num_classes):
        super().__init__()
        
        self.projections = nn.ModuleList([nn.Linear(c, decoder_dim) for c in in_channels])
        
        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_channels)*decoder_dim, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout2d(p=0.1)
        
        self.head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        
    def forward(self, features):
        target_H, target_W = features[0].shape[2], features[0].shape[3]
        projected = []
        for feat, proj in zip(features, self.projections):
            B, C, H, W = feat.shape
            x = feat.permute(0, 2, 3, 1).reshape(B, H*W, C) # B, HW, C
            x = proj(x) # B, HW, decoder_dim
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2) # B, decoder_dim, H, W
            
            x = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
            projected.append(x)
        
        x = torch.cat(projected, dim=1)
        x = self.fuse(x) # B, decoder_dim, H/4, W/4
        
        x = self.dropout(x)
        x = self.head(x) # B, num_classes, H/4, W/4
        
        return x
    
    
if __name__ == "__main__":
    
    in_channels = [96, 192, 384, 768]
    
    dummy_features = [
        torch.randn(1, 96, 512 // 4, 512 // 4),
        torch.randn(1, 192, 512 // 8, 512 // 8),
        torch.randn(1, 384, 512 // 16, 512 // 16),
        torch.randn(1, 768, 512 // 32, 512 // 32),
    ]
    for i in dummy_features:
        print(i.shape)
        
    decoder = MLPDecoder(in_channels, 256, 12)
    decoder.eval()
    with torch.no_grad():
        logits = decoder(dummy_features)
    
    print(f"logits: {tuple(logits.shape)}")
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Trainable params: {params / 1e6:.2f} M")