import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_encoder import SwinEncoder
from mlp_decoder import MLPDecoder

class SwinSegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = SwinEncoder(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=3,
            num_classes=12,
            head_dim=32,
            window_size=8,
            downscaling_factors = (4,2,2,2),
            relative_pos_embedding=True,
            dropout=0.1,
            drop_path_rate=0.1
        )
        
        in_channels = [96 * (2 ** i) for i in range(4)]
        self.decoder = MLPDecoder(in_channels=in_channels, decoder_dim=256, num_classes=12)
    
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        features = self.encoder(x) 
        logits = self.decoder(features) # B, num_classes, H/4, W/4
        
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits # B, num_classes, H, W
        
        
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinSegFormer().to(device)
    
    total = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for p in model.encoder.parameters())
    decoder = sum(p.numel() for p in model.decoder.parameters())
    print(f"Encoder : {encoder/1e6:.2f}M\nDecoder : {decoder/1e6:.2f}M\nTotal : {total/1e6:.2f}M")
    
    dummy = torch.randn(4, 3, 512, 512, device=device)
    with torch.no_grad():
        logits = model(dummy)
    print(f"input : {tuple(dummy.shape)}")
    print(f"logits : {tuple(logits.shape)}")
    