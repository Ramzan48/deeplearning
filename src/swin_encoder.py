import torch
import torch.nn as nn
import numpy as np

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # B, 1, 1, 1
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) # One random value per B. [0.1,0.5,,0.9]
        random_tensor = torch.floor(random_tensor + keep_prob) # last : 0.9 + 0.05 = 0.0
        return x / keep_prob * random_tensor


def create_mask(window_size, s, upper_lower):
    mask = torch.zeros(window_size**2, window_size**2) # 49x49
    if upper_lower:
        mask[-s*window_size:, :-s*window_size] = float("-inf") # down left
        mask[:-s*window_size, -s*window_size:] = float("-inf") # up right
    else:
        mask = mask.contiguous().view(window_size, window_size, window_size, window_size)
        mask[:, -s:, :, :-s] = float("-inf")
        mask[:, :-s, :, -s:] = float("-inf")
        mask = mask.view(window_size*window_size, window_size*window_size)
        
    return mask


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, heads_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = heads_dim*heads # = C in paper
        self.heads = heads
        self.scale = heads_dim**-0.5
        self.window_size = window_size # = 7
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        
        if self.shifted: # cyclic shift implementation (paper shows it's better than padding)
            self.s = window_size // 2
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, s=self.s, upper_lower = True), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, s=self.s, upper_lower = False), requires_grad=False)
            
        self.qkv = nn.Linear(dim, inner_dim*3, bias=False)
        
        if self.relative_pos_embedding:
            indices = torch.tensor(np.array([[i, j] for i in range(window_size) for j in range(window_size)]))
            self.register_buffer("relative_indices", indices[None, :, :] - indices[:, None, :])
            self.pos_embedding = nn.Parameter(torch.randn(2*window_size-1, 2*window_size-1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))
            
        self.out = nn.Linear(inner_dim, dim)
    
    def forward(self, x): # B, (56, 28, 14, 7), (56, 28, 14, 7), (96, 192, 384, 768)
        if self.shifted:
            x = torch.roll(x, shifts=(-self.s, -self.s), dims=(1, 2))
        _, height, weight, _ = x.shape
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        
        # num_win_h = height // self.window_size # (56/7=8, 28/7=4, 14/7=2, 7/7=1)
        # num_win_w = weight // self.window_size # (56/7=8, 28/7=4, 14/7=2, 7/7=1)
        
        num_win = height // self.window_size # From now I consider H = W (for now at least)
        
        def reshape_windows(t):
            """
            Core idea of swin. We will compute self-attention in window partitions instead of for the whole image (vit).
            input shape: B, H, W, heads*head_dim where H = W = num_win*window_size
            
            We split heads*head_dim into heads and H and W into num_win. Then we rearrange to put heads in front and group the windows_size and num_win together.
            
            output shape: B, heads, num_win**2, windows_size**2, heads_dim
            """
            B, _, _, HD = t.shape
            head_dim = HD // self.heads
            t = t.contiguous().view(B, num_win, self.window_size, num_win, self.window_size, self.heads, head_dim)
            t = t.permute(0, 5, 1, 3, 2, 4, 6).contiguous()
            t = t.view(B, self.heads, num_win**2, self.window_size**2, head_dim)
            return t
        
        # NOT FORGET FOR POSTER PRES:
        # Let N = H*W be the resolution of the input images.
        # In vit:
        #   we are computing attention for the whole image using: Q @ K^T => O(N*N*C) so it's quadradic with the resolution
        # In swin:
        #   Let M = window_size
        #   we are computing attention for all windows so we do num_win^2 times Q_w @ K_w^T => num_win^2 *O(M^2*M^2*C) = N / M^2 * O(M^2*M^2*C) = O(N*M^2*C)
        # => The advantage of swin: O(N*M^2*C) linear with the resoluton.
        q, k, v = map(reshape_windows, qkv) # each of size:  B, (3, 6, 12, 24), (64, 16, 4, 1), 49, 32
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # B, (3, 6, 12, 24), (64, 16, 4, 1), 49, 49
        
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:,:,0], self.relative_indices[:,:,1]]
        else:
            dots += self.pos_embedding
        
        if self.shifted:
            dots[:, :, -num_win:] += self.upper_lower_mask
            dots[:, :, num_win-1::num_win] += self.left_right_mask
        
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v) # B, (3, 6, 12, 24), (64, 16, 4, 1), 49, 32
        
        out = out.view(out.shape[0], out.shape[1], num_win, num_win, self.window_size, self.window_size, out.shape[-1])
        out = out.permute(0, 2, 4, 3, 5, 1, 6).contiguous()
        out = out.view(out.shape[0], out.shape[1]*out.shape[2], out.shape[3]*out.shape[4], out.shape[5]*out.shape[6])
        
        out = self.out(out)
        
        if self.shifted:
            out = torch.roll(out, shifts=(self.s, self.s), dims=(1, 2))
            
        return out
        
        
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn, drop_path):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x, **kwargs):
        return x + self.drop_path(self.fn(self.norm(x), **kwargs))

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, heads_dim, mlp_dim, shifted, window_size, relative_pos_embedding, dropout, drop_path):
        super().__init__()
        self.attention_block = PreNorm(dim, WindowAttention(dim=dim, heads=heads, heads_dim=heads_dim, shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding), drop_path=drop_path)
        self.mlp_block = PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout), drop_path=drop_path)
    
    def forward(self, x):
        # B, (56, 28, 14, 7), (56, 28, 14, 7), (96, 192, 384, 768) (don't change)
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x
    

class SwinStage(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size, relative_pos_embedding, dropout, drop_path_rates):
        super().__init__()
        
        if drop_path_rates is None:
            drop_path_rates = [0.0] * layers
            
        # before: B, (3, 96, 192, 384), (224, 56, 28, 14), (224, 56, 28, 14)
        self.patch_merging = nn.Conv2d(in_channels, hidden_dimension, kernel_size=downscaling_factor, stride=downscaling_factor) # B, (96, 192, 384, 768), (56, 28, 14, 7), (56, 28, 14, 7)
        
        self.layers = nn.ModuleList([])
        for i in range(layers // 2):
            self.layers.append(nn.ModuleList([
                # worst case (layers=6) drop_path_rates idx: 0, 2, 4 for impair layer
                SwinBlock(dim=hidden_dimension, heads=num_heads, heads_dim=head_dim, mlp_dim=hidden_dimension*4, shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout, drop_path=drop_path_rates[i*2]),
                SwinBlock(dim=hidden_dimension, heads=num_heads, heads_dim=head_dim, mlp_dim=hidden_dimension*4, shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout, drop_path=drop_path_rates[i*2+1])
            ]))
            
    def forward(self, x):
        # before: B, (3, 96, 192, 384), (224, 56, 28, 14), (224, 56, 28, 14)
        x = self.patch_merging(x) # B, (96, 192, 384, 768), (56, 28, 14, 7), (56, 28, 14, 7)
        x = x.permute(0, 2, 3, 1) # B, (56, 28, 14, 7), (56, 28, 14, 7), (96, 192, 384, 768)
        for blocks in self.layers: # impair block is w-msa, pair is sw-msa
            x = blocks[0](x) 
            x = blocks[1](x)
        return x.permute(0, 3, 1, 2) # B, (96, 192, 384, 768), (56, 28, 14, 7), (56, 28, 14, 7)
        
        
class SwinEncoder(nn.Module):
    def __init__(self, hidden_dim, layers, heads, channels=3, num_classes=12, head_dim=32, window_size=7, downscaling_factors=(4,2,2,2), relative_pos_embedding=True, dropout=0.0, drop_path_rate=0.1):
        super().__init__()
        
        total_layers = sum(layers)
        dp_rates = [i.item() for i in torch.linspace(0, drop_path_rate, total_layers)]
        s = 0
        dim = hidden_dim
        in_channels = channels
        self.stages = nn.ModuleList()
        for i in range(len(layers)):
            self.stages.append(
                SwinStage(in_channels=in_channels, hidden_dimension=dim, layers=layers[i], downscaling_factor=downscaling_factors[i], num_heads=heads[i], head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout, drop_path_rates=dp_rates[s : s+layers[i]])
            )
            s += layers[i]
            in_channels = dim
            dim*=2

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            
        # f1 : B, 96, H/4, W/4
        # f2 : B, 192, H/8, W/8
        # f3 : B, 384, H/16, W/16
        # f4 : B, 768, H/32, W/32
        
        return features
    
if __name__ == "__main__":
    
    net = SwinEncoder(
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
    net.eval()

    dummy_image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        features = net(dummy_image)
    for f in features:
        print(f.shape)

    params = sum(np.prod(p.size()) for p in net.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {params / 1e6:.2f} M")

