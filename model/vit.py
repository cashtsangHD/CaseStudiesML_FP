
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import *


class CNN(nn.Module):
    
    def __init__(self, input_dim=3, embed_dim=512, num_layers=4, device="cuda"):
        super().__init__()
        
        dim = embed_dim//(2**num_layers)
        self.input_layer = nn.Conv2d(input_dim, dim, kernel_size=3, padding=1)
        
        self.hidden_layers = nn.ModuleList()
        
        for i in range(num_layers):
            factor = 2**i
            self.hidden_layers.append(nn.Sequential(
                ResidualBlock(dim*factor, dim*factor),
                ResidualBlock(dim*factor, dim*factor),
                ResidualBlock(dim*factor, dim*factor),
                nn.Conv2d(dim*factor, dim*factor*2, kernel_size=2, stride=2)
                ))
            
        self.out_layer = nn.Conv2d(dim*factor*2, embed_dim, kernel_size=1)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        x = x + self.out_layer(x)
        
        return x



class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, n_heads=8, embed_dim=512, dropout=0.1, device="cuda"):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.pos_encoder = PositionalEncoding()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)          
            )
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.device = device
        self.to(device)
    
    def forward(self, src):
        ## x in dimension N x SeqLen x EmbedDim  
        ## Forward Norm Pre
        x = src
        x = self.norm_1(x)
        x = self.pos_encoder(x)
        src2 = self.dropout_1(self.self_attention(x,x,src)[0])
        src = src + src2
        src2 = self.norm_2(src)
        src2 = self.mlp(src2)
        src = src + self.dropout_2(src2)

        return src




class VIT(nn.Module):
    
    def __init__(self, n_backbone_layers=4, n_layers=4, embed_dim=512, n_heads=4, bias=False, device="cuda"):
        super().__init__()
        self.backbone = CNN(num_layers=n_backbone_layers, embed_dim=embed_dim)
        self.trans_layers = nn.ModuleList([TransformerEncoderLayer(n_heads, embed_dim, 0.1, device) for i in range(n_layers)])
        self.projection_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
            )
        self.temperature = nn.Parameter(torch.log(torch.tensor([0.7], requires_grad=True)))
        if bias:
            self.bias = nn.Parameter(torch.log(torch.tensor([0.2], requires_grad=True)))
        else:
            self.bias = None
        self.device = device
        self.to(device)
    
    def forward(self, x):
        
        ## x: B, C, H, W
        out = self.backbone(x)
        
        b, c, nh, nw = out.shape
        out = out.view(b, c, nh*nw)
        out = out.permute(0,2,1) # B, L, D

        for layer in self.trans_layers:
            out = layer(out)

        embed = out.mean(dim=1)
        
        return self.projection_layer(embed)
        
        

if __name__ == "__main__":
    
    img = torch.randn(4, 3, 256, 256).cuda()
    
    vit = VIT()
    
    o = vit(img)