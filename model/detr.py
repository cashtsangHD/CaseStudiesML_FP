
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




class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, n_heads=8, embed_dim=512, dropout=0.1, device="cuda"):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.pos_encoder = PositionalEncoding()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)          
            )
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.device = device
        self.to(device)
    
    def forward(self, src, target):
        ## x in dimension N x SeqLen x EmbedDim  
        ## Forward Norm Pre
        x = src # B x 1 x D
        x = self.norm_1(x)
        src2 = self.dropout_1(self.self_attention(x,x,x)[0])
        src = src + src2
        src2 = self.norm_2(src)
        target_pos = self.pos_encoder(target)
        src2 = self.dropout_2(self.cross_attention(src2, target_pos, target)[0])
        src = src + src2
        src2 = self.norm_3(src)
        src2 = self.mlp(src2)
        src = src + self.dropout_3(src2)
        return src



class DETR(nn.Module):
    
    def __init__(self, n_backbone_layers=4, n_layers=2, embed_dim=512, n_heads=4, device="cuda"):
        super().__init__()
        self.backbone = CNN(num_layers=n_backbone_layers, embed_dim=embed_dim)
        self.encode_layers = nn.ModuleList([TransformerEncoderLayer(n_heads, embed_dim, 0.1, device) for i in range(n_layers)])
        self.decode_layers = nn.ModuleList([TransformerDecoderLayer(n_heads, embed_dim, 0.1, device) for i in range(n_layers)])
        self.detection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4)
            )
        self.logits_head = nn.Linear(embed_dim, 1)
            
        self.query_embed = nn.Embedding(4, embed_dim)
        self.device = device
        self.to(device)
    
    def forward(self, x):
        
        ## x: B, C, H, W
        B = x.shape[0]
        out = self.backbone(x)
        
        b, c, nh, nw = out.shape
        out = out.view(b, c, nh*nw)
        out = out.permute(0,2,1) # B, L, D

        for layer in self.encode_layers:
            out = layer(out)
        
        query  = self.query_embed.weight.repeat(B,1,1)
        
        for layer in self.decode_layers:
            query = layer(query, out)
        
        bboxes = self.detection_head(query) # B x L x 4
        logits = self.logits_head(query) # B x L x 1

        return bboxes, logits
    
if __name__ == "__main__":
    
    img = torch.randn(4, 3, 256, 256).cuda()
    
    detr = DETR()
    
    o = detr(img)
