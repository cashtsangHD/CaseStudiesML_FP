#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:12:37 2024

@author: cashtsang
"""

import torch
from torch import nn
from torch.nn import functional as F
import tqdm



class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        return x + self.residual_layer(residue)
    
    def initiate_weight(self, value:float):
        self.conv_1.weight.data.fill_(value)
        self.conv_2.weight.data.fill_(value)

        self.conv_1.bias.data.fill_(value)
        self.conv_2.bias.data.fill_(value)
        
        if isinstance(self.residual_layer, nn.Conv2d):
            self.residual_layer.weight.data.fill_(value)
            self.residual_layer.bias.data.fill_(value)
            
        
        


class AttentionBlock(nn.Module):
    
    def __init__(self, in_channels, n_heads=1):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.attention = nn.MultiheadAttention(in_channels, n_heads, batch_first=True)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (N x C x H x W)
        residue = x
        n, c, h, w = x.shape
        x = self.groupnorm(x)
        x = x.view(n, c, h*w) # (N x C x H x W) -> (N x C x HW)
        x = x.permute(0,2,1) # (N x C x HW) -> (N x HW x C)
        x = self.attention(x,x,x,need_weights=False)[0] # (N x HW x C)
        x = x.permute(0,2,1) # (N x C x HW)
        x = x.view(n, c, h, w)
        
        x = x + residue
        
        return x





class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    
    class LayerNorm(nn.Module):
        r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
        The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
        shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
        with shape (batch_size, channels, height, width).
        """
        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.data_format = data_format
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError 
            self.normalized_shape = (normalized_shape, )
        
        def forward(self, x):
            if self.data_format == "channels_last":
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x
            
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = self.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x




class MLP(torch.nn.Module):
    
    def __init__(self, features, hidden_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(features, hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim, features)
        self.norm_1 = torch.nn.LayerNorm(features)
        self.norm_2 = torch.nn.LayerNorm(features)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm_1(x)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.norm_2(x)
        x = self.linear_2(x) + res
        return x




class PositionalEncoding(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ## x: N x seq_len x embed_dim
        device = x.device
        N, S, D = x.shape
        pos = torch.arange(0,S).to(device) # seq_len
        pos_encode = torch.arange(0, D, 2).view(-1,1).repeat(1,2).view(-1).to(device) ## embed_dim
        pos_encode = torch.pow(10000, pos_encode / D)
        pos_encode = pos[:, None] / pos_encode[None, :] # seq_len x embed_dim
        pos_encode[:, ::2] = torch.sin(pos_encode[:, ::2])
        pos_encode[:, 1::2] = torch.cos(pos_encode[:, 1::2])
        
        return pos_encode[None,:,:] + x
        
        


class PatchEmbedding(torch.nn.Module):
    
    def __init__(self, img_size, embed_dim, num_patch=8):
        super().__init__()
        self.img_size = img_size
        self.patch_len = img_size // num_patch
        self.embed_dim = embed_dim
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=embed_dim, 
                                    kernel_size=self.patch_len, 
                                    stride=self.patch_len)
        
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        return self.conv(x)
        
        