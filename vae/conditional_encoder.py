import torch
from torch import nn
import torch.nn.functional as F


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class VascularFeatureEnhancer(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        
        group_size = max(1, channels // 4)
        

        self.fine_vessel = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=group_size),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.medium_vessel = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2, groups=group_size),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.coarse_vessel = nn.Sequential(
            nn.Conv2d(channels, channels, 7, 1, 3, groups=group_size),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.branch_vessel = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, groups=group_size),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.fine_vessel(x)
        medium = self.medium_vessel(x)
        coarse = self.coarse_vessel(x)
        branch = self.branch_vessel(x)
        
        vascular_features = torch.cat([fine, medium, coarse, branch], dim=1)
        
        enhanced = self.fusion(vascular_features)
        
        output = x + self.residual_weight * enhanced
        return output


class MaskConditionEncoder(nn.Module):

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 192,
        res_ch: int = 768,
        stride: int = 16,
    ) -> None:
        super().__init__()
        
        channels = []
        stride_ = stride
        while stride_ > 1:
            stride_ = stride_ // 2
            in_ch_ = out_ch * 2
            if out_ch > res_ch:
                out_ch = res_ch
            if stride_ == 1:
                in_ch_ = res_ch
            channels.append((in_ch_, out_ch))
            out_ch *= 2
        
        out_channels = []
        for _in_ch, _out_ch in channels:
            out_channels.append(_out_ch)
        out_channels.append(channels[-1][0])
        
        layers = []
        vascular_enhancers = []
        
        in_ch_ = in_ch
        for l in range(len(out_channels)):
            out_ch_ = out_channels[l]
            
            # Conv → BatchNorm → SiLU
            if l == 0:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch_),
                    nn.SiLU()
                )
            else:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch_),
                    nn.SiLU()
                )
            
            layers.append(conv_block)
            
            vascular_enhancers.append(
                VascularFeatureEnhancer(out_ch_)
            )
            
            in_ch_ = out_ch_
        
        self.layers = nn.Sequential(*layers)
        self.vascular_enhancers = nn.ModuleList(vascular_enhancers)
        self.layers = zero_module(self.layers)
    
    def forward(self, x: torch.Tensor, mask=None) -> dict:
        out = {}
        for l in range(len(self.layers)):
            layer = self.layers[l]
            vascular_enhancer = self.vascular_enhancers[l]
            
            x = layer(x)
            
            x_enhanced = vascular_enhancer(x)
            
            out[str(tuple(x_enhanced.shape))] = x_enhanced
            
            x = torch.relu(x_enhanced)
        
        return out

