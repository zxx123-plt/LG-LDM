

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List, Optional

class MultiScaleAttentionNetwork(nn.Module):

    def __init__(self, in_channels: int = 1, feature_channels: int = 32):
        super(MultiScaleAttentionNetwork, self).__init__()
        

        self.conv_network = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.PReLU()
        )
        

        self.pool_scales = [1, 2, 4]
        

        self.weight_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_channels, 1),
                nn.Sigmoid()
            ) for _ in self.pool_scales
        ])
        

        self.attention_fusion = nn.Sequential(
            nn.Conv2d(len(self.pool_scales), feature_channels, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, 64, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.conv_network(x)  # [B, feature_channels, H, W]
        
        attention_maps = []
        

        for i, scale in enumerate(self.pool_scales):
            if scale == 1:
               
                pooled_features = F.adaptive_avg_pool2d(features, 1)
            else:
                
                pooled_features = F.avg_pool2d(features, scale, scale)
                pooled_features = F.adaptive_avg_pool2d(pooled_features, 1)
            
            
            pooled_features_flat = pooled_features.view(pooled_features.size(0), -1)
            weights = self.weight_networks[i](pooled_features_flat)  # [B, 1]
            
            
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            weighted_features = features * weights  # 广播乘法
            
          
            attention_map = torch.sum(weighted_features, dim=1, keepdim=True)  # [B, 1, H, W]
            
            
            attention_map_min = attention_map.view(attention_map.size(0), -1).min(dim=1, keepdim=True)[0]
            attention_map_max = attention_map.view(attention_map.size(0), -1).max(dim=1, keepdim=True)[0]
            attention_map_min = attention_map_min.unsqueeze(-1).unsqueeze(-1)
            attention_map_max = attention_map_max.unsqueeze(-1).unsqueeze(-1)
            
            attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min + 1e-8)
            
            attention_maps.append(attention_map)
        
        
        multi_scale_attention = torch.cat(attention_maps, dim=1)  # [B, num_scales, H, W]
        
        
        final_attention = self.attention_fusion(multi_scale_attention)  # [B, 1, H, W]
        
        return final_attention


class MultiScaleAttentionDiscriminator(nn.Module):

    def __init__(self, in_channels: int = 1, feature_channels: int = 32):
        super(MultiScaleAttentionDiscriminator, self).__init__()
        

        self.attention_network = MultiScaleAttentionNetwork(in_channels, feature_channels)
        

        self.discriminator = nn.Sequential(

            nn.Conv2d(in_channels + 1, 64, 3, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 128, 3, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 256, 3, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        attention_map = self.attention_network(x)
        

        x_with_attention = torch.cat([x, attention_map], dim=1)
        

        features = self.discriminator(x_with_attention)
        output = self.classifier(features)
        
        return output, attention_map


class SaliencyDetector:

    def __init__(self, alpha: float = 1.5, threshold: int = 64, min_ratio: float = 0.01):

        self.alpha = alpha
        self.threshold = threshold
        self.min_ratio = min_ratio
        

    def compute_saliency_map(self, image: np.ndarray) -> np.ndarray:

        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        

        img_float = image.astype(np.float32)
        

        background = cv2.medianBlur(image, 51).astype(np.float32)
        

        filtered_image = cv2.GaussianBlur(img_float, (7, 7), 1.0)
        

        saliency_map = self.alpha * (filtered_image - background)
        

        saliency_map = np.clip(saliency_map, 0, 255).astype(np.uint8)
        
        return saliency_map
    
    def detect_saliency(self, image_tensor: torch.Tensor) -> Tuple[bool, float, Optional[List[np.ndarray]]]:

        batch_size = image_tensor.size(0)
        has_saliency_list = []
        saliency_ratios = []
        saliency_maps = []
        

        with torch.no_grad():

            image_tensor_detached = image_tensor.detach().cpu()
            
            for i in range(batch_size):
                try:
 
                    img = image_tensor_detached[i].squeeze().numpy()
                    

                    saliency_map = self.compute_saliency_map(img)
                    saliency_maps.append(saliency_map)
                    

                    high_saliency_pixels = (saliency_map > self.threshold).sum()
                    total_pixels = saliency_map.size
                    saliency_ratio = high_saliency_pixels / total_pixels
                    
                    has_saliency = saliency_ratio > self.min_ratio
                    
                    has_saliency_list.append(has_saliency)
                    saliency_ratios.append(saliency_ratio)
                    
                except Exception as e:

                    has_saliency_list.append(False)
                    saliency_ratios.append(0.0)

                    default_map = np.zeros((64, 64), dtype=np.uint8)
                    saliency_maps.append(default_map)
        

        batch_has_saliency = any(has_saliency_list)
        avg_saliency_ratio = np.mean(saliency_ratios) if saliency_ratios else 0.0
        
        return batch_has_saliency, avg_saliency_ratio, saliency_maps
    
    def detect_saliency_safe(self, image_tensor: torch.Tensor) -> Tuple[bool, float]:

        batch_size = image_tensor.size(0)
        saliency_ratios = []
        

        with torch.no_grad():

            image_tensor_detached = image_tensor.detach().cpu()
            
            for i in range(batch_size):
                try:
                    
                    img = image_tensor_detached[i].squeeze().numpy()
                    
                   
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    
                    img_mean = img.mean()
                    high_intensity_pixels = (img > (img_mean + self.threshold)).sum()
                    total_pixels = img.size
                    saliency_ratio = high_intensity_pixels / total_pixels
                    
                    saliency_ratios.append(saliency_ratio)
                    
                except Exception as e:
                    
                    saliency_ratios.append(0.0)

        avg_saliency_ratio = np.mean(saliency_ratios) if saliency_ratios else 0.0
        batch_has_saliency = avg_saliency_ratio > self.min_ratio
        
        return batch_has_saliency, avg_saliency_ratio
    
    def get_saliency_stats(self, saliency_map: np.ndarray) -> dict:

        total_pixels = saliency_map.size
        

        high_threshold = 150
        very_high_threshold = 200
        
        high_saliency_pixels = (saliency_map > high_threshold).sum()
        very_high_saliency_pixels = (saliency_map > very_high_threshold).sum()
        threshold_pixels = (saliency_map > self.threshold).sum()
        
        stats = {
            'mean': float(saliency_map.mean()),
            'std': float(saliency_map.std()),
            'min': int(saliency_map.min()),
            'max': int(saliency_map.max()),
            'threshold_ratio': float(threshold_pixels / total_pixels),
            'high_saliency_ratio': float(high_saliency_pixels / total_pixels),
            'very_high_saliency_ratio': float(very_high_saliency_pixels / total_pixels)
        }
        
        return stats


def create_discriminators(config) -> Tuple[nn.Module, MultiScaleAttentionDiscriminator]:

    from generative.networks.nets import PatchDiscriminator
    

    patch_discriminator = PatchDiscriminator(
        spatial_dims=2, 
        num_channels=64, 
        in_channels=config.out_channels, 
        out_channels=1
    )
    

    attention_discriminator = MultiScaleAttentionDiscriminator(
        in_channels=config.out_channels,
        feature_channels=getattr(config, 'attention_feature_channels', 32)
    )
    
    return patch_discriminator, attention_discriminator

def compute_attention_loss(attention_map_fake: torch.Tensor, 
                          attention_map_real: torch.Tensor) -> torch.Tensor:

    return F.mse_loss(attention_map_fake, attention_map_real)

def visualize_attention_map(attention_map: torch.Tensor, 
                           save_path: Optional[str] = None) -> np.ndarray:


    if isinstance(attention_map, torch.Tensor):
        if attention_map.dim() == 4:
            attention_map = attention_map.squeeze(0).squeeze(0)
        elif attention_map.dim() == 3:
            attention_map = attention_map.squeeze(0)
        

        if attention_map.requires_grad:
            attention_map = attention_map.detach().cpu().numpy()
        else:
            attention_map = attention_map.cpu().numpy()
    

    attention_map = (attention_map * 255).astype(np.uint8)
    

    vis_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    

    if save_path:
        cv2.imwrite(save_path, vis_map)
    
    return vis_map

def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:

    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()





    



    


