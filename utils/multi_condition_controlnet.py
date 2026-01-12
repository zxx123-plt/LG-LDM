from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as F
import math
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
from torch import nn
from generative.networks.nets.diffusion_model_unet import get_down_block, get_mid_block, get_timestep_embedding


def zero_module(module):

    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResidualBlock(nn.Module):

    def __init__(self, spatial_dims: int, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels,
            out_channels=channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels,
            out_channels=channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class MultiScaleFeatureExtraction(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()

        self.branch_1x1 = Convolution(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels//4,
            strides=1, kernel_size=1, conv_only=True
        )
        self.branch_3x3 = Convolution(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels//4,
            strides=1, kernel_size=3, padding=1, conv_only=True
        )
        self.branch_5x5 = Convolution(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels//4,
            strides=1, kernel_size=5, padding=2, conv_only=True
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Convolution(
                spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels//4,
                strides=1, kernel_size=1, conv_only=True
            )
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        

        branch1 = self.branch_1x1(x)
        branch2 = self.branch_3x3(x)
        branch3 = self.branch_5x5(x)
        

        branch4 = self.branch_pool(x)
        if branch4.shape[-2:] != (h, w):
            branch4 = F.interpolate(branch4, size=(h, w), mode='bilinear', align_corners=False)
            
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class ImprovedSingleConditionEncoder(nn.Module):

    
    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        out_channels: int, 
        num_channels: Sequence[int] = (16, 32, 64, 128),
        use_multiscale: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        

        if use_multiscale:
            self.conv_in = MultiScaleFeatureExtraction(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels[0]
            )
        else:
            self.conv_in = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        

        self.blocks = nn.ModuleList([])
        self.residual_blocks = nn.ModuleList([])
        
        for i in range(len(num_channels) - 1):
            channel_in = num_channels[i]
            channel_out = num_channels[i + 1]
            
 
            self.residual_blocks.append(
                ResidualBlock(spatial_dims, channel_in, dropout)
            )
            

            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_out,
                    strides=2,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )
        

        self.final_residual = ResidualBlock(spatial_dims, num_channels[-1], dropout)
        
        self.conv_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
    
    def forward(self, x):

        x = self.conv_in(x)
        x = F.silu(x)
        

        for residual_block, downsample_block in zip(self.residual_blocks, self.blocks):
           
            x = residual_block(x)
            
            x = downsample_block(x)
            x = F.silu(x)
        

        x = self.final_residual(x)

        x = self.conv_out(x)
        return x


class CrossModalAttentionFusion(nn.Module):

    
    def __init__(
        self, 
        spatial_dims: int,
        feature_channels: int,
        num_conditions: int = 3,
        num_heads: int = 8
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        

        self.q_proj = nn.Conv2d(feature_channels, feature_channels, 1)
        self.k_proj = nn.Conv2d(feature_channels, feature_channels, 1)
        self.v_proj = nn.Conv2d(feature_channels, feature_channels, 1)
        

        self.cross_modal_conv = nn.ModuleList([
            nn.Conv2d(feature_channels * 2, feature_channels, 1) 
            for _ in range(num_conditions)
        ])
        

        self.adaptive_weights = nn.Parameter(torch.ones(num_conditions, 1, 1, 1))
        

        self.refinement = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, 1)
        )
        
    def forward(self, condition_features):
        """
        Args:
            condition_features: List[Tensor] - [od_feat, cfp_feat, vessel_feat]
            每个都是 [B, 128, 64, 64]
        """
        B, C, H, W = condition_features[0].shape
        

        enhanced_features = []
        for i, feat in enumerate(condition_features):

            other_feats = [condition_features[j] for j in range(len(condition_features)) if j != i]
            other_feat = torch.mean(torch.stack(other_feats), dim=0)  # 其他模态的平均
            

            cross_modal_input = torch.cat([feat, other_feat], dim=1)  # [B, 256, H, W]
            enhanced_feat = self.cross_modal_conv[i](cross_modal_input)  # [B, 128, H, W]
            enhanced_features.append(enhanced_feat)
        

        attention_weights = []
        for feat in enhanced_features:

            q = self.q_proj(feat).view(B, self.num_heads, self.head_dim, H*W)
            k = self.k_proj(feat).view(B, self.num_heads, self.head_dim, H*W)
            

            attn = torch.softmax(
                torch.sum(q * k, dim=2, keepdim=True) / math.sqrt(self.head_dim), 
                dim=-1
            )  # [B, num_heads, 1, H*W]
            

            attn = attn.mean(dim=1).view(B, 1, H, W)  # [B, 1, H, W]
            attention_weights.append(attn)
        

        adaptive_weights = F.softmax(self.adaptive_weights, dim=0)
        

        attention_stack = torch.stack(attention_weights, dim=0)  # [3, B, 1, H, W]
        spatial_weights = F.softmax(attention_stack, dim=0)  # 空间注意力权重
        

        final_weights = spatial_weights * adaptive_weights.view(-1, 1, 1, 1, 1)
        final_weights = final_weights / final_weights.sum(dim=0, keepdim=True)  # 重新归一化
        

        fused = sum(w * f for w, f in zip(final_weights, enhanced_features))
        

        refined = self.refinement(fused)
        
        return refined + fused 


class ProgressiveFusionModule(nn.Module):

    
    def __init__(self, spatial_dims: int, feature_channels: int):
        super().__init__()
        

        self.pairwise_fusion_1 = CrossModalAttentionFusion(
            spatial_dims, feature_channels, num_conditions=2, num_heads=4
        )
        

        self.final_fusion = CrossModalAttentionFusion(
            spatial_dims, feature_channels, num_conditions=2, num_heads=8
        )
        
    def forward(self, condition_features):

        od_feat, cfp_feat, vessel_feat = condition_features
        
        # 第一阶段：OD和CFP融合
        od_cfp_fused = self.pairwise_fusion_1([od_feat, cfp_feat])
        
        # 第二阶段：融合结果与Vessel融合
        final_fused = self.final_fusion([od_cfp_fused, vessel_feat])
        
        return final_fused


class ConditionFusionModule(nn.Module):

    
    def __init__(
        self, 
        spatial_dims: int,
        feature_channels: int,
        fusion_type: str = "attention",
        num_conditions: int = 3
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_conditions = num_conditions
        
        if fusion_type == "concat":
            self.fusion_conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=feature_channels * num_conditions,
                out_channels=feature_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
            
        elif fusion_type == "weighted_sum":
            self.condition_weights = nn.Parameter(torch.ones(num_conditions) / num_conditions)
            
        elif fusion_type == "attention":
            self.attention_conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=feature_channels,
                out_channels=1,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
            

        self.output_proj = Convolution(
            spatial_dims=spatial_dims,
            in_channels=feature_channels,
            out_channels=feature_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
    
    def forward(self, condition_features):
        if self.fusion_type == "concat":
            fused = torch.cat(condition_features, dim=1)
            fused = self.fusion_conv(fused)
            
        elif self.fusion_type == "weighted_sum":
            weights = F.softmax(self.condition_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, condition_features))
            
        elif self.fusion_type == "attention":
            attention_weights = []
            for feat in condition_features:
                attn = self.attention_conv(feat)
                attention_weights.append(attn)
            
            attention_weights = torch.stack(attention_weights, dim=0)
            attention_weights = F.softmax(attention_weights, dim=0)
            
            fused = sum(w * feat for w, feat in zip(attention_weights, condition_features))
        
        fused = self.output_proj(fused)
        return fused


class MultiConditionControlNetEncoder(nn.Module):

    
    def __init__(
        self,
        spatial_dims: int = 2,
        od_channels: int = 1,
        cfp_channels: int = 3,
        vessel_channels: int = 1,
        out_channels: int = 320,
        encoder_channels: Sequence[int] = (16, 32, 64, 128),
        fusion_type: str = "progressive_attention",  # "attention", "progressive_attention", "cross_modal"
        use_multiscale: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        

        self.od_encoder = ImprovedSingleConditionEncoder(
            spatial_dims=spatial_dims,
            in_channels=od_channels,
            out_channels=encoder_channels[-1],
            num_channels=encoder_channels,
            use_multiscale=use_multiscale,
            dropout=dropout
        )
        
        self.cfp_encoder = ImprovedSingleConditionEncoder(
            spatial_dims=spatial_dims,
            in_channels=cfp_channels,
            out_channels=encoder_channels[-1],
            num_channels=encoder_channels,
            use_multiscale=use_multiscale,
            dropout=dropout
        )
        
        self.vessel_encoder = ImprovedSingleConditionEncoder(
            spatial_dims=spatial_dims,
            in_channels=vessel_channels,
            out_channels=encoder_channels[-1],
            num_channels=encoder_channels,
            use_multiscale=use_multiscale,
            dropout=dropout
        )
        

        if fusion_type == "progressive_attention":
            self.fusion_module = ProgressiveFusionModule(
                spatial_dims=spatial_dims,
                feature_channels=encoder_channels[-1]
            )
        elif fusion_type == "cross_modal":
            self.fusion_module = CrossModalAttentionFusion(
                spatial_dims=spatial_dims,
                feature_channels=encoder_channels[-1],
                num_conditions=3
            )
        else:

            self.fusion_module = ConditionFusionModule(
                spatial_dims=spatial_dims,
                feature_channels=encoder_channels[-1],
                fusion_type=fusion_type,
                num_conditions=3
            )
        

        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1], 3, padding=1),
            nn.GroupNorm(8, encoder_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1], 1),
            nn.SiLU(),
            zero_module(nn.Conv2d(encoder_channels[-1], out_channels, 3, padding=1))
        )
        
    def forward(self, od_condition, cfp_condition, vessel_condition):
        """
        Args:
            od_condition: OD条件图像 [B, 1, 512, 512]
            cfp_condition: CFP条件图像 [B, 3, 512, 512]  
            vessel_condition: Vessel条件图像 [B, 1, 512, 512]
        Returns:
            output: 融合后的条件特征 [B, 320, 64, 64]
        """
        

        od_feat = self.od_encoder(od_condition)      # [B, 128, 64, 64]
        cfp_feat = self.cfp_encoder(cfp_condition)   # [B, 128, 64, 64]
        vessel_feat = self.vessel_encoder(vessel_condition)  # [B, 128, 64, 64]
        

        fused_features = self.fusion_module([od_feat, cfp_feat, vessel_feat])  # [B, 128, 64, 64]
        

        output = self.final_conv(fused_features)  # [B, 320, 64, 64]
        
        return output


class MultiConditionControlNet(nn.Module):

    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,

        od_channels: int = 1,
        cfp_channels: int = 3,
        vessel_channels: int = 1,
        condition_fusion_type: str = "progressive_attention",
        encoder_channels: Sequence[int] = (16, 32, 64, 128),
        use_multiscale: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))
            
        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning
        

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        

        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), 
            nn.SiLU(), 
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        

        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        

        self.multi_condition_encoder = MultiConditionControlNetEncoder(
            spatial_dims=spatial_dims,
            od_channels=od_channels,
            cfp_channels=cfp_channels,
            vessel_channels=vessel_channels,
            out_channels=num_channels[0],
            encoder_channels=encoder_channels,
            fusion_type=condition_fusion_type,
            use_multiscale=use_multiscale,
            dropout=dropout
        )
        

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        
        # 初始控制块
        controlnet_block = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=output_channel,
                out_channels=output_channel,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )
        self.controlnet_down_blocks.append(controlnet_block)
        

        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1
            
            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
            )
            
            self.down_blocks.append(down_block)
            

            for _ in range(num_res_blocks[i]):
                controlnet_block = zero_module(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=output_channel,
                        out_channels=output_channel,
                        strides=1,
                        kernel_size=1,
                        padding=0,
                        conv_only=True,
                    )
                )
                self.controlnet_down_blocks.append(controlnet_block)
            

            if not is_final_block:
                controlnet_block = zero_module(
                    Convolution(
                        spatial_dims=spatial_dims,
                        in_channels=output_channel,
                        out_channels=output_channel,
                        strides=1,
                        kernel_size=1,
                        padding=0,
                        conv_only=True,
                    )
                )
                self.controlnet_down_blocks.append(controlnet_block)
        

        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )
        

        self.controlnet_mid_block = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=output_channel,
                out_channels=output_channel,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor = None,  # 兼容原始接口
        od_condition: torch.Tensor = None,
        cfp_condition: torch.Tensor = None,
        vessel_condition: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:


        if controlnet_cond is not None and od_condition is None:
            od_condition = controlnet_cond[:, :1, :, :]
            cfp_condition = controlnet_cond.repeat(1, 1, 1, 1)
            vessel_condition = controlnet_cond[:, :1, :, :]
        

        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)
        

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb
        

        h = self.conv_in(x)

        multi_cond = self.multi_condition_encoder(
            od_condition, cfp_condition, vessel_condition
        )
        
        h += multi_cond
        

        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)
        

        h = self.middle_block(hidden_states=h, temb=emb, context=context)
        

        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)
        
        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(h)
        

        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]
        mid_block_res_sample *= conditioning_scale
        
        return down_block_res_samples, mid_block_res_sample