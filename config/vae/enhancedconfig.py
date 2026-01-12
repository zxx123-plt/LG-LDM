from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
import os


@dataclass
class Config:
    train_bc: int = 1
    eval_bc: int = 1
    num_epochs: int = 1000
    data_path: str = 'data_example/train'
    eval_path: str = 'data_example/test'
    
    val_inter: int = 10
    save_inter: int = 10
    sample_size: Tuple[int, int] = (512, 512)
    
    in_channels: int = 1
    out_channels: int = 1
    up_and_down: Tuple[int, int, int] = (128, 256, 512)
    num_res_layers: int = 2
    
    vae_path: str = ''
    dis_path: str = ''
    
    autoencoder_warm_up_n_epochs: int = 10
    
    split_batches: bool = False
    mixed_precision: str = 'fp16'
    log_with: str = 'tensorboard'
    project_dir: str = 'DMD_cvae_experiment'
    gradient_accumulation_steps: int = 1
    
    attention_dis_path: str = ''
    
    saliency_alpha: float = 1.5
    saliency_threshold: int = 64
    saliency_detection_ratio: float = 0.01
    
    reconstruction_weight: float = 1.0
    kl_weight: float = 1e-6
    patch_adv_weight: float = 0.01
    attention_adv_weight: float = 0.005
    perceptual_weight: float = 0.001
    attention_loss_weight: float = 0.1
    
    attention_feature_channels: int = 32
    attention_pool_scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    force_attention_discriminator: bool = False
    skip_saliency_detection: bool = False
    
    save_attention_discriminator: bool = True
    save_attention_maps: bool = True
    show_attention_visualization: bool = True
    attention_map_save_interval: int = 1
    
    log_saliency_detection: bool = True
    saliency_log_interval: int = 100
    log_attention_stats: bool = True
    
    vae_lr: float = 1e-4
    patch_discriminator_lr: float = 1e-4
    attention_discriminator_lr: float = 1e-4
    
    vae_beta1: float = 0.5
    vae_beta2: float = 0.999
    vae_weight_decay: float = 0.0
    
    experiment_name: str = 'ffa_autoencoding_multiscale_vae'
    seed: int = 42
    
    gradient_clip_norm: float = 1.0
    
    def get_phase_info(self) -> dict:
        return {
            'vae_warmup_epochs': self.autoencoder_warm_up_n_epochs,
            'patch_discriminator_start_epoch': self.autoencoder_warm_up_n_epochs + 1,
            'attention_discriminator_trigger': 'saliency_detection' if not self.skip_saliency_detection else 'immediate',
            'saliency_threshold': self.saliency_threshold,
            'training_mode': 'ffa_autoencoding'
        }
    

def create_config(config_type: str = "default", **kwargs) -> Config:
    config_map = {
        'default': Config,
    }
    
    if config_type not in config_map:
        available_types = list(config_map.keys())
        raise ValueError(f"Unknown configuration type: {config_type}. Available types: {available_types}")
    
    ConfigClass = config_map[config_type]
    config = ConfigClass()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    

    
    return config