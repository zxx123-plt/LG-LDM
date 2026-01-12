import sys;sys.path.append('./')

from tqdm import tqdm
from utils.multi_image_dataloader import FFAcondi_form_dataloader

import torch
from torch.nn import functional as F
import argparse
import numpy as np
from generative.networks.nets import AutoencoderKL
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from utils.common import get_parameters, save_config_to_yaml, one_to_three
from config.diffusion.diffusion_config import Config
from os.path import join as j
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
import os

from utils.multi_condition_controlnet import MultiConditionControlNetEncoder

try:
    from thop import profile, clever_format
    has_thop = True
except ImportError:
    has_thop = False


class LatentGradientVarianceAlignmentLoss:
    
    def __init__(self, alpha: float = 1.0, beta: float = 1):
        self.alpha = alpha
        self.beta = beta
    
    def compute_latent_gradient_variance_loss(
        self, 
        noisy_latent: torch.Tensor, 
        denoised_latent: torch.Tensor,
        timestep: int,
        total_timesteps: int = 1000
    ):
        noisy_grad = self._compute_spatial_gradient(noisy_latent)
        denoised_grad = self._compute_spatial_gradient(denoised_latent)
        
        noisy_magnitude = torch.norm(noisy_grad, dim=1, keepdim=True)
        denoised_magnitude = torch.norm(denoised_grad, dim=1, keepdim=True)
        
        magnitude_loss = F.mse_loss(noisy_magnitude, denoised_magnitude)
        
        noisy_grad_norm = F.normalize(noisy_grad, p=2, dim=1)
        denoised_grad_norm = F.normalize(denoised_grad, p=2, dim=1)
        
        cosine_sim = torch.sum(noisy_grad_norm * denoised_grad_norm, dim=1, keepdim=True)
        direction_loss = 1.0 - cosine_sim.mean()
        
        time_weight = timestep / total_timesteps
        adaptive_alpha = self.alpha * (1.0 + time_weight)
        adaptive_beta = self.beta * (2.0 - time_weight)
        
        total_loss = adaptive_alpha * magnitude_loss + adaptive_beta * direction_loss
        
        return total_loss, {
            'magnitude_loss': magnitude_loss.item(),
            'direction_loss': direction_loss.item(),
            'time_weight': time_weight,
            'adaptive_alpha': adaptive_alpha,
            'adaptive_beta': adaptive_beta,
            'cosine_similarity': cosine_sim.mean().item()
        }
    
    def _compute_spatial_gradient(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)
        
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
        
        gradient = torch.stack([grad_x, grad_y], dim=2)
        gradient = gradient.view(batch_size, channels * 2, height, width)
        
        return gradient


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape, device="cuda"):
    if not has_thop:
        return 0, "N/A"
    
    try:
        example_input = torch.randn(input_shape).to(device)
        flops, params = profile(model, inputs=(example_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except Exception as e:
        return 0, "N/A"


def main():
    cf = save_config_to_yaml(Config, Config.project_dir)
    accelerator = Accelerator(**get_parameters(Accelerator, cf))
    
    train_dataloader = FFAcondi_form_dataloader(
        Config.data_path, 
        Config.sample_size, 
        Config.train_bc, 
        Config.mode, 
        read_channel='gray'
    )
    
    device = 'cuda'
    
    val_dataloader = FFAcondi_form_dataloader(
        Config.eval_path, 
        Config.sample_size, 
        Config.eval_bc, 
        'val',
        data_aug=False,
        read_channel='gray'
    )
    
    gva_loss_calculator = LatentGradientVarianceAlignmentLoss(
        alpha=Config.gva_alpha, 
        beta=Config.gva_beta
    )
    gva_loss_weight = Config.gva_loss_weight
    gva_start_step = Config.gva_start_step
    
    attention_levels = (False, ) * len(Config.up_and_down)
    vae = AutoencoderKL(
        spatial_dims=2, 
        in_channels=Config.in_channels, 
        out_channels=Config.out_channels, 
        num_channels=Config.up_and_down, 
        latent_channels=4,
        num_res_blocks=Config.num_res_layers, 
        attention_levels=attention_levels
    )
    vae = vae.eval().to(device)
    if len(Config.vae_resume_path):
        vae.load_state_dict(torch.load(Config.vae_resume_path))

    model = DiffusionModelUNet(
        num_res_blocks=2, 
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        num_channels=Config.sd_num_channels,
        attention_levels=Config.attention_levels,
    )
    if len(Config.sd_resume_path):
        model.load_state_dict(torch.load(Config.sd_resume_path))
    model = model.to(device)

    condition_encoder = MultiConditionControlNetEncoder(
        spatial_dims=2,
        od_channels=getattr(Config, 'od_channels', 1),
        cfp_channels=getattr(Config, 'cfp_channels', 3),
        vessel_channels=getattr(Config, 'vessel_channels', 1),
        out_channels=Config.sd_num_channels[0],
        encoder_channels=getattr(Config, 'encoder_channels', (16, 32, 64, 128)),
        fusion_type=getattr(Config, 'condition_fusion_type', 'progressive_attention'),
        use_multiscale=getattr(Config, 'use_multiscale', True),
        dropout=getattr(Config, 'encoder_dropout', 0.1)
    ).to(device)
    
    diffusion_params = count_parameters(model)
    condition_params = count_parameters(condition_encoder)
    total_params = diffusion_params + condition_params
    
    if has_thop:
        diffusion_flops, _ = count_flops(
            model, 
            input_shape=(1, 4, 64, 64),
            device=device
        )
        
        condition_flops, _ = count_flops(
            condition_encoder,
            input_shape=(
                (1, getattr(Config, 'od_channels', 1), Config.sample_size, Config.sample_size),
                (1, getattr(Config, 'cfp_channels', 3), Config.sample_size, Config.sample_size),
                (1, getattr(Config, 'vessel_channels', 1), Config.sample_size, Config.sample_size)
            ),
            device=device
        )
    
    channel_adapter = None
    
    if len(Config.controlnet_path):
        try:
            condition_encoder.load_state_dict(torch.load(Config.controlnet_path), strict=False)
        except Exception as e:
            pass
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    optimizer_con = torch.optim.Adam(params=condition_encoder.parameters(), lr=Config.lr_controlnet)
    optimizer_sd = torch.optim.Adam(params=model.parameters(), lr=Config.lr_diffusion)
    
    inferer = DiffusionInferer(scheduler)

    val_interval = Config.val_inter
    save_interval = Config.save_inter

    if len(Config.log_with):
        accelerator.init_trackers('multi_condition_train')

    global_step = 0
    latent_shape = None
    scaling_factor = Config.scaling_factor
 
    for epoch in range(Config.num_epochs):
        model.train()
        condition_encoder.train()
        epoch_loss = 0
        epoch_diffusion_loss = 0
        epoch_gva_loss = 0
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        for step, batch in enumerate(train_dataloader):
            ffa_tensor, cfp_tensor, od_tensor, vessel_tensor, ffa_names = batch
            
            ffa_image = ffa_tensor.to(device)
            cfp_condition = cfp_tensor.to(device)
            od_condition = od_tensor.to(device)
            vessel_condition = vessel_tensor.to(device)
            
            optimizer_con.zero_grad(set_to_none=True)
            optimizer_sd.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                ffa_latent = vae.encode_stage_2_inputs(ffa_image)
                ffa_latent = ffa_latent * scaling_factor
                
            latent_shape = list(ffa_latent.shape)
            latent_shape[0] = Config.eval_bc
            
            noise = torch.randn_like(ffa_latent) + 0.1 * torch.randn(
                ffa_latent.shape[0], ffa_latent.shape[1], 1, 1
            ).to(ffa_latent.device)
            
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (ffa_latent.shape[0],), device=ffa_latent.device
            ).long()
            
            ffa_noised = scheduler.add_noise(ffa_latent, noise, timesteps)
            
            multi_condition_features = condition_encoder(
                od_condition=od_condition,
                cfp_condition=cfp_condition,
                vessel_condition=vessel_condition
            )
            
            if multi_condition_features.shape[-2:] != ffa_noised.shape[-2:]:
                multi_condition_features = F.interpolate(
                    multi_condition_features, 
                    size=ffa_noised.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            if multi_condition_features.shape[1] != ffa_noised.shape[1]:
                if channel_adapter is None:
                    channel_adapter = torch.nn.Conv2d(
                        multi_condition_features.shape[1], 
                        ffa_noised.shape[1], 
                        1
                    ).to(device)
                multi_condition_features = channel_adapter(multi_condition_features)
            
            conditioned_noised = ffa_noised + multi_condition_features
            
            noise_pred = model(
                x=conditioned_noised,
                timesteps=timesteps
            )
            
            diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())
            
            gva_loss_value = 0.0
            gva_info = {}
            
            if global_step > gva_start_step:
                with torch.no_grad():
                    denoised_latents = []
                    for i in range(ffa_noised.shape[0]):
                        single_noisy = ffa_noised[i:i+1]
                        single_noise_pred = noise_pred[i:i+1]
                        single_timestep = timesteps[i:i+1]
                        
                        step_result = scheduler.step(
                            model_output=single_noise_pred,
                            timestep=single_timestep[0],
                            sample=single_noisy
                        )
                        denoised_latents.append(step_result.prev_sample)
                    
                    denoised_latent = torch.cat(denoised_latents, dim=0)
                
                total_gva_loss = 0.0
                for i in range(ffa_noised.shape[0]):
                    single_gva_loss, single_gva_info = gva_loss_calculator.compute_latent_gradient_variance_loss(
                        ffa_noised[i:i+1], 
                        denoised_latent[i:i+1],
                        timestep=timesteps[i].item(),
                        total_timesteps=scheduler.num_train_timesteps
                    )
                    
                    total_gva_loss += single_gva_loss
                    
                    if not gva_info:
                        gva_info = {k: [] for k in single_gva_info.keys()}
                    for k, v in single_gva_info.items():
                        gva_info[k].append(v)
                
                gva_loss_value = total_gva_loss / ffa_noised.shape[0]
                
                gva_info = {k: np.mean(v) for k, v in gva_info.items()}
            
            total_loss = diffusion_loss + gva_loss_weight * gva_loss_value
            
            total_loss.backward()
            optimizer_con.step()
            optimizer_sd.step()
            
            epoch_loss += total_loss.item()
            epoch_diffusion_loss += diffusion_loss.item()
            epoch_gva_loss += gva_loss_value if isinstance(gva_loss_value, float) else gva_loss_value.item()
            
            logs = {
                "total_loss": epoch_loss / (step + 1),
                "diff_loss": epoch_diffusion_loss / (step + 1),
                "gva_loss": epoch_gva_loss / (step + 1)
            }
            
            if gva_info:
                logs.update({f"gva_{k}": v for k, v in gva_info.items()})
            
            progress_bar.update()
            progress_bar.set_postfix(logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            try:
                model.eval()
                condition_encoder.eval()
                
                val_batch = next(iter(val_dataloader))
                val_ffa, val_cfp, val_od, val_vessel, val_names = val_batch
                
                val_ffa = val_ffa.to(device, dtype=torch.float32)
                val_cfp = val_cfp.to(device, dtype=torch.float32)
                val_od = val_od.to(device, dtype=torch.float32)
                val_vessel = val_vessel.to(device, dtype=torch.float32)
                
                noise = torch.randn(latent_shape, device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    val_condition_features = condition_encoder(
                        od_condition=val_od,
                        cfp_condition=val_cfp,
                        vessel_condition=val_vessel
                    )
                    
                    if val_condition_features.shape[-2:] != noise.shape[-2:]:
                        val_condition_features = F.interpolate(
                            val_condition_features, 
                            size=noise.shape[-2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                    
                    if val_condition_features.shape[1] != noise.shape[1]:
                        if channel_adapter is None:
                            channel_adapter = torch.nn.Conv2d(
                                val_condition_features.shape[1], 
                                noise.shape[1], 
                                1
                            ).to(device)
                        val_condition_features = channel_adapter(val_condition_features)
                    
                    sampling_steps = len(scheduler.timesteps)
                    
                    sampling_progress = tqdm(range(1, 1001), desc="sampling")
                    
                    for idx, t in enumerate(scheduler.timesteps):
                        conditioned_noise = noise + val_condition_features
                        
                        noise_pred = model(
                            conditioned_noise,
                            timesteps=torch.tensor([t], device=device, dtype=torch.long),
                        )
                        
                        noise, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=noise)
                        
                        current_step = int((idx + 1) / sampling_steps * 1000)
                        sampling_progress.n = current_step
                        sampling_progress.refresh()
                    
                    sampling_progress.close()

                with torch.no_grad():
                    generated_image = vae.decode_stage_2_outputs(noise / scaling_factor)
                
                def process_image_for_save(img_tensor, name=""):
                    if img_tensor.shape[1] == 1:
                        img_3ch = img_tensor.repeat(1, 3, 1, 1)
                    else:
                        img_3ch = img_tensor
                    
                    img_min, img_max = img_3ch.min(), img_3ch.max()
                    if img_max > img_min:
                        img_normalized = (img_3ch - img_min) / (img_max - img_min)
                    else:
                        img_normalized = img_3ch
                    
                    return img_normalized
                
                generated_processed = process_image_for_save(generated_image, "Generated")
                val_ffa_processed = process_image_for_save(val_ffa, "FFA")
                val_cfp_processed = process_image_for_save(val_cfp, "CFP")
                val_od_processed = process_image_for_save(val_od, "OD")
                val_vessel_processed = process_image_for_save(val_vessel, "Vessel")
                
                try:
                    combined_image = torch.cat([
                        val_ffa_processed, 
                        generated_processed, 
                        val_cfp_processed, 
                        val_od_processed, 
                        val_vessel_processed
                    ], dim=-1)
                    
                    grid_image = make_grid(combined_image, nrow=1, normalize=False, padding=2)
                    
                    grid_image = torch.clamp(grid_image, 0, 1)
                    
                    save_path = j(Config.project_dir, 'image_save')
                    os.makedirs(save_path, exist_ok=True)
                    
                    filename = f'epoch_{epoch + 1}_EnhancedMultiCondition.png'
                    full_path = j(save_path, filename)
                    
                    save_image(grid_image, full_path)
                    
                    debug_path = j(save_path, 'debug')
                    os.makedirs(debug_path, exist_ok=True)
                    
                    save_image(generated_processed[0], j(debug_path, f'epoch_{epoch + 1}_generated.png'))
                    save_image(val_ffa_processed[0], j(debug_path, f'epoch_{epoch + 1}_original.png'))
                    
                except Exception as e:
                    save_path = j(Config.project_dir, 'image_save')
                    os.makedirs(save_path, exist_ok=True)
                    save_image(generated_processed[0], j(save_path, f'epoch_{epoch + 1}_generated_only.png'))
                
            except Exception as e:
                import traceback
                traceback.print_exc()

        if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
            save_path = j(Config.project_dir, 'model_save')
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), j(save_path, 'diffusion_model.pth'))
            
            torch.save(condition_encoder.state_dict(), j(save_path, 'enhanced_condition_encoder.pth'))
            
            if channel_adapter is not None:
                torch.save(channel_adapter.state_dict(), j(save_path, 'channel_adapter.pth'))


if __name__ == '__main__':
    main()