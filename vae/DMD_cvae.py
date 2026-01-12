import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

import torch
from torch.nn import functional as F
import numpy as np

from vae.normal_vae import AutoencoderKL
from vae.conditional_encoder import MaskConditionEncoder

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator

from utils.ffa_aug_dataloader import ffa_dataloader
from utils.common import get_parameters, save_config_to_yaml

from config.vae.enhancedconfig import create_config
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from os.path import join as j

from utils.multiscale_attention_system import (
    MultiScaleAttentionDiscriminator,
    SaliencyDetector,
    create_discriminators,
    compute_attention_loss,
    visualize_attention_map
)


def main():
    config = create_config('default')
    cf = save_config_to_yaml(config, config.project_dir)
    accelerator = Accelerator(**get_parameters(Accelerator, cf))

    train_dataloader = ffa_dataloader(
        config.data_path,
        config.sample_size,
        config.train_bc,
        read_channel='gray',
        data_aug=True,
        shuffle=True
    )
    val_dataloader = ffa_dataloader(
        config.eval_path,
        config.sample_size,
        config.eval_bc,
        read_channel='gray',
        data_aug=False,
        shuffle=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    up_and_down = config.up_and_down
    attention_levels = (False,) * len(up_and_down)

    vae = AutoencoderKL(
        spatial_dims=2,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_channels=config.up_and_down,
        latent_channels=4,
        num_res_blocks=config.num_res_layers,
        attention_levels=attention_levels
    )
    if len(config.vae_path):
        vae.load_state_dict(torch.load(config.vae_path))

    con_encoder = MaskConditionEncoder(
        config.in_channels,
        config.up_and_down[0],
        config.up_and_down[-1],
        stride=4
    )

    patch_discriminator = PatchDiscriminator(
        spatial_dims=2,
        num_channels=64,
        in_channels=config.out_channels,
        out_channels=1
    )
    if len(config.dis_path):
        patch_discriminator.load_state_dict(torch.load(config.dis_path))

    attention_discriminator = MultiScaleAttentionDiscriminator(
        in_channels=config.out_channels,
        feature_channels=config.attention_feature_channels
    )
    if len(config.attention_dis_path):
        attention_discriminator.load_state_dict(torch.load(config.attention_dis_path))

    vae = vae.to(device)
    con_encoder = con_encoder.to(device)
    patch_discriminator = patch_discriminator.to(device)
    attention_discriminator = attention_discriminator.to(device)

    saliency_detector = SaliencyDetector(
        alpha=config.saliency_alpha,
        threshold=config.saliency_threshold,
        min_ratio=config.saliency_detection_ratio
    )

    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="vgg")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = perceptual_loss.to(device)

    vae.requires_grad_(False).eval()
    con_encoder.train()

    optimizer_g = torch.optim.Adam(
        params=con_encoder.parameters(),
        lr=config.vae_lr
    )
    optimizer_d_patch = torch.optim.Adam(
        params=patch_discriminator.parameters(),
        lr=config.patch_discriminator_lr
    )
    optimizer_d_attention = torch.optim.Adam(
        params=attention_discriminator.parameters(),
        lr=config.attention_discriminator_lr
    )

    gen_path = j(config.project_dir, 'gen_save')
    dis_path = j(config.project_dir, 'dis_save')
    attention_dis_path = j(config.project_dir, 'attention_dis_save')
    vae_path = j(config.project_dir, 'vae_save')
    best_model_path = j(config.project_dir, 'best_model')
    save_path = j(config.project_dir, 'image_save')

    for path in [gen_path, dis_path, attention_dis_path, vae_path, best_model_path, save_path]:
        os.makedirs(path, exist_ok=True)

    if len(config.log_with):
        accelerator.init_trackers(config.experiment_name)

    phase_info = {
        'vae_warmup': True,
        'patch_discriminator_active': False,
        'attention_discriminator_active': config.force_attention_discriminator or config.skip_saliency_detection,
        'saliency_detected_epoch': None,
        'first_saliency_step': None
    }

    best_val_loss = float('inf')
    best_epoch = 0

    val_interval = config.val_inter
    save_interval = config.save_inter
    autoencoder_warm_up_n_epochs = config.autoencoder_warm_up_n_epochs

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            leave=True,
            desc=f"Epoch {epoch+1}/{config.num_epochs}"
        )

        con_encoder.train()
        vae.eval()
        patch_discriminator.train()
        attention_discriminator.train()

        if epoch + 1 > autoencoder_warm_up_n_epochs:
            phase_info['vae_warmup'] = False
            phase_info['patch_discriminator_active'] = True

        epoch_saliency_detected = False
        epoch_saliency_ratios = []

        for step, (ffa_batch, filenames) in enumerate(train_dataloader):
            ffa_images = ffa_batch.to(device).clip(-1, 1)

            optimizer_g.zero_grad(set_to_none=True)

            condition_im = con_encoder(ffa_images)
            reconstruction, mu, log_var = vae(ffa_images, condition_im=condition_im)

            recons_loss = F.mse_loss(reconstruction.float(), ffa_images.float())
            p_loss = perceptual_loss(reconstruction.float(), ffa_images.float())

            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss = kl_loss / ffa_images.numel()

            loss_vae = (config.reconstruction_weight * recons_loss +
                        config.perceptual_weight * p_loss +
                        config.kl_weight * kl_loss)

            saliency_ratio = 0.0
            if not config.skip_saliency_detection:
                has_saliency, saliency_ratio, _ = saliency_detector.detect_saliency(reconstruction)
                epoch_saliency_ratios.append(saliency_ratio)

                if has_saliency and not phase_info['attention_discriminator_active']:
                    phase_info['attention_discriminator_active'] = True
                    phase_info['saliency_detected_epoch'] = epoch + 1
                    phase_info['first_saliency_step'] = global_step
                    epoch_saliency_detected = True
            else:
                epoch_saliency_ratios.append(0.0)

            patch_generator_loss = torch.tensor(0.0, device=device)
            attention_generator_loss = torch.tensor(0.0, device=device)
            attention_loss = torch.tensor(0.0, device=device)

            if phase_info['patch_discriminator_active']:
                logits_fake = patch_discriminator(reconstruction.contiguous().float())[-1]
                patch_generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_vae += config.patch_adv_weight * patch_generator_loss

            if phase_info['attention_discriminator_active']:
                attention_output, attention_map_fake = attention_discriminator(reconstruction.contiguous().float())
                attention_generator_loss = F.binary_cross_entropy_with_logits(
                    attention_output, torch.ones_like(attention_output)
                )
                loss_vae += config.attention_adv_weight * attention_generator_loss

                if config.attention_loss_weight > 0:
                    with torch.no_grad():
                        _, attention_map_real = attention_discriminator(ffa_images.contiguous().float())
                    attention_loss = compute_attention_loss(attention_map_fake, attention_map_real)
                    loss_vae += config.attention_loss_weight * attention_loss

            loss_vae.backward()

            if hasattr(config, 'gradient_clip_norm') and config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(con_encoder.parameters(), config.gradient_clip_norm)

            optimizer_g.step()

            patch_d_loss = torch.tensor(0.0, device=device)
            if phase_info['patch_discriminator_active']:
                optimizer_d_patch.zero_grad(set_to_none=True)

                logits_fake = patch_discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                logits_real = patch_discriminator(ffa_images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

                patch_discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                patch_d_loss = config.patch_adv_weight * patch_discriminator_loss

                patch_d_loss.backward()
                optimizer_d_patch.step()

            attention_d_loss = torch.tensor(0.0, device=device)
            if phase_info['attention_discriminator_active']:
                optimizer_d_attention.zero_grad(set_to_none=True)

                fake_output, _ = attention_discriminator(reconstruction.contiguous().detach())
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_output, torch.zeros_like(fake_output)
                )

                real_output, _ = attention_discriminator(ffa_images.contiguous().detach())
                real_loss = F.binary_cross_entropy_with_logits(
                    real_output, torch.ones_like(real_output)
                )

                attention_d_loss = config.attention_adv_weight * (fake_loss + real_loss) * 0.5
                attention_d_loss.backward()
                optimizer_d_attention.step()

            progress_bar.update(1)
            loss_info = f"G_loss: {loss_vae.item():.4f}, D_loss: {patch_d_loss.item():.4f}, recons: {recons_loss.item():.4f}"
            progress_bar.set_postfix_str(loss_info)

            global_step += 1

        progress_bar.close()

        if (epoch + 1) % val_interval == 0 or epoch == config.num_epochs - 1:
            con_encoder.eval()
            vae.eval()
            total_mse_loss = 0.0

            with torch.no_grad():
                for batch_idx, (ffa_batch, filenames) in enumerate(val_dataloader):
                    ffa_images = ffa_batch.to(device)

                    condition_im = con_encoder(ffa_images)
                    val_recon, _, _ = vae(ffa_images, condition_im=condition_im)

                    mse_loss = F.mse_loss(val_recon, ffa_images)
                    total_mse_loss += mse_loss

                    if batch_idx == 0:
                        original_normalized = (ffa_images + 1) / 2
                        recon_normalized = (val_recon + 1) / 2

                        comparison = torch.cat([
                            original_normalized,
                            recon_normalized
                        ], dim=-1)

                        comparison_grid = make_grid(comparison, nrow=1).unsqueeze(0)
                        save_image(comparison_grid.clip(0, 1),
                                   j(save_path, f'epoch_{epoch + 1}_ffa_reconstruction.png'))

                average_mse_loss = total_mse_loss / len(val_dataloader)
                current_val_loss = average_mse_loss.item()

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch + 1
                    save_best_model(
                        best_model_path, vae, con_encoder, patch_discriminator,
                        attention_discriminator, best_epoch, best_val_loss,
                        config, phase_info
                    )

                del average_mse_loss, total_mse_loss, mse_loss

        if (epoch + 1) % save_interval == 0 or epoch == config.num_epochs - 1:
            save_checkpoint(
                gen_path, dis_path, attention_dis_path, vae_path,
                vae, con_encoder, patch_discriminator, attention_discriminator,
                epoch + 1
            )


def save_best_model(best_model_path, vae, con_encoder, patch_discriminator, attention_discriminator,
                    best_epoch, best_val_loss, config, phase_info):
    torch.save(vae.state_dict(), j(best_model_path, 'best_vae.pth'))
    torch.save(con_encoder.state_dict(), j(best_model_path, 'best_con_encoder.pth'))
    torch.save(patch_discriminator.state_dict(), j(best_model_path, 'best_patch_discriminator.pth'))
    torch.save(attention_discriminator.state_dict(), j(best_model_path, 'best_attention_discriminator.pth'))

    with open(j(best_model_path, 'best_model_info.txt'), 'w') as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation MSE Loss: {best_val_loss:.6f}\n")
        f.write(f"Training Mode: FFA Autoencoding (Conditional VAE with self-conditioning)\n")
        f.write(f"VAE warm-up epochs: {config.autoencoder_warm_up_n_epochs}\n")
        f.write(f"Total epochs trained: {best_epoch}\n")
        f.write(f"Saliency first detected: Epoch {phase_info['saliency_detected_epoch']}\n")
        f.write(f"PatchGAN active: {phase_info['patch_discriminator_active']}\n")
        f.write(f"Attention Discriminator active: {phase_info['attention_discriminator_active']}\n")
        f.write(f"Configuration: {config.__class__.__name__}\n")
        f.write(f"Experiment name: {config.experiment_name}\n")


def save_checkpoint(gen_path, dis_path, attention_dis_path, vae_path,
                    vae, con_encoder, patch_discriminator, attention_discriminator, epoch):
    torch.save(vae.state_dict(), j(vae_path, f'vae_epoch_{epoch}.pth'))
    torch.save(vae.state_dict(), j(vae_path, 'latest_vae.pth'))
    torch.save(con_encoder.state_dict(), j(gen_path, f'con_encoder_epoch_{epoch}.pth'))
    torch.save(patch_discriminator.state_dict(), j(dis_path, f'patch_discriminator_epoch_{epoch}.pth'))
    torch.save(attention_discriminator.state_dict(), j(attention_dis_path, f'attention_discriminator_epoch_{epoch}.pth'))
    torch.save(con_encoder.state_dict(), j(gen_path, 'latest_con_encoder.pth'))
    torch.save(patch_discriminator.state_dict(), j(dis_path, 'latest_patch_discriminator.pth'))
    torch.save(attention_discriminator.state_dict(), j(attention_dis_path, 'latest_attention_discriminator.pth'))


if __name__ == '__main__':
    main()