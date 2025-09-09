"""
DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model
Based on the paper: https://arxiv.org/abs/2304.11582
Implementation adapted from: https://github.com/Yasoz/DiffTraj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ..base_model import BaseTrajectoryModel


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    """Group normalization"""
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """ResNet block for U-Net architecture"""
    
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, 
                 dropout=0.1, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, 
                                                   kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, 
                                                  kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """Attention block for U-Net"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)
        h_ = self.proj_out(h_)

        return x + h_


class Downsample(nn.Module):
    """Downsampling layer"""
    
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels, in_channels, 
                                      kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """Upsampling layer"""
    
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels, in_channels, 
                                      kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class TrajectoryUNet(nn.Module):
    """U-Net architecture for trajectory diffusion"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model parameters
        ch = getattr(config, 'ch', 128)
        out_ch = getattr(config, 'out_ch', 2)  # x, y coordinates
        ch_mult = getattr(config, 'ch_mult', [1, 2, 4])
        num_res_blocks = getattr(config, 'num_res_blocks', 2)
        attn_resolutions = getattr(config, 'attn_resolutions', [16])
        dropout = getattr(config, 'dropout', 0.1)
        in_channels = getattr(config, 'in_channels', 2)
        resolution = getattr(config, 'sequence_length', 64)
        resamp_with_conv = getattr(config, 'resamp_with_conv', True)
        
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # Downsampling
        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                       temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                     temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                     temb_channels=self.temb_ch, dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                       temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # End
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, extra_embed=None):
        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h, (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # End
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiffTrajModel(BaseTrajectoryModel):
    """
    DiffTraj: Trajectory generation using diffusion probabilistic models
    Based on "DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model"
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Diffusion parameters
        self.num_timesteps = getattr(config, 'num_timesteps', 1000)
        self.beta_start = getattr(config, 'beta_start', 0.0001)
        self.beta_end = getattr(config, 'beta_end', 0.02)
        
        # Create beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # U-Net model
        self.unet = TrajectoryUNet(config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        """Forward pass for training"""
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)
        
        # Add noise to the input
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict the noise
        predicted_noise = self.unet(x_noisy, t)
        
        return predicted_noise, noise
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: add noise to x_start"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x, t):
        """Reverse diffusion process: remove noise from x"""
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        sqrt_recip_alphas_t = (1.0 / self.alphas[t]).sqrt().view(-1, 1, 1)
        
        # Predict noise
        predicted_noise = self.unet(x, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t
            noise = torch.randn_like(x)
            return model_mean + posterior_variance_t.sqrt() * noise
    
    def generate_trajectory(self, start_pose, end_pose, num_points=None):
        """Generate trajectory from start to end pose"""
        if num_points is None:
            num_points = getattr(self.config, 'sequence_length', 64)
        
        device = next(self.parameters()).device
        
        # Start with random noise
        shape = (1, 2, num_points)  # batch_size=1, 2D coordinates, num_points
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        
        # Apply boundary conditions (start and end poses)
        trajectory = x.squeeze(0).transpose(0, 1)  # [num_points, 2]
        
        # Linear interpolation to enforce start and end poses
        if start_pose is not None and end_pose is not None:
            trajectory[0] = torch.tensor(start_pose[:2], device=device)
            trajectory[-1] = torch.tensor(end_pose[:2], device=device)
        
        return trajectory
    
    def compute_loss(self, batch):
        """Compute training loss"""
        trajectories = batch['trajectories']
        
        # Forward pass
        predicted_noise, true_noise = self.forward(trajectories)
        
        # MSE loss between predicted and true noise
        loss = self.criterion(predicted_noise, true_noise)
        
        return loss
    
    def train_step(self, batch):
        """Single training step"""
        loss = self.compute_loss(batch)
        return loss