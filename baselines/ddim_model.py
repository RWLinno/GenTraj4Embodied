"""
DDIM: Denoising Diffusion Implicit Models
Based on the paper: "Denoising Diffusion Implicit Models" (Song et al., 2021)
arXiv: https://arxiv.org/abs/2010.02502
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ..base_model import BaseTrajectoryModel


class SimpleUNet(nn.Module):
    """Simplified U-Net for DDIM trajectory generation"""
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.input_dim = getattr(config, 'input_dim', 7)  # 7D pose
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_layers = getattr(config, 'num_layers', 4)
        self.time_embed_dim = getattr(config, 'time_embed_dim', 128)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU()
                )
            )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Skip connection
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU()
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)
        
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Create sinusoidal timestep embeddings"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb
    
    def forward(self, x, t):
        """
        Forward pass
        Args:
            x: [batch_size, sequence_length, input_dim]
            t: [batch_size] timesteps
        """
        batch_size, seq_len, _ = x.shape
        
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # [batch_size, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Input projection
        h = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        h = h + t_emb
        
        # Encoder with skip connections
        skip_connections = []
        for layer in self.encoder_layers:
            skip_connections.append(h)
            h = layer(h) + h  # Residual connection
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)
        
        # Output projection
        output = self.output_proj(h)
        
        return output


class DDIMModel(BaseTrajectoryModel):
    """
    DDIM: Denoising Diffusion Implicit Models for trajectory generation
    
    Key differences from DDPM:
    1. Deterministic sampling process (can be made stochastic)
    2. Faster sampling with fewer steps
    3. Implicit generative process
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # DDIM parameters
        self.num_train_timesteps = getattr(config, 'num_timesteps', 1000)
        self.num_inference_steps = getattr(config, 'num_inference_steps', 50)  # Much fewer steps
        self.beta_start = getattr(config, 'beta_start', 0.0001)
        self.beta_end = getattr(config, 'beta_end', 0.02)
        self.eta = getattr(config, 'eta', 0.0)  # 0 = deterministic, 1 = DDPM
        
        # Create noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For inference, we'll use a subset of timesteps
        self.inference_timesteps = None
        
        # U-Net model
        self.unet = SimpleUNet(config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def set_inference_timesteps(self, num_inference_steps=None):
        """Set up timesteps for DDIM sampling"""
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
            
        # Create evenly spaced timesteps for inference
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.inference_timesteps = torch.arange(0, self.num_train_timesteps, step_ratio)
        
        if self.inference_timesteps[-1] != self.num_train_timesteps - 1:
            self.inference_timesteps = torch.cat([
                self.inference_timesteps, 
                torch.tensor([self.num_train_timesteps - 1])
            ])
    
    def forward(self, x):
        """Forward pass for training"""
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=x.device)
        
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
        
        # Move tensors to same device as x_start
        alphas_cumprod = self.alphas_cumprod.to(x_start.device)
        
        sqrt_alphas_cumprod_t = alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[t]).sqrt()
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def ddim_step(self, x_t, t, t_prev, predicted_noise):
        """Single DDIM sampling step"""
        device = x_t.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        # Current and previous alpha_cumprod
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        
        # Reshape for broadcasting
        while len(alpha_prod_t.shape) < len(x_t.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Compute predicted original sample
        pred_original_sample = (x_t - beta_prod_t.sqrt() * predicted_noise) / alpha_prod_t.sqrt()
        
        # Compute variance
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = self.eta * variance.sqrt()
        
        # Compute predicted sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * predicted_noise
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if self.eta > 0:
            noise = torch.randn_like(x_t)
            prev_sample = prev_sample + std_dev_t * noise
            
        return prev_sample
    
    def generate_trajectory(self, start_pose, end_pose, num_points=None):
        """Generate trajectory using DDIM sampling"""
        if num_points is None:
            num_points = getattr(self.config, 'sequence_length', 64)
        
        device = next(self.parameters()).device
        
        # Set up inference timesteps
        self.set_inference_timesteps()
        timesteps = self.inference_timesteps.to(device)
        
        # Start with random noise
        shape = (1, num_points, self.config.input_dim)
        x = torch.randn(shape, device=device)
        
        # DDIM sampling loop
        for i, t in enumerate(reversed(timesteps)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.unet(x, t_tensor)
            
            # Get previous timestep
            t_prev = timesteps[len(timesteps) - i - 2] if i < len(timesteps) - 1 else -1
            
            # DDIM step
            x = self.ddim_step(x, t, t_prev, predicted_noise)
        
        trajectory = x.squeeze(0)  # [num_points, input_dim]
        
        # Apply boundary conditions if provided
        if start_pose is not None:
            trajectory[0] = torch.tensor(start_pose, device=device, dtype=trajectory.dtype)
        if end_pose is not None:
            trajectory[-1] = torch.tensor(end_pose, device=device, dtype=trajectory.dtype)
        
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