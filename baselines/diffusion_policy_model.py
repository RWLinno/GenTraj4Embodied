"""
Diffusion Policy Model for 3D End-Effector Trajectory Generation

This module implements the Diffusion Policy approach for robotic trajectory generation,
based on the work "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
by Chi et al.

Key Features:
- Denoising Diffusion Probabilistic Models (DDPM) for trajectory generation
- U-Net architecture with temporal convolutions and attention mechanisms
- Flexible conditioning on start/goal poses and environmental context
- Support for multi-modal trajectory generation

Mathematical Foundation:
The diffusion process is formulated as:
- Forward: q(T_t | T_{t-1}) = N(T_t; √(1-β_t)T_{t-1}, β_t I)
- Reverse: p_θ(T_{t-1} | T_t, c) = N(T_{t-1}; μ_θ(T_t, t, c), Σ_θ(T_t, t, c))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import math
from .base_model import ProbabilisticGenerativeModel


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for diffusion models
    """
    
    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int = 256,
                 num_layers: int = 4, time_embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # U-Net layers
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.down_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            
            self.up_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            condition: Condition tensor [batch_size, condition_dim]
            timesteps: Timestep tensor [batch_size]
            
        Returns:
            Output tensor [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Time embedding
        time_emb = self.get_timestep_embedding(timesteps, self.time_embed_dim)
        time_emb = self.time_mlp(time_emb)  # [batch_size, hidden_dim]
        
        # Condition embedding
        cond_emb = self.condition_mlp(condition)  # [batch_size, hidden_dim]
        
        # Combine time and condition embeddings
        context = time_emb + cond_emb  # [batch_size, hidden_dim]
        context = context.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Input projection
        h = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        h = h + context
        
        # Store skip connections
        skip_connections = []
        
        # Downsampling path
        for layer in self.down_layers:
            skip_connections.append(h)
            h = layer(h) + h  # Residual connection
        
        # Upsampling path
        for layer in self.up_layers:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)
        
        # Output projection
        output = self.output_proj(h)
        
        return output
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class DDPMScheduler:
    """
    DDPM noise scheduler
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "linear",
                 prediction_type: str = "epsilon", clip_sample: bool = True):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For inference
        self.timesteps = None
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        Add noise to samples according to the noise schedule
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def set_timesteps(self, num_inference_steps: int):
        """
        Set timesteps for inference
        """
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor):
        """
        Predict the sample at the previous timestep
        """
        t = timestep
        
        if t > 0:
            prev_t = t - self.num_train_timesteps // len(self.timesteps)
        else:
            prev_t = 0
        
        # Compute alphas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "x0":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip sample if needed
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t] ** 0.5 * (1 - alpha_prod_t_prev) / beta_prod_t
        
        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # Add noise if not the last step
        if t > 0:
            noise = torch.randn_like(sample)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
            pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise
        
        class SchedulerOutput:
            def __init__(self, prev_sample):
                self.prev_sample = prev_sample
        
        return SchedulerOutput(pred_prev_sample)


class DiffusionPolicyModel(ProbabilisticGenerativeModel):
    """
    Diffusion Policy Model for trajectory generation.
    
    This class implements the core diffusion policy algorithm for generating
    robotic trajectories. It uses a U-Net architecture to learn the reverse
    diffusion process, enabling generation of diverse, high-quality trajectories
    conditioned on start and goal poses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration parameters
        arch_config = config.get('architecture', {}) if isinstance(config.get('architecture'), dict) else {}
        self.action_dim = 7  # 3D position (3) + quaternion orientation (4)
        self.observation_dim = 14  # start_pose (7) + end_pose (7)
        self.horizon = arch_config.get('horizon', 16)  # Prediction horizon
        self.num_diffusion_steps = arch_config.get('num_steps', 100)
        
        # Initialize diffusion network (U-Net architecture)
        self.network = DiffusionUNet(
            input_dim=self.action_dim,
            condition_dim=self.observation_dim,
            hidden_dim=arch_config.get('unet_dim', 256),
            num_layers=arch_config.get('num_layers', 4),
            time_embed_dim=arch_config.get('time_embed_dim', 128),
            dropout=arch_config.get('dropout', 0.1)
        )
        
        # Initialize noise scheduler for diffusion process
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule=arch_config.get('beta_schedule', 'cosine'),
            prediction_type=arch_config.get('prediction_type', 'epsilon'),
            clip_sample=arch_config.get('clip_sample', True)
        )
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training/inference
        
        Args:
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim]
            context: Optional context information [batch_size, context_dim]
            
        Returns:
            Generated trajectory [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        
        # Generate trajectory using diffusion process
        trajectory = self.generate_trajectory_tensor(conditions)
        
        return trajectory
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        Generate trajectory from start to end pose
        
        Args:
            start_pose: Starting pose [input_dim]
            end_pose: Ending pose [input_dim]
            num_points: Number of trajectory points
            
        Returns:
            Generated trajectory [num_points, output_dim]
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0).to(self.device)
            end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0).to(self.device)
            
            # Generate trajectory
            trajectory = self.forward(start_tensor, end_tensor)
            
            # Convert back to numpy
            trajectory_np = trajectory.cpu().numpy()[0]
            
            # Interpolate to desired number of points if needed
            if trajectory_np.shape[0] != num_points:
                from scipy.interpolate import interp1d
                old_indices = np.linspace(0, 1, trajectory_np.shape[0])
                new_indices = np.linspace(0, 1, num_points)
                
                interpolated = []
                for dim in range(trajectory_np.shape[1]):
                    f = interp1d(old_indices, trajectory_np[:, dim], kind='cubic')
                    interpolated.append(f(new_indices))
                
                trajectory_np = np.column_stack(interpolated)
            
            return trajectory_np
    
    def generate_trajectory_tensor(self, conditions: torch.Tensor, 
                                 num_samples: int = 1) -> torch.Tensor:
        """
        Generate trajectories using the reverse diffusion process
        """
        device = conditions.device
        batch_size = conditions.shape[0]
        
        # Initialize with pure noise
        shape = (batch_size * num_samples, self.horizon, self.action_dim)
        trajectories = torch.randn(shape, device=device)
        
        # Set up scheduler for inference
        self.scheduler.set_timesteps(self.num_diffusion_steps)
        
        # Reverse diffusion process (denoising loop)
        for i, t in enumerate(self.scheduler.timesteps):
            # Prepare timestep tensor
            timesteps = torch.full(
                (trajectories.shape[0],), t, device=device, dtype=torch.long
            )
            
            # Expand conditions if needed
            if num_samples > 1:
                expanded_conditions = conditions.repeat_interleave(num_samples, dim=0)
            else:
                expanded_conditions = conditions
            
            # Predict noise
            noise_pred = self.network(trajectories, expanded_conditions, timesteps)
            
            # Perform denoising step
            scheduler_output = self.scheduler.step(noise_pred, t, trajectories)
            trajectories = scheduler_output.prev_sample
        
        return trajectories
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute training loss for the diffusion model
        
        Args:
            predictions: Model predictions [batch_size, seq_length, output_dim]
            targets: Target trajectory [batch_size, seq_length, output_dim]
            
        Returns:
            Loss value
        """
        batch_size = targets.shape[0]
        
        # Sample random timesteps for each trajectory in the batch
        timesteps = torch.randint(
            0, self.num_diffusion_steps, (batch_size,),
            device=targets.device, dtype=torch.long
        )
        
        # Sample noise from standard Gaussian distribution
        noise = torch.randn_like(targets)
        
        # Add noise according to diffusion schedule
        noisy_trajectories = self.scheduler.add_noise(targets, noise, timesteps)
        
        # Extract conditions from kwargs or use start/end poses
        if 'conditions' in kwargs:
            conditions = kwargs['conditions']
        else:
            # Use first and last poses as conditions
            start_poses = targets[:, 0]  # [batch_size, action_dim]
            end_poses = targets[:, -1]   # [batch_size, action_dim]
            conditions = torch.cat([start_poses, end_poses], dim=-1)
        
        # Predict noise using the model
        predicted_noise = self.network(noisy_trajectories, conditions, timesteps)
        
        # Compute mean squared error loss
        mse_loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        return mse_loss


# Factory function for creating diffusion policy model
def create_diffusion_policy_model(config: Dict[str, Any]) -> DiffusionPolicyModel:
    """
    Factory function to create diffusion policy model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated diffusion policy model
    """
    return DiffusionPolicyModel(config)