"""
Diffusion Policy Model for 3D End-Effector Trajectory Generation

This module implements the Diffusion Policy approach for robotic trajectory generation,
based on the seminal work "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
by Chi et al. The implementation includes standard diffusion policy, conditional variants,
and hierarchical extensions for complex manipulation tasks.

Key Features:
- Denoising Diffusion Probabilistic Models (DDPM) for trajectory generation
- U-Net architecture with temporal convolutions and attention mechanisms
- Flexible conditioning on start/goal poses and environmental context
- Support for multi-modal trajectory generation
- Hierarchical generation for long-horizon planning

Mathematical Foundation:
The diffusion process is formulated as:
- Forward: q(T_t | T_{t-1}) = N(T_t; √(1-β_t)T_{t-1}, β_t I)
- Reverse: p_θ(T_{t-1} | T_t, c) = N(T_{t-1}; μ_θ(T_t, t, c), Σ_θ(T_t, t, c))

Authors: Research Team
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import math
from .diffusion_policy_network import DiffusionUNet
from .diffusion_policy_scheduler import DDPMScheduler


class DiffusionPolicyModel(nn.Module):
    """
    Diffusion Policy Model for trajectory generation.
    
    This class implements the core diffusion policy algorithm for generating
    robotic trajectories. It uses a U-Net architecture to learn the reverse
    diffusion process, enabling generation of diverse, high-quality trajectories
    conditioned on start and goal poses.
    
    Architecture:
    - U-Net backbone with temporal convolutions
    - FiLM conditioning layers for observation integration
    - Positional encoding for temporal structure
    - DDPM scheduler for noise management
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Diffusion Policy model.
        
        Args:
            config: Model configuration containing:
                - architecture: Network architecture parameters
                - training: Training hyperparameters
                - diffusion: Diffusion process parameters
        """
        super().__init__()
        self.config = config
        
        # Add required attributes for compatibility
        self.architecture = config.get('architecture', 'diffusion_unet')
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        
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
        
        # Positional encoding for temporal structure
        self.pos_encoding = PositionalEncoding(
            d_model=self.action_dim,
            max_len=arch_config.get('max_trajectory_length', 1000)
        )
        
        # Optional: Learnable trajectory embedding
        if arch_config.get('use_trajectory_embedding', False):
            self.trajectory_embedding = nn.Linear(self.action_dim, self.action_dim)
        else:
            self.trajectory_embedding = None
            
        # Training statistics for normalization
        self.register_buffer('trajectory_mean', torch.zeros(self.action_dim))
        self.register_buffer('trajectory_std', torch.ones(self.action_dim))
        
    def forward(self, trajectories: torch.Tensor, conditions: torch.Tensor, 
                timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the diffusion model.
        
        This method performs the core denoising operation, predicting either
        the noise to be removed (epsilon prediction) or the clean trajectory
        (x0 prediction) based on the current noisy trajectory and conditioning.
        
        Args:
            trajectories: Noisy trajectory tensor [batch_size, horizon, action_dim]
            conditions: Conditioning information [batch_size, condition_dim]
            timesteps: Diffusion timesteps [batch_size] (optional, sampled if None)
            
        Returns:
            Predicted noise or clean trajectory [batch_size, horizon, action_dim]
        """
        batch_size = trajectories.shape[0]
        
        # Sample random timesteps if not provided (training mode)
        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_diffusion_steps, (batch_size,), 
                device=trajectories.device, dtype=torch.long
            )
        
        # Normalize trajectories (optional, for training stability)
        if self.training and hasattr(self, 'normalize_trajectories'):
            trajectories = self._normalize_trajectories(trajectories)
        
        # Add positional encoding to capture temporal structure
        trajectories_encoded = self.pos_encoding(trajectories)
        
        # Optional trajectory embedding
        if self.trajectory_embedding is not None:
            trajectories_encoded = self.trajectory_embedding(trajectories_encoded)
        
        # Pass through U-Net with conditioning
        prediction = self.network(trajectories_encoded, conditions, timesteps)
        
        return prediction
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for the diffusion model.
        
        Implements the standard DDPM training objective:
        L = E[||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]
        
        Args:
            batch: Training batch containing:
                - trajectory: Ground truth trajectories [batch_size, seq_len, action_dim]
                - start_pose: Start poses [batch_size, 7]
                - end_pose: Goal poses [batch_size, 7]
                - task_id: Task identifiers (optional)
                - modality: Behavioral modality (optional)
                
        Returns:
            Dictionary containing loss components:
                - loss: Total training loss
                - mse_loss: Mean squared error component
                - additional metrics for monitoring
        """
        # Extract and prepare data
        trajectories = batch['trajectory'][:, :self.horizon]  # Truncate to horizon
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        
        batch_size = trajectories.shape[0]
        
        # Sample random timesteps for each trajectory in the batch
        timesteps = torch.randint(
            0, self.num_diffusion_steps, (batch_size,),
            device=trajectories.device, dtype=torch.long
        )
        
        # Sample noise from standard Gaussian distribution
        noise = torch.randn_like(trajectories)
        
        # Add noise according to diffusion schedule
        noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
        
        # Predict noise using the model
        predicted_noise = self.forward(noisy_trajectories, conditions, timesteps)
        
        # Compute target based on prediction type
        if self.scheduler.prediction_type == 'epsilon':
            target = noise  # Predict the noise
        elif self.scheduler.prediction_type == 'x0':
            target = trajectories  # Predict the clean trajectory
        elif self.scheduler.prediction_type == 'v':
            # Velocity parameterization: v = α_t * ε - σ_t * x_0
            alpha_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1)
            sigma_t = (1 - alpha_t).sqrt()
            target = alpha_t.sqrt() * noise - sigma_t * trajectories
        else:
            raise ValueError(f"Unknown prediction type: {self.scheduler.prediction_type}")
        
        # Compute mean squared error loss
        mse_loss = F.mse_loss(predicted_noise, target, reduction='mean')
        
        # Optional: Add additional loss components
        loss_dict = {
            'loss': mse_loss,
            'mse_loss': mse_loss
        }
        
        # Optional: Add pose consistency loss (ensure start/end pose matching)
        if self.config.get('use_pose_consistency_loss', False):
            pose_loss = self._compute_pose_consistency_loss(predicted_noise, batch)
            loss_dict['pose_loss'] = pose_loss
            loss_dict['loss'] = mse_loss + 0.1 * pose_loss
        
        # Optional: Add smoothness regularization
        if self.config.get('use_smoothness_loss', False):
            smoothness_loss = self._compute_smoothness_loss(predicted_noise)
            loss_dict['smoothness_loss'] = smoothness_loss
            loss_dict['loss'] += 0.01 * smoothness_loss
        
        return loss_dict
    
    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, num_samples: int = 1,
                generator: Optional[torch.Generator] = None,
                guidance_scale: float = 1.0,
                num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate trajectories using the reverse diffusion process.
        
        This method implements the DDPM sampling algorithm, starting from
        pure noise and iteratively denoising to produce clean trajectories.
        Supports classifier-free guidance for improved conditioning.
        
        Args:
            conditions: Conditioning tensor [batch_size, condition_dim]
            num_samples: Number of samples per condition
            generator: Random number generator for reproducibility
            guidance_scale: Strength of classifier-free guidance (>1.0 for stronger conditioning)
            num_inference_steps: Number of denoising steps (defaults to training steps)
            
        Returns:
            Generated trajectories [batch_size * num_samples, horizon, action_dim]
        """
        device = conditions.device
        batch_size = conditions.shape[0]
        
        # Expand conditions to match number of samples
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
        
        # Initialize with pure noise
        shape = (batch_size * num_samples, self.horizon, self.action_dim)
        trajectories = torch.randn(shape, device=device, generator=generator)
        
        # Set up scheduler for inference
        inference_steps = num_inference_steps or self.num_diffusion_steps
        self.scheduler.set_timesteps(inference_steps)
        
        # Reverse diffusion process (denoising loop)
        for i, t in enumerate(self.scheduler.timesteps):
            # Prepare timestep tensor
            timesteps = torch.full(
                (trajectories.shape[0],), t, device=device, dtype=torch.long
            )
            
            # Predict noise
            noise_pred = self.forward(trajectories, conditions, timesteps)
            
            # Apply classifier-free guidance if enabled
            if guidance_scale != 1.0 and hasattr(self, '_unconditional_forward'):
                # Predict unconditional noise
                unconditional_noise = self._unconditional_forward(trajectories, timesteps)
                # Apply guidance
                noise_pred = unconditional_noise + guidance_scale * (noise_pred - unconditional_noise)
            
            # Perform denoising step
            scheduler_output = self.scheduler.step(noise_pred, t, trajectories)
            trajectories = scheduler_output.prev_sample
            
            # Optional: Apply trajectory constraints
            if hasattr(self, '_apply_constraints'):
                trajectories = self._apply_constraints(trajectories, conditions)
        
        return trajectories
    
    @torch.no_grad()
    def sample(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
               num_samples: int = 1, **kwargs) -> torch.Tensor:
        """
        Sample trajectories from start to end poses.
        
        Convenience method for trajectory generation with pose conditioning.
        
        Args:
            start_pose: Starting poses [batch_size, 7]
            end_pose: Goal poses [batch_size, 7]
            num_samples: Number of trajectory samples per pose pair
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            Sampled trajectories [batch_size * num_samples, horizon, action_dim]
        """
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        return self.generate(conditions, num_samples, **kwargs)
    
    def get_action_sequence(self, trajectories: torch.Tensor, 
                           current_step: int = 0, 
                           action_horizon: int = 8) -> torch.Tensor:
        """
        Extract action sequence for online control (receding horizon).
        
        This method implements the receding horizon control strategy used
        in Diffusion Policy, where only a subset of the generated trajectory
        is executed before replanning.
        
        Args:
            trajectories: Generated trajectory [batch_size, horizon, action_dim]
            current_step: Current execution step
            action_horizon: Number of actions to extract
            
        Returns:
            Action sequence [batch_size, action_horizon, action_dim]
        """
        batch_size, seq_len, action_dim = trajectories.shape
        
        # Extract actions from current step
        if current_step < seq_len:
            end_step = min(current_step + action_horizon, seq_len)
            actions = trajectories[:, current_step:end_step]
            
            # Pad if necessary
            if actions.shape[1] < action_horizon:
                last_action = actions[:, -1:].repeat(1, action_horizon - actions.shape[1], 1)
                actions = torch.cat([actions, last_action], dim=1)
        else:
            # If beyond trajectory, repeat last action
            actions = trajectories[:, -1:].repeat(1, action_horizon, 1)
        
        return actions
    
    def _normalize_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Normalize trajectories using running statistics."""
        return (trajectories - self.trajectory_mean) / (self.trajectory_std + 1e-8)
    
    def _denormalize_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Denormalize trajectories."""
        return trajectories * self.trajectory_std + self.trajectory_mean
    
    def _compute_pose_consistency_loss(self, prediction: torch.Tensor, 
                                     batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss to ensure generated trajectories match start/end poses."""
        # Extract first and last poses from prediction
        if self.scheduler.prediction_type == 'x0':
            first_pose = prediction[:, 0]  # Start pose
            last_pose = prediction[:, -1]   # End pose
            
            start_loss = F.mse_loss(first_pose, batch['start_pose'])
            end_loss = F.mse_loss(last_pose, batch['end_pose'])
            
            return start_loss + end_loss
        return torch.tensor(0.0, device=prediction.device)
    
    def _compute_smoothness_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute smoothness regularization loss."""
        # Second-order differences (acceleration)
        if trajectories.shape[1] >= 3:
            accel = trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2]
            return torch.mean(torch.norm(accel, dim=-1))
        return torch.tensor(0.0, device=trajectories.device)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for trajectory sequences.
    
    Adds positional information to trajectory waypoints to help the model
    understand temporal structure and ordering.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension (should match action_dim)
            max_len: Maximum sequence length supported
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # For odd dimensions, we need to handle sin/cos pairs carefully
        # We'll use (d_model + 1) // 2 to ensure we have enough pairs
        num_pairs = (d_model + 1) // 2
        div_term = torch.exp(torch.arange(0, num_pairs, dtype=torch.float) * 
                           (-math.log(10000.0) / num_pairs))
        
        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term[:d_model//2 + d_model%2])
        
        # Apply cos to odd indices (1, 3, 5, ...) if they exist
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


class ConditionalDiffusionPolicy(DiffusionPolicyModel):
    """
    Enhanced Diffusion Policy with sophisticated conditioning mechanisms.
    
    This variant supports more complex conditioning beyond simple pose pairs,
    including environmental context, task specifications, and learned embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Enhanced condition encoder
        condition_hidden = config['architecture'].get('condition_hidden_dim', 256)
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, condition_hidden),
            nn.LayerNorm(condition_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(condition_hidden, condition_hidden),
            nn.LayerNorm(condition_hidden),
            nn.ReLU(),
            nn.Linear(condition_hidden, self.observation_dim)
        )
        
        # Optional: Task embedding for multi-task learning
        if config.get('num_tasks', 0) > 0:
            self.task_embedding = nn.Embedding(
                config['num_tasks'], 
                config['architecture'].get('task_embed_dim', 64)
            )
            self.observation_dim += config['architecture'].get('task_embed_dim', 64)
        else:
            self.task_embedding = None
    
    def encode_conditions(self, conditions: torch.Tensor, 
                         task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode conditioning information with optional task embeddings.
        
        Args:
            conditions: Raw conditioning tensor
            task_ids: Task identifiers for multi-task learning
            
        Returns:
            Encoded conditioning tensor
        """
        encoded = self.condition_encoder(conditions)
        
        # Add task embeddings if available
        if self.task_embedding is not None and task_ids is not None:
            task_embeds = self.task_embedding(task_ids)
            encoded = torch.cat([encoded, task_embeds], dim=-1)
        
        return encoded
    
    def forward(self, trajectories: torch.Tensor, conditions: torch.Tensor,
                timesteps: Optional[torch.Tensor] = None,
                task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with enhanced conditioning."""
        encoded_conditions = self.encode_conditions(conditions, task_ids)
        return super().forward(trajectories, encoded_conditions, timesteps)


class HierarchicalDiffusionPolicy(DiffusionPolicyModel):
    """
    Hierarchical Diffusion Policy for long-horizon trajectory generation.
    
    Implements a two-stage generation process:
    1. Generate sparse keypoints that capture high-level trajectory structure
    2. Generate dense trajectory conditioned on keypoints for detailed motion
    
    This approach is particularly effective for complex manipulation tasks
    requiring long-horizon planning and multi-step reasoning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_keypoints = config['architecture'].get('num_keypoints', 4)
        keypoint_steps = config['architecture'].get('keypoint_diffusion_steps', 50)
        
        # Keypoint generation network (coarser, fewer parameters)
        self.keypoint_network = DiffusionUNet(
            input_dim=self.action_dim,
            condition_dim=self.observation_dim,
            hidden_dim=config['architecture'].get('unet_dim', 256) // 2,
            num_layers=config['architecture'].get('num_layers', 3),
            time_embed_dim=config['architecture'].get('time_embed_dim', 64)
        )
        
        # Keypoint scheduler (fewer steps for efficiency)
        self.keypoint_scheduler = DDPMScheduler(
            num_train_timesteps=keypoint_steps,
            beta_schedule=config['architecture'].get('beta_schedule', 'linear'),
            prediction_type='epsilon'
        )
        
        # Dense trajectory refinement network
        keypoint_condition_dim = self.observation_dim + self.action_dim * self.num_keypoints
        self.refinement_network = DiffusionUNet(
            input_dim=self.action_dim,
            condition_dim=keypoint_condition_dim,
            hidden_dim=config['architecture'].get('unet_dim', 256),
            num_layers=config['architecture'].get('num_layers', 4),
            time_embed_dim=config['architecture'].get('time_embed_dim', 128)
        )
        
        # Keypoint interpolation for trajectory initialization
        self.keypoint_interpolator = nn.Sequential(
            nn.Linear(self.action_dim * self.num_keypoints, 256),
            nn.ReLU(),
            nn.Linear(256, self.horizon * self.action_dim)
        )
    
    @torch.no_grad()
    def generate_hierarchical(self, conditions: torch.Tensor, 
                            num_samples: int = 1,
                            generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Generate trajectories using hierarchical approach.
        
        Args:
            conditions: Conditioning tensor [batch_size, condition_dim]
            num_samples: Number of samples per condition
            generator: Random number generator
            
        Returns:
            Generated trajectories [batch_size * num_samples, horizon, action_dim]
        """
        device = conditions.device
        batch_size = conditions.shape[0]
        
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
        
        # Stage 1: Generate keypoints
        keypoint_shape = (batch_size * num_samples, self.num_keypoints, self.action_dim)
        keypoints = torch.randn(keypoint_shape, device=device, generator=generator)
        
        # Set keypoint scheduler
        self.keypoint_scheduler.set_timesteps(self.keypoint_scheduler.num_train_timesteps)
        
        # Denoise keypoints
        for t in self.keypoint_scheduler.timesteps:
            timesteps = torch.full((keypoints.shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.keypoint_network(keypoints, conditions, timesteps)
            keypoints = self.keypoint_scheduler.step(predicted_noise, t, keypoints).prev_sample
        
        # Stage 2: Generate dense trajectory conditioned on keypoints
        trajectory_shape = (batch_size * num_samples, self.horizon, self.action_dim)
        
        # Initialize trajectory using keypoint interpolation
        keypoint_flat = keypoints.flatten(start_dim=1)
        trajectory_init = self.keypoint_interpolator(keypoint_flat)
        trajectory_init = trajectory_init.view(trajectory_shape)
        
        # Add noise for diffusion process
        trajectories = trajectory_init + 0.1 * torch.randn_like(trajectory_init)
        
        # Prepare enhanced conditioning (original + keypoints)
        keypoint_conditions = torch.cat([conditions, keypoint_flat], dim=-1)
        
        # Set main scheduler
        self.scheduler.set_timesteps(self.num_diffusion_steps)
        
        # Denoise dense trajectory
        for t in self.scheduler.timesteps:
            timesteps = torch.full((trajectories.shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.refinement_network(trajectories, keypoint_conditions, timesteps)
            trajectories = self.scheduler.step(predicted_noise, t, trajectories).prev_sample
        
        return trajectories
    
    def compute_hierarchical_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for hierarchical training.
        
        Trains both keypoint and refinement networks jointly with appropriate
        loss weighting and curriculum learning strategies.
        """
        # Extract keypoints from ground truth trajectories
        trajectories = batch['trajectory'][:, :self.horizon]
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        
        # Sample keypoints uniformly from trajectory
        keypoint_indices = torch.linspace(
            0, trajectories.shape[1] - 1, self.num_keypoints, 
            device=trajectories.device, dtype=torch.long
        )
        gt_keypoints = trajectories[:, keypoint_indices]
        
        # Keypoint generation loss
        keypoint_loss = self._compute_keypoint_loss(gt_keypoints, conditions)
        
        # Dense trajectory loss (conditioned on ground truth keypoints)
        trajectory_loss = self._compute_trajectory_loss(trajectories, conditions, gt_keypoints)
        
        # Combined loss with adaptive weighting
        total_loss = keypoint_loss + 0.5 * trajectory_loss
        
        return {
            'loss': total_loss,
            'keypoint_loss': keypoint_loss,
            'trajectory_loss': trajectory_loss
        }
    
    def _compute_keypoint_loss(self, keypoints: torch.Tensor, 
                              conditions: torch.Tensor) -> torch.Tensor:
        """Compute loss for keypoint generation."""
        batch_size = keypoints.shape[0]
        
        # Sample timesteps for keypoint diffusion
        timesteps = torch.randint(
            0, self.keypoint_scheduler.num_train_timesteps, (batch_size,),
            device=keypoints.device, dtype=torch.long
        )
        
        # Add noise and predict
        noise = torch.randn_like(keypoints)
        noisy_keypoints = self.keypoint_scheduler.add_noise(keypoints, noise, timesteps)
        predicted_noise = self.keypoint_network(noisy_keypoints, conditions, timesteps)
        
        return F.mse_loss(predicted_noise, noise)
    
    def _compute_trajectory_loss(self, trajectories: torch.Tensor,
                               conditions: torch.Tensor,
                               keypoints: torch.Tensor) -> torch.Tensor:
        """Compute loss for dense trajectory generation."""
        batch_size = trajectories.shape[0]
        
        # Prepare keypoint conditioning
        keypoint_conditions = torch.cat([
            conditions, 
            keypoints.flatten(start_dim=1)
        ], dim=-1)
        
        # Sample timesteps for trajectory diffusion
        timesteps = torch.randint(
            0, self.num_diffusion_steps, (batch_size,),
            device=trajectories.device, dtype=torch.long
        )
        
        # Add noise and predict
        noise = torch.randn_like(trajectories)
        noisy_trajectories = self.scheduler.add_noise(trajectories, noise, timesteps)
        predicted_noise = self.refinement_network(noisy_trajectories, keypoint_conditions, timesteps)
        
        return F.mse_loss(predicted_noise, noise)


# Factory function for creating different diffusion policy variants
def create_diffusion_policy(config: Dict[str, Any]) -> DiffusionPolicyModel:
    """
    Factory function to create appropriate diffusion policy variant.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated diffusion policy model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'standard':
        return DiffusionPolicyModel(config)
    elif model_type == 'conditional':
        return ConditionalDiffusionPolicy(config)
    elif model_type == 'hierarchical':
        return HierarchicalDiffusionPolicy(config)
    else:
        raise ValueError(f"Unknown diffusion policy type: {model_type}")