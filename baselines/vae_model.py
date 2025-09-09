"""
Variational Autoencoder (VAE) Model for 3D End-Effector Trajectory Generation

This module implements VAE-based approaches for robotic trajectory generation,
leveraging the probabilistic latent space to capture trajectory distributions
and enable diverse generation through sampling. The implementation includes
standard VAE, conditional VAE (CVAE), and β-VAE variants for disentangled
representation learning.

Key Features:
- Probabilistic latent space modeling for trajectory distributions
- Reparameterization trick for differentiable sampling
- Conditional generation with pose and task conditioning
- Latent space interpolation for trajectory morphing
- β-VAE formulation for disentangled representations
- Support for variable-length trajectory handling

Mathematical Foundation:
- ELBO: log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
- Encoder: q_φ(z|x) = N(z; μ_φ(x), σ_φ²(x))
- Decoder: p_θ(x|z) = N(x; μ_θ(z), σ_θ²(z))
- Reparameterization: z = μ + σ ⊙ ε, where ε ~ N(0,I)

Authors: Research Team
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import math


class VAETrajectoryModel(nn.Module):
    """
    Variational Autoencoder for trajectory generation.
    
    This model learns a probabilistic latent representation of trajectory
    distributions, enabling generation of diverse trajectories through
    sampling from the learned latent space. The model supports conditional
    generation based on start/goal poses and task specifications.
    
    Architecture:
    - Encoder: Maps trajectories to latent distribution parameters (μ, σ)
    - Decoder: Reconstructs trajectories from latent samples
    - Conditional embedding: Incorporates conditioning information
    - Reparameterization: Enables differentiable sampling
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VAE trajectory model.
        
        Args:
            config: Model configuration containing:
                - architecture: Network architecture parameters
                - training: Training hyperparameters including β parameter
                - latent: Latent space configuration
        """
        super().__init__()
        self.config = config
        
        # Add required attributes for compatibility
        self.architecture = config.get('architecture', 'vae')
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        
        # Extract configuration parameters
        arch_config = config.get('architecture', {}) if isinstance(config.get('architecture'), dict) else {}
        self.encoder_dims = arch_config.get('encoder_dims', [512, 256, 128])
        self.decoder_dims = arch_config.get('decoder_dims', [128, 256, 512])
        self.latent_dim = arch_config.get('latent_dim', 64)
        self.beta = arch_config.get('beta', 1.0)  # β-VAE parameter for KL weighting
        self.use_batch_norm = arch_config.get('use_batch_norm', False)
        
        # Trajectory and condition dimensions
        self.action_dim = 7  # 3D position (3) + quaternion orientation (4)
        self.condition_dim = 14  # start_pose (7) + end_pose (7)
        self.max_seq_len = arch_config.get('max_seq_len', 50)
        
        # Flattened input dimension for fully connected layers
        self.input_dim = self.max_seq_len * self.action_dim
        
        # Build encoder network
        self.encoder = self._build_encoder()
        
        # Latent space parameterization (mean and log-variance)
        self.fc_mu = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        
        # Build decoder network
        self.decoder = self._build_decoder()
        
        # Condition embedding network
        condition_hidden = arch_config.get('condition_hidden_dim', 128)
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.condition_dim, condition_hidden),
            nn.BatchNorm1d(condition_hidden) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(condition_hidden, condition_hidden),
            nn.BatchNorm1d(condition_hidden) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(condition_hidden, self.latent_dim)
        )
        
        # Optional: Learnable prior for conditional generation
        if arch_config.get('use_learnable_prior', False):
            self.prior_network = nn.Sequential(
                nn.Linear(self.condition_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.latent_dim * 2)  # μ and log σ²
            )
        else:
            self.prior_network = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Register buffers for running statistics (optional normalization)
        self.register_buffer('trajectory_mean', torch.zeros(self.action_dim))
        self.register_buffer('trajectory_std', torch.ones(self.action_dim))
    
    def _init_weights(self, module):
        """
        Initialize model weights using Xavier initialization.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _build_encoder(self) -> nn.Module:
        """
        Build the encoder network.
        
        Returns:
            Sequential encoder network
        """
        layers = []
        input_dim = self.input_dim
        
        for i, hidden_dim in enumerate(self.encoder_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            
            # Add dropout except for the last layer
            if i < len(self.encoder_dims) - 1:
                layers.append(nn.Dropout(0.2))
            
            input_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """
        Build the decoder network.
        
        Returns:
            Sequential decoder network
        """
        layers = []
        input_dim = self.latent_dim * 2  # latent + condition embedding
        
        for i, hidden_dim in enumerate(self.decoder_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            
            # Add dropout except for the last layer
            if i < len(self.decoder_dims) - 1:
                layers.append(nn.Dropout(0.2))
            
            input_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(input_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input trajectories to latent distribution parameters.
        
        Args:
            x: Input trajectories [batch_size, seq_len, action_dim]
            
        Returns:
            Tuple of (μ, log σ²) for the latent distribution
        """
        batch_size = x.shape[0]
        
        # Flatten trajectory for fully connected layers
        x_flat = x.reshape(batch_size, -1)
        
        # Pass through encoder
        h = self.encoder(x_flat)
        
        # Parameterize latent distribution
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clamp log variance for numerical stability
        logvar = torch.clamp(logvar, min=-20, max=10)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor,
                      deterministic: bool = False) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        
        Implements z = μ + σ ⊙ ε where ε ~ N(0,I)
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            deterministic: If True, return mean without sampling
            
        Returns:
            Sampled latent variables [batch_size, latent_dim]
        """
        if deterministic:
            return mu
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variables to trajectories with conditioning.
        
        Args:
            z: Latent variables [batch_size, latent_dim]
            conditions: Conditioning information [batch_size, condition_dim]
            
        Returns:
            Reconstructed trajectories [batch_size, seq_len, action_dim]
        """
        batch_size = z.shape[0]
        
        # Embed conditions
        cond_emb = self.condition_embedding(conditions)
        
        # Concatenate latent variables and condition embedding
        decoder_input = torch.cat([z, cond_emb], dim=-1)
        
        # Decode to flattened trajectory
        x_recon_flat = self.decoder(decoder_input)
        
        # Reshape to trajectory format
        x_recon = x_recon_flat.reshape(batch_size, self.max_seq_len, self.action_dim)
        
        # Optional: Apply trajectory constraints
        x_recon = self._apply_trajectory_constraints(x_recon)
        
        return x_recon
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the VAE (compatible interface).
        
        Args:
            start_pose: Starting poses [batch_size, 7]
            end_pose: Goal poses [batch_size, 7]
            context: Optional context information (unused)
            
        Returns:
            Predicted trajectories [batch_size, seq_len, action_dim]
        """
        # Combine start and end poses as conditions
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        
        # Create dummy trajectory sequence starting with start_pose
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # Initialize with start pose and generate sequence
        dummy_traj = start_pose.unsqueeze(1).expand(-1, self.max_seq_len, -1)
        
        # Use the VAE forward method and return only the reconstructed trajectories
        recon_trajectories, _, _ = self._forward_with_trajectory(dummy_traj, conditions)
        return recon_trajectories
    
    def _forward_with_trajectory(self, trajectories: torch.Tensor, conditions: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            trajectories: Input trajectories [batch_size, seq_len, action_dim]
            conditions: Conditioning information [batch_size, condition_dim]
            deterministic: Whether to use deterministic encoding (no sampling)
            
        Returns:
            Tuple of (reconstructed_trajectories, μ, log σ²)
        """
        # Resize trajectory to fixed length if necessary
        if trajectories.shape[1] != self.max_seq_len:
            trajectories = self._resize_trajectory(trajectories, self.max_seq_len)
        
        # Encode to latent distribution
        mu, logvar = self.encode(trajectories)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        
        # Decode to trajectory
        recon_trajectories = self.decode(z, conditions)
        
        return recon_trajectories, mu, logvar
    
    def _resize_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Resize trajectory to target length using interpolation.
        
        Args:
            trajectory: Input trajectory [batch_size, current_length, action_dim]
            target_length: Desired trajectory length
            
        Returns:
            Resized trajectory [batch_size, target_length, action_dim]
        """
        batch_size, current_length, action_dim = trajectory.shape
        
        if current_length == target_length:
            return trajectory
        
        # Use linear interpolation for resizing
        indices = torch.linspace(0, current_length - 1, target_length, 
                               device=trajectory.device, dtype=torch.float32)
        
        # Interpolate each dimension separately
        resized_trajectory = torch.zeros(batch_size, target_length, action_dim, 
                                       device=trajectory.device)
        
        for b in range(batch_size):
            for d in range(action_dim):
                # Use torch.nn.functional.interpolate for 1D interpolation
                # Reshape to [batch=1, channels=1, length]
                input_seq = trajectory[b, :, d].unsqueeze(0).unsqueeze(0)  # [1, 1, current_length]
                # Interpolate to target length
                output_seq = F.interpolate(input_seq, size=target_length, mode='linear', align_corners=True)
                # Extract the interpolated sequence
                resized_trajectory[b, :, d] = output_seq.squeeze(0).squeeze(0)
        
        return resized_trajectory
    
    def _apply_trajectory_constraints(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Apply physical constraints to generated trajectories.
        
        Args:
            trajectories: Generated trajectories
            
        Returns:
            Constrained trajectories
        """
        # Create a completely new tensor to avoid any in-place operations
        batch_size, seq_len, action_dim = trajectories.shape
        constrained_trajectories = torch.zeros_like(trajectories)
        
        # Copy positions and clamp to workspace bounds
        positions = trajectories[:, :, :3]
        constrained_trajectories[:, :, :3] = torch.clamp(positions, -2.0, 2.0)
        
        # Copy and normalize quaternions for valid orientations
        quats = trajectories[:, :, 3:7]
        quats_normalized = F.normalize(quats, p=2, dim=-1)
        constrained_trajectories[:, :, 3:7] = quats_normalized
        
        return constrained_trajectories
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute VAE training loss (ELBO).
        
        Implements the Evidence Lower Bound:
        ELBO = E_q[log p(x|z)] - β * KL(q(z|x) || p(z))
        
        Args:
            batch: Training batch containing:
                - trajectory: Ground truth trajectories
                - start_pose: Start poses
                - end_pose: Goal poses
                
        Returns:
            Dictionary containing loss components
        """
        trajectories = batch['trajectory']
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        
        # Forward pass
        recon_trajectories, mu, logvar = self._forward_with_trajectory(trajectories, conditions)
        
        # Ensure both trajectories have the same length before computing loss
        if recon_trajectories.shape[1] != trajectories.shape[1]:
            # Resize reconstructed trajectories to match input trajectories
            recon_trajectories = self._resize_trajectory(recon_trajectories, trajectories.shape[1])
        
        # Reconstruction loss (negative log-likelihood)
        recon_loss = F.mse_loss(recon_trajectories, trajectories, reduction='mean')
        
        # KL divergence loss
        if self.prior_network is not None:
            # Use learned conditional prior
            prior_params = self.prior_network(conditions)
            prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
            
            # KL divergence between q(z|x,c) and p(z|c)
            kl_loss = -0.5 * torch.sum(
                1 + logvar - prior_logvar - 
                ((mu - prior_mu).pow(2) + logvar.exp()) / prior_logvar.exp()
            ) / trajectories.shape[0]
        else:
            # Standard Gaussian prior
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / trajectories.shape[0]
        
        # Total VAE loss with β weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        # Optional: Add additional regularization terms
        reg_loss = torch.tensor(0.0, device=trajectories.device)
        if self.config.get('use_smoothness_regularization', False):
            reg_loss = self._compute_smoothness_regularization(recon_trajectories)
            total_loss += 0.01 * reg_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'reg_loss': reg_loss,
            'beta': self.beta
        }
    
    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, num_samples: int = 1,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate trajectories by sampling from the latent space.
        
        Args:
            conditions: Conditioning tensor [batch_size, condition_dim]
            num_samples: Number of samples per condition
            temperature: Sampling temperature (>1.0 for more diversity)
            
        Returns:
            Generated trajectories [batch_size * num_samples, seq_len, action_dim]
        """
        batch_size = conditions.shape[0]
        device = conditions.device
        
        # Expand conditions to match number of samples
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
        
        # Sample from latent space
        if self.prior_network is not None:
            # Sample from learned conditional prior
            prior_params = self.prior_network(conditions)
            prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
            
            std = torch.exp(0.5 * prior_logvar) * temperature
            eps = torch.randn_like(std)
            z = prior_mu + eps * std
        else:
            # Sample from standard Gaussian prior
            z = torch.randn(batch_size * num_samples, self.latent_dim, 
                          device=device) * temperature
        
        # Decode to trajectories
        generated_trajectories = self.decode(z, conditions)
        
        return generated_trajectories
    
    @torch.no_grad()
    def sample(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
               num_samples: int = 1, **kwargs) -> torch.Tensor:
        """
        Sample trajectories from start to end poses.
        
        Args:
            start_pose: Starting poses [batch_size, 7]
            end_pose: Goal poses [batch_size, 7]
            num_samples: Number of samples per pose pair
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            Sampled trajectories [batch_size * num_samples, seq_len, action_dim]
        """
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        return self.generate(conditions, num_samples, **kwargs)
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [7,]
            end_pose: 目标位姿 [7,]
            num_points: 轨迹点数量
            **kwargs: 其他参数
            
        Returns:
            生成的轨迹 [num_points, 7]
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        original_seq_length = self.max_seq_len
        self.max_seq_len = num_points
        
        with torch.no_grad():
            trajectory = self.forward(start_tensor, end_tensor)
            
        self.max_seq_len = original_seq_length
        
        return trajectory.squeeze(0).numpy()
    
    @torch.no_grad()
    def interpolate(self, traj1: torch.Tensor, traj2: torch.Tensor,
                   conditions: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two trajectories in latent space.
        
        This method enables smooth morphing between different trajectory
        behaviors by interpolating in the learned latent representation.
        
        Args:
            traj1: First trajectory [1, seq_len, action_dim]
            traj2: Second trajectory [1, seq_len, action_dim]
            conditions: Conditioning information [1, condition_dim]
            steps: Number of interpolation steps
            
        Returns:
            Interpolated trajectory sequence [steps, seq_len, action_dim]
        """
        # Encode both trajectories to latent space
        mu1, _ = self.encode(traj1)
        mu2, _ = self.encode(traj2)
        
        # Create interpolation coefficients
        alphas = torch.linspace(0, 1, steps, device=traj1.device).unsqueeze(1)
        
        # Linear interpolation in latent space
        z_interp = (1 - alphas) * mu1 + alphas * mu2
        
        # Expand conditions for all interpolation steps
        conditions_expanded = conditions.repeat(steps, 1)
        
        # Decode interpolated latent variables
        interp_trajectories = self.decode(z_interp, conditions_expanded)
        
        return interp_trajectories
    
    @torch.no_grad()
    def reconstruct(self, trajectories: torch.Tensor, 
                   conditions: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct trajectories through the VAE (encode then decode).
        
        Args:
            trajectories: Input trajectories
            conditions: Conditioning information
            
        Returns:
            Reconstructed trajectories
        """
        recon_trajectories, _, _ = self._forward_with_trajectory(trajectories, conditions, deterministic=True)
        return recon_trajectories
    
    def _compute_smoothness_regularization(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute smoothness regularization term."""
        if trajectories.shape[1] >= 3:
            # Second-order differences (acceleration)
            accel = trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2]
            return torch.mean(torch.norm(accel, dim=-1))
        return torch.tensor(0.0, device=trajectories.device)


class ConditionalVAE(VAETrajectoryModel):
    """
    Enhanced Conditional VAE with sophisticated conditioning mechanisms.
    
    This variant extends the standard VAE with more sophisticated conditioning
    capabilities, including task-specific embeddings, hierarchical conditioning,
    and learned conditional priors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Enhanced condition processing
        condition_layers = config['architecture'].get('condition_layers', 3)
        condition_hidden = config['architecture'].get('condition_hidden_dim', 256)
        
        # Multi-layer condition encoder
        condition_encoder_layers = []
        input_dim = self.condition_dim
        
        for i in range(condition_layers):
            condition_encoder_layers.extend([
                nn.Linear(input_dim, condition_hidden),
                nn.BatchNorm1d(condition_hidden) if self.use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = condition_hidden
        
        condition_encoder_layers.append(nn.Linear(input_dim, self.latent_dim))
        self.enhanced_condition_encoder = nn.Sequential(*condition_encoder_layers)
        
        # Learned conditional prior network
        self.condition_prior = nn.Sequential(
            nn.Linear(self.condition_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim * 2)  # μ and log σ²
        )
        
        # Task embedding (if multi-task learning is enabled)
        if config.get('num_tasks', 0) > 0:
            self.task_embedding = nn.Embedding(
                config['num_tasks'],
                config['architecture'].get('task_embed_dim', 32)
            )
            self.condition_dim += config['architecture'].get('task_embed_dim', 32)
        else:
            self.task_embedding = None
    
    def encode_with_condition(self, x: torch.Tensor, conditions: torch.Tensor,
                            task_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced encoding with sophisticated condition integration.
        
        Args:
            x: Input trajectories
            conditions: Conditioning information
            task_ids: Optional task identifiers
            
        Returns:
            Enhanced latent distribution parameters
        """
        # Add task embeddings if available
        if self.task_embedding is not None and task_ids is not None:
            task_emb = self.task_embedding(task_ids)
            conditions = torch.cat([conditions, task_emb], dim=-1)
        
        # Base encoding
        mu_base, logvar_base = self.encode(x)
        
        # Enhanced condition encoding
        cond_emb = self.enhanced_condition_encoder(conditions)
        
        # Integrate condition information into latent parameters
        mu = mu_base + cond_emb  # Additive conditioning
        logvar = logvar_base      # Keep variance structure
        
        return mu, logvar
    
    def compute_conditional_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced conditional VAE loss with learned prior.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss dictionary with conditional components
        """
        trajectories = batch['trajectory']
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        task_ids = batch.get('task_id', None)
        
        # Resize trajectories if necessary
        if trajectories.shape[1] != self.max_seq_len:
            trajectories = self._resize_trajectory(trajectories, self.max_seq_len)
        
        # Enhanced conditional encoding
        mu, logvar = self.encode_with_condition(trajectories, conditions, task_ids)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        recon_trajectories = self.decode(z, conditions)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_trajectories, trajectories, reduction='mean')
        
        # Conditional prior parameters
        prior_params = self.condition_prior(conditions)
        prior_mu, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
        
        # Conditional KL divergence: KL(q(z|x,c) || p(z|c))
        kl_loss = -0.5 * torch.sum(
            1 + logvar - prior_logvar - 
            ((mu - prior_mu).pow(2) + logvar.exp()) / prior_logvar.exp()
        ) / trajectories.shape[0]
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'prior_mu_norm': torch.norm(prior_mu, dim=-1).mean(),
            'posterior_mu_norm': torch.norm(mu, dim=-1).mean()
        }


class HierarchicalVAE(VAETrajectoryModel):
    """
    Hierarchical VAE for multi-scale trajectory generation.
    
    This model learns hierarchical representations by modeling trajectories
    at multiple temporal scales, enabling better capture of both fine-grained
    motion details and high-level trajectory structure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Hierarchical parameters
        self.num_levels = config['architecture'].get('num_levels', 2)
        self.level_dims = config['architecture'].get('level_latent_dims', [32, 32])
        
        assert len(self.level_dims) == self.num_levels, "Level dimensions must match number of levels"
        
        # Multi-scale encoders
        self.level_encoders = nn.ModuleList()
        self.level_mu_layers = nn.ModuleList()
        self.level_logvar_layers = nn.ModuleList()
        
        for i, level_dim in enumerate(self.level_dims):
            # Encoder for this level
            encoder = self._build_level_encoder(i)
            self.level_encoders.append(encoder)
            
            # Latent parameterization for this level
            self.level_mu_layers.append(nn.Linear(self.encoder_dims[-1], level_dim))
            self.level_logvar_layers.append(nn.Linear(self.encoder_dims[-1], level_dim))
        
        # Hierarchical decoder
        total_latent_dim = sum(self.level_dims) + self.latent_dim
        self.hierarchical_decoder = self._build_hierarchical_decoder(total_latent_dim)
    
    def _build_level_encoder(self, level: int) -> nn.Module:
        """Build encoder for specific hierarchical level."""
        # Different temporal pooling for different levels
        if level == 0:  # Fine-grained level
            return self._build_encoder()
        else:  # Coarser levels with temporal pooling
            return nn.Sequential(
                nn.AdaptiveAvgPool1d(self.max_seq_len // (2 ** level)),
                nn.Flatten(),
                *[layer for layer in self._build_encoder()]
            )
    
    def _build_hierarchical_decoder(self, input_dim: int) -> nn.Module:
        """Build decoder that takes hierarchical latent representation."""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in self.decoder_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if self.use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, self.input_dim))
        return nn.Sequential(*layers)
    
    def encode_hierarchical(self, x: torch.Tensor) -> Tuple[list, list]:
        """
        Encode trajectory at multiple hierarchical levels.
        
        Args:
            x: Input trajectory
            
        Returns:
            Lists of (μ, log σ²) for each hierarchical level
        """
        mu_list, logvar_list = [], []
        
        for i, (encoder, mu_layer, logvar_layer) in enumerate(
            zip(self.level_encoders, self.level_mu_layers, self.level_logvar_layers)
        ):
            # Encode at this level
            h = encoder(x.reshape(x.shape[0], -1))
            
            # Parameterize latent distribution
            mu = mu_layer(h)
            logvar = logvar_layer(h)
            
            mu_list.append(mu)
            logvar_list.append(logvar)
        
        return mu_list, logvar_list
    
    def forward_hierarchical(self, trajectories: torch.Tensor, 
                           conditions: torch.Tensor) -> Tuple[torch.Tensor, list, list]:
        """
        Forward pass through hierarchical VAE.
        
        Args:
            trajectories: Input trajectories
            conditions: Conditioning information
            
        Returns:
            Tuple of (reconstructed_trajectories, μ_list, logvar_list)
        """
        # Resize if necessary
        if trajectories.shape[1] != self.max_seq_len:
            trajectories = self._resize_trajectory(trajectories, self.max_seq_len)
        
        # Encode at all levels
        mu_list, logvar_list = self.encode_hierarchical(trajectories)
        
        # Sample from all levels
        z_list = []
        for mu, logvar in zip(mu_list, logvar_list):
            z = self.reparameterize(mu, logvar)
            z_list.append(z)
        
        # Concatenate all latent variables
        z_combined = torch.cat(z_list, dim=-1)
        
        # Add condition embedding
        cond_emb = self.condition_embedding(conditions)
        decoder_input = torch.cat([z_combined, cond_emb], dim=-1)
        
        # Decode
        recon_flat = self.hierarchical_decoder(decoder_input)
        recon_trajectories = recon_flat.reshape(trajectories.shape[0], self.max_seq_len, self.action_dim)
        
        return recon_trajectories, mu_list, logvar_list


# Factory function for creating different VAE variants
def create_vae_model(config: Dict[str, Any]) -> VAETrajectoryModel:
    """
    Factory function to create appropriate VAE variant.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated VAE model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'standard':
        return VAETrajectoryModel(config)
    elif model_type == 'conditional':
        return ConditionalVAE(config)
    elif model_type == 'hierarchical':
        return HierarchicalVAE(config)
    else:
        raise ValueError(f"Unknown VAE model type: {model_type}")


# Alias for backward compatibility
VAEModel = VAETrajectoryModel