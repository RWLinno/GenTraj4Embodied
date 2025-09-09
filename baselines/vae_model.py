"""
Variational Autoencoder (VAE) Model for 3D End-Effector Trajectory Generation

This module implements VAE-based approaches for robotic trajectory generation,
including standard VAE, Conditional VAE (CVAE), and β-VAE variants.

Key Features:
- Probabilistic latent space representation
- Conditional generation based on start/goal poses
- Disentangled representation learning with β-VAE
- Support for trajectory interpolation in latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from .base_model import ProbabilisticGenerativeModel


class VAEModel(ProbabilisticGenerativeModel):
    """
    Variational Autoencoder for trajectory generation
    
    This class implements a VAE architecture for generating robotic trajectories.
    It learns a probabilistic latent representation of trajectory distributions
    and can generate diverse trajectories by sampling from the latent space.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Architecture parameters
        self.latent_dim = config.get('latent_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        self.beta = config.get('beta', 1.0)  # β-VAE parameter
        
        # Input/output dimensions
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        self.trajectory_dim = self.max_seq_length * self.output_dim
        
        # Condition dimensions (start + end pose)
        self.condition_dim = self.input_dim * 2
        
        # Encoder network
        encoder_layers = []
        encoder_input_dim = self.trajectory_dim + self.condition_dim
        
        for i in range(self.num_layers):
            if i == 0:
                encoder_layers.extend([
                    nn.Linear(encoder_input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
            else:
                encoder_layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # Decoder network
        decoder_layers = []
        decoder_input_dim = self.latent_dim + self.condition_dim
        
        for i in range(self.num_layers):
            if i == 0:
                decoder_layers.extend([
                    nn.Linear(decoder_input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
            elif i == self.num_layers - 1:
                decoder_layers.append(nn.Linear(self.hidden_dim, self.trajectory_dim))
            else:
                decoder_layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, trajectory: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode trajectory and condition to latent parameters
        
        Args:
            trajectory: Input trajectory [batch_size, seq_len, output_dim]
            condition: Condition tensor [batch_size, condition_dim]
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        batch_size = trajectory.shape[0]
        
        # Flatten trajectory
        trajectory_flat = trajectory.view(batch_size, -1)
        
        # Concatenate trajectory and condition
        encoder_input = torch.cat([trajectory_flat, condition], dim=-1)
        
        # Encode
        h = self.encoder(encoder_input)
        
        # Get latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector and condition to trajectory
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            condition: Condition tensor [batch_size, condition_dim]
            
        Returns:
            Decoded trajectory [batch_size, seq_len, output_dim]
        """
        batch_size = z.shape[0]
        
        # Concatenate latent vector and condition
        decoder_input = torch.cat([z, condition], dim=-1)
        
        # Decode
        trajectory_flat = self.decoder(decoder_input)
        
        # Reshape to trajectory format
        trajectory = trajectory_flat.view(batch_size, self.max_seq_length, self.output_dim)
        
        return trajectory
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for generation (inference mode)
        
        Args:
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim]
            context: Optional context information [batch_size, context_dim]
            
        Returns:
            Generated trajectory [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        
        # Create condition vector
        condition = torch.cat([start_pose, end_pose], dim=-1)
        
        # Sample from prior distribution
        z = torch.randn(batch_size, self.latent_dim, device=start_pose.device)
        
        # Decode to trajectory
        trajectory = self.decode(z, condition)
        
        return trajectory
    
    def forward_train(self, trajectory: torch.Tensor, start_pose: torch.Tensor, 
                     end_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            trajectory: Ground truth trajectory [batch_size, seq_len, output_dim]
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_trajectory, mu, logvar)
        """
        # Create condition vector
        condition = torch.cat([start_pose, end_pose], dim=-1)
        
        # Encode
        mu, logvar = self.encode(trajectory, condition)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed_trajectory = self.decode(z, condition)
        
        return reconstructed_trajectory, mu, logvar
    
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
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence)
        
        Args:
            predictions: Model predictions [batch_size, seq_length, output_dim]
            targets: Target trajectory [batch_size, seq_length, output_dim]
            **kwargs: Additional arguments (should contain mu and logvar)
            
        Returns:
            Total VAE loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(predictions, targets, reduction='sum')
        
        # KL divergence loss
        if 'mu' in kwargs and 'logvar' in kwargs:
            mu = kwargs['mu']
            logvar = kwargs['logvar']
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kld_loss = torch.tensor(0.0, device=predictions.device)
        
        # Total loss with β weighting
        total_loss = recon_loss + self.beta * kld_loss
        
        return total_loss
    
    def sample_latent(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the prior latent distribution
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Sampled latent vectors [num_samples, latent_dim]
        """
        return torch.randn(num_samples, self.latent_dim, device=device)
    
    def interpolate_trajectories(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                               num_interpolations: int = 10) -> torch.Tensor:
        """
        Generate interpolated trajectories in latent space
        
        Args:
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim]
            num_interpolations: Number of interpolation steps
            
        Returns:
            Interpolated trajectories [batch_size, num_interpolations, seq_len, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # Create condition vector
        condition = torch.cat([start_pose, end_pose], dim=-1)
        
        # Sample two latent points
        z1 = torch.randn(batch_size, self.latent_dim, device=device)
        z2 = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Interpolate in latent space
        interpolated_trajectories = []
        for i in range(num_interpolations):
            alpha = i / (num_interpolations - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Decode interpolated latent vector
            trajectory = self.decode(z_interp, condition)
            interpolated_trajectories.append(trajectory)
        
        # Stack trajectories
        interpolated_trajectories = torch.stack(interpolated_trajectories, dim=1)
        
        return interpolated_trajectories


class ConditionalVAEModel(VAEModel):
    """
    Conditional VAE with enhanced conditioning mechanisms
    
    This variant provides more sophisticated conditioning beyond simple pose pairs,
    including task embeddings and environmental context.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Modify condition dimension for enhanced conditioning
        self.task_embed_dim = config.get('task_embed_dim', 32)
        self.context_dim = config.get('context_dim', 16)
        
        # Update condition dimension
        original_condition_dim = config.get('input_dim', 7) * 2
        config['condition_dim'] = original_condition_dim + self.task_embed_dim + self.context_dim
        
        super().__init__(config)
        
        # Task embedding layer
        num_tasks = config.get('num_tasks', 10)
        self.task_embedding = nn.Embedding(num_tasks, self.task_embed_dim)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim),
            nn.ReLU(),
            nn.Linear(self.context_dim, self.context_dim)
        )
    
    def create_enhanced_condition(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                                task_id: Optional[torch.Tensor] = None,
                                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create enhanced conditioning vector
        
        Args:
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim]
            task_id: Task identifier [batch_size]
            context: Environmental context [batch_size, context_dim]
            
        Returns:
            Enhanced condition vector [batch_size, enhanced_condition_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # Basic pose conditioning
        pose_condition = torch.cat([start_pose, end_pose], dim=-1)
        
        # Task embedding
        if task_id is not None:
            task_emb = self.task_embedding(task_id)
        else:
            task_emb = torch.zeros(batch_size, self.task_embed_dim, device=device)
        
        # Context encoding
        if context is not None:
            context_emb = self.context_encoder(context)
        else:
            context_emb = torch.zeros(batch_size, self.context_dim, device=device)
        
        # Combine all conditioning information
        enhanced_condition = torch.cat([pose_condition, task_emb, context_emb], dim=-1)
        
        return enhanced_condition
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                task_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with enhanced conditioning
        """
        batch_size = start_pose.shape[0]
        
        # Create enhanced condition vector
        condition = self.create_enhanced_condition(start_pose, end_pose, task_id, context)
        
        # Sample from prior distribution
        z = torch.randn(batch_size, self.latent_dim, device=start_pose.device)
        
        # Decode to trajectory
        trajectory = self.decode(z, condition)
        
        return trajectory


# Factory function for creating VAE models
def create_vae_model(config: Dict[str, Any]) -> Union[VAEModel, ConditionalVAEModel]:
    """
    Factory function to create VAE model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated VAE model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'conditional':
        return ConditionalVAEModel(config)
    else:
        return VAEModel(config)