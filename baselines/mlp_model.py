"""
Multi-Layer Perceptron (MLP) Model for 3D End-Effector Trajectory Generation

This module implements MLP-based approaches for robotic trajectory generation,
serving as efficient baseline models for comparison with more complex generative
approaches. The implementation includes standard MLP, residual MLP variants,
and conditional MLP architectures for trajectory generation tasks.

Key Features:
- Simple yet effective feedforward architecture
- Residual connections for improved gradient flow
- Conditional generation with enhanced condition processing
- Fast inference and training compared to complex generative models
- Strong baseline performance for structured trajectory tasks
- Support for various activation functions and regularization techniques

Mathematical Foundation:
- Standard MLP: f(x) = W_n σ(W_{n-1} σ(...σ(W_1 x + b_1)...) + b_{n-1}) + b_n
- Residual MLP: f(x) = x + g(x), where g is a residual function
- Conditional MLP: f(x,c) = MLP(concat(embed(c), x)) or f(x,c) = MLP(x; θ(c))

Authors: Research Team
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import math
import numpy as np


class MLPTrajectoryModel(nn.Module):
    """
    Multi-Layer Perceptron for trajectory generation.
    
    This model provides a simple yet effective baseline for trajectory generation
    by directly mapping from conditioning information (start/goal poses) to
    complete trajectories. Despite its simplicity, MLPs can achieve strong
    performance on well-structured trajectory generation tasks.
    
    Architecture:
    - Input: Conditioning information (start pose, goal pose, optional context)
    - Hidden layers: Fully connected layers with activation and dropout
    - Output: Flattened trajectory that is reshaped to sequence format
    - Optional: Residual connections, batch normalization, advanced activations
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MLP trajectory model.
        
        Args:
            config: Model configuration containing:
                - architecture: Network architecture parameters
                - training: Training hyperparameters
                - regularization: Regularization settings
        """
        super().__init__()
        self.config = config
        
        # Add required attributes for compatibility
        self.architecture = config.get('architecture', 'mlp')
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        
        # Extract configuration parameters
        arch_config = config.get('architecture', {}) if isinstance(config.get('architecture'), dict) else {}
        self.hidden_dims = arch_config.get('hidden_dims', [1024, 512, 256, 128])
        self.activation = arch_config.get('activation', 'relu')
        self.dropout = arch_config.get('dropout', 0.2)
        self.use_batch_norm = arch_config.get('use_batch_norm', False)
        self.use_layer_norm = arch_config.get('use_layer_norm', False)
        
        # Trajectory and condition dimensions
        self.action_dim = 7  # 3D position (3) + quaternion orientation (4)
        self.condition_dim = 14  # start_pose (7) + end_pose (7)
        self.max_seq_len = arch_config.get('max_seq_len', 50)
        
        # Input/output dimensions
        self.input_dim = self.condition_dim
        self.output_dim = self.max_seq_len * self.action_dim
        
        # Optional: Task embedding for multi-task learning
        if config.get('num_tasks', 0) > 0:
            self.task_embedding = nn.Embedding(
                config['num_tasks'],
                arch_config.get('task_embed_dim', 32)
            )
            self.input_dim += arch_config.get('task_embed_dim', 32)
        else:
            self.task_embedding = None
        
        # Build main network
        self.network = self._build_network()
        
        # Optional: Output post-processing layers
        if arch_config.get('use_output_refinement', False):
            self.output_refinement = self._build_output_refinement()
        else:
            self.output_refinement = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Register buffers for trajectory statistics (optional normalization)
        self.register_buffer('trajectory_mean', torch.zeros(self.action_dim))
        self.register_buffer('trajectory_std', torch.ones(self.action_dim))
    
    def _init_weights(self, module):
        """
        Initialize model weights using Xavier/He initialization.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            # Use He initialization for ReLU-like activations
            if self.activation in ['relu', 'leaky_relu']:
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            else:
                torch.nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on configuration."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif self.activation == 'swish' or self.activation == 'silu':
            return nn.SiLU()
        elif self.activation == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()  # Default fallback
    
    def _build_network(self) -> nn.Module:
        """
        Build the main MLP network.
        
        Returns:
            Sequential network with specified architecture
        """
        layers = []
        input_dim = self.input_dim
        activation = self._get_activation()
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Normalization (optional)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(activation)
            
            # Dropout (except for last hidden layer)
            if i < len(self.hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout))
            
            input_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(input_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _build_output_refinement(self) -> nn.Module:
        """
        Build output refinement layers for trajectory post-processing.
        
        Returns:
            Refinement network for trajectory smoothing and constraint enforcement
        """
        return nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, self.output_dim)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the MLP model (compatible interface).
        
        Args:
            start_pose: Starting poses [batch_size, 7]
            end_pose: Goal poses [batch_size, 7]
            context: Optional context information (unused)
            task_ids: Optional task identifiers for multi-task learning
            
        Returns:
            Predicted trajectories [batch_size, seq_len, action_dim]
        """
        # Combine start and end poses as conditions
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        return self._forward_with_conditions(conditions, task_ids=task_ids)
    
    def _forward_with_conditions(self, conditions: torch.Tensor,
                task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the MLP model.
        
        Args:
            conditions: Conditioning tensor [batch_size, condition_dim]
            task_ids: Optional task identifiers for multi-task learning
            
        Returns:
            Predicted trajectories [batch_size, seq_len, action_dim]
        """
        batch_size = conditions.shape[0]
        
        # Add task embeddings if available
        if self.task_embedding is not None and task_ids is not None:
            task_emb = self.task_embedding(task_ids)
            conditions = torch.cat([conditions, task_emb], dim=-1)
        
        # Pass through main network
        output_flat = self.network(conditions)
        
        # Optional output refinement
        if self.output_refinement is not None:
            output_flat = output_flat + self.output_refinement(output_flat)  # Residual connection
        
        # Reshape to trajectory format
        trajectory = output_flat.view(batch_size, self.max_seq_len, self.action_dim)
        
        # Apply trajectory constraints
        trajectory = self._apply_trajectory_constraints(trajectory)
        
        return trajectory
    
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
        Compute training loss for the MLP model.
        
        Implements standard supervised learning objective with optional
        regularization terms for trajectory smoothness and physical constraints.
        
        Args:
            batch: Training batch containing:
                - trajectory: Ground truth trajectories
                - start_pose: Start poses
                - end_pose: Goal poses
                - task_id: Task identifiers (optional)
                
        Returns:
            Dictionary containing loss components
        """
        trajectories = batch['trajectory']
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        task_ids = batch.get('task_id', None)
        
        # Resize trajectories to fixed length if necessary
        if trajectories.shape[1] != self.max_seq_len:
            trajectories = self._resize_trajectory(trajectories, self.max_seq_len)
        
        # Forward pass
        pred_trajectories = self._forward_with_conditions(conditions, task_ids=task_ids)
        
        # Primary reconstruction loss
        mse_loss = F.mse_loss(pred_trajectories, trajectories, reduction='mean')
        
        # Optional: Smoothness regularization
        smoothness_loss = torch.tensor(0.0, device=trajectories.device)
        if self.config.get('use_smoothness_loss', True):
            smoothness_loss = self._compute_smoothness_loss(pred_trajectories)
        
        # Optional: Goal consistency loss
        goal_loss = torch.tensor(0.0, device=trajectories.device)
        if self.config.get('use_goal_consistency_loss', False):
            goal_loss = self._compute_goal_consistency_loss(pred_trajectories, batch['end_pose'])
        
        # Optional: Start consistency loss
        start_loss = torch.tensor(0.0, device=trajectories.device)
        if self.config.get('use_start_consistency_loss', False):
            start_loss = self._compute_start_consistency_loss(pred_trajectories, batch['start_pose'])
        
        # Combine losses with weights
        total_loss = (mse_loss + 
                     0.1 * smoothness_loss + 
                     0.1 * goal_loss + 
                     0.1 * start_loss)
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'smoothness_loss': smoothness_loss,
            'goal_loss': goal_loss,
            'start_loss': start_loss
        }
    
    def _compute_smoothness_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory smoothness regularization loss.
        
        Penalizes large accelerations to encourage smooth, natural trajectories.
        
        Args:
            trajectories: Trajectory tensor [batch_size, seq_len, action_dim]
            
        Returns:
            Smoothness loss scalar
        """
        if trajectories.shape[1] < 3:
            return torch.tensor(0.0, device=trajectories.device)
        
        # Compute first-order differences (velocity)
        first_diff = trajectories[:, 1:] - trajectories[:, :-1]
        
        # Compute second-order differences (acceleration)
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
        # L2 norm of accelerations
        smoothness_loss = torch.mean(torch.sum(second_diff**2, dim=-1))
        
        return smoothness_loss
    
    def _compute_goal_consistency_loss(self, trajectories: torch.Tensor,
                                     goal_poses: torch.Tensor) -> torch.Tensor:
        """Compute loss to ensure trajectories reach goal poses."""
        final_poses = trajectories[:, -1]  # Last waypoint
        return F.mse_loss(final_poses, goal_poses)
    
    def _compute_start_consistency_loss(self, trajectories: torch.Tensor,
                                      start_poses: torch.Tensor) -> torch.Tensor:
        """Compute loss to ensure trajectories start from correct poses."""
        initial_poses = trajectories[:, 0]  # First waypoint
        return F.mse_loss(initial_poses, start_poses)
    
    def _resize_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Resize trajectory to target length using linear interpolation.
        
        Args:
            trajectory: Input trajectory [batch_size, current_length, action_dim]
            target_length: Desired trajectory length
            
        Returns:
            Resized trajectory [batch_size, target_length, action_dim]
        """
        batch_size, current_length, action_dim = trajectory.shape
        
        if current_length == target_length:
            return trajectory
        
        # Linear interpolation indices
        indices = torch.linspace(0, current_length - 1, target_length,
                               device=trajectory.device, dtype=torch.float32)
        
        # Interpolate each dimension
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
    
    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, num_samples: int = 1,
                task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate trajectories from conditioning information.
        
        Note: MLPs are deterministic, so multiple samples will be identical
        unless noise is added during generation.
        
        Args:
            conditions: Conditioning tensor [batch_size, condition_dim]
            num_samples: Number of samples per condition (limited effect for deterministic model)
            task_ids: Optional task identifiers
            
        Returns:
            Generated trajectories [batch_size * num_samples, seq_len, action_dim]
        """
        batch_size = conditions.shape[0]
        
        # Expand conditions to match number of samples
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
            if task_ids is not None:
                task_ids = task_ids.repeat_interleave(num_samples, dim=0)
        
        # Generate trajectories
        trajectories = self._forward_with_conditions(conditions, task_ids=task_ids)
        
        # Add small amount of noise for diversity (optional)
        if num_samples > 1 and self.config.get('add_generation_noise', False):
            noise_std = self.config.get('generation_noise_std', 0.01)
            noise = torch.randn_like(trajectories) * noise_std
            trajectories = trajectories + noise
            trajectories = self._apply_trajectory_constraints(trajectories)
        
        return trajectories
    
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
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            
        Returns:
            生成的轨迹 [num_points, output_dim]
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


class ResidualMLPModel(MLPTrajectoryModel):
    """
    MLP with residual connections for improved gradient flow.
    
    This variant incorporates residual connections to enable training of
    deeper networks and improve gradient flow. Particularly effective
    for complex trajectory generation tasks requiring deeper representations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize base class but rebuild network with residual connections
        super().__init__(config)
        
        # Replace standard network with residual version
        self.network = self._build_residual_network()
    
    def _build_residual_network(self) -> nn.Module:
        """
        Build network with residual connections.
        
        Returns:
            ResidualMLP network
        """
        return ResidualMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout=self.dropout,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm
        )


class ResidualMLP(nn.Module):
    """
    Multi-Layer Perceptron with residual connections.
    
    Implements skip connections between layers to improve gradient flow
    and enable training of deeper networks for complex trajectory generation.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.2, activation: str = 'relu',
                 use_batch_norm: bool = False):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        self.use_batch_norm = use_batch_norm
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(
                    in_dim=hidden_dims[i],
                    out_dim=hidden_dims[i + 1],
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm
                )
            )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dims[-1], output_dim)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Optional batch normalization for input
        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(hidden_dims[0])
        else:
            self.input_norm = nn.Identity()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'silu': nn.SiLU(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual MLP."""
        # Input projection
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.activation(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class ResidualBlock(nn.Module):
    """
    Residual block with optional batch normalization.
    
    Implements: output = activation(input + F(input))
    where F is a two-layer MLP with dropout and normalization.
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2,
                 activation: str = 'relu', use_batch_norm: bool = False):
        super().__init__()
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'silu': nn.SiLU(),
            'elu': nn.ELU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        # Batch normalization layers
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Residual connection projection
        if in_dim != out_dim:
            self.residual_projection = nn.Linear(in_dim, out_dim)
        else:
            self.residual_projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = self.residual_projection(x)
        
        # First layer
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second layer
        out = self.linear2(out)
        out = self.norm2(out)
        
        # Residual connection
        out = self.activation(out + residual)
        
        return out


class ConditionalMLPModel(MLPTrajectoryModel):
    """
    Enhanced MLP with sophisticated condition processing.
    
    This variant includes enhanced condition encoding, attention mechanisms
    for condition integration, and support for hierarchical conditioning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize base class
        super().__init__(config)
        
        # Enhanced condition encoder
        condition_hidden = config['architecture'].get('condition_hidden_dim', 256)
        condition_layers = config['architecture'].get('condition_layers', 3)
        
        # Multi-layer condition encoder
        encoder_layers = []
        input_dim = self.condition_dim
        
        for i in range(condition_layers):
            encoder_layers.extend([
                nn.Linear(input_dim, condition_hidden),
                nn.BatchNorm1d(condition_hidden) if self.use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = condition_hidden
        
        # Final condition embedding
        encoder_layers.append(nn.Linear(input_dim, condition_hidden))
        self.condition_encoder = nn.Sequential(*encoder_layers)
        
        # Attention mechanism for condition integration (optional)
        if config['architecture'].get('use_condition_attention', False):
            self.condition_attention = nn.MultiheadAttention(
                embed_dim=condition_hidden,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.condition_attention = None
        
        # Rebuild main network with encoded condition dimension
        self.input_dim = condition_hidden
        self.network = self._build_network()
    
    def forward(self, conditions: torch.Tensor,
                task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with enhanced condition processing.
        
        Args:
            conditions: Raw conditioning tensor
            task_ids: Optional task identifiers
            
        Returns:
            Predicted trajectories
        """
        batch_size = conditions.shape[0]
        
        # Add task embeddings if available
        if self.task_embedding is not None and task_ids is not None:
            task_emb = self.task_embedding(task_ids)
            conditions = torch.cat([conditions, task_emb], dim=-1)
        
        # Encode conditions
        encoded_conditions = self.condition_encoder(conditions)
        
        # Optional attention mechanism
        if self.condition_attention is not None:
            # Treat encoded conditions as a sequence of length 1
            encoded_conditions = encoded_conditions.unsqueeze(1)  # [batch, 1, hidden]
            attended_conditions, _ = self.condition_attention(
                encoded_conditions, encoded_conditions, encoded_conditions
            )
            encoded_conditions = attended_conditions.squeeze(1)  # [batch, hidden]
        
        # Pass through main network
        output_flat = self.network(encoded_conditions)
        
        # Optional output refinement
        if self.output_refinement is not None:
            output_flat = output_flat + self.output_refinement(output_flat)
        
        # Reshape to trajectory format
        trajectory = output_flat.view(batch_size, self.max_seq_len, self.action_dim)
        
        # Apply constraints
        trajectory = self._apply_trajectory_constraints(trajectory)
        
        return trajectory


class EnsembleMLPModel(nn.Module):
    """
    Ensemble of MLP models for improved robustness and diversity.
    
    Combines predictions from multiple MLP models to improve performance
    and provide uncertainty estimates through ensemble disagreement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Ensemble parameters
        self.num_models = config.get('num_ensemble_models', 3)
        self.ensemble_method = config.get('ensemble_method', 'mean')  # 'mean', 'weighted', 'attention'
        
        # Create ensemble of models
        self.models = nn.ModuleList()
        for i in range(self.num_models):
            # Add slight variations to each model
            model_config = config.copy()
            model_config['architecture'] = config['architecture'].copy()
            
            # Vary dropout rates
            model_config['architecture']['dropout'] = config['architecture'].get('dropout', 0.2) + i * 0.05
            
            # Vary hidden dimensions slightly
            base_dims = config['architecture'].get('hidden_dims', [1024, 512, 256, 128])
            varied_dims = [int(dim * (1 + (i - 1) * 0.1)) for dim in base_dims]
            model_config['architecture']['hidden_dims'] = varied_dims
            
            self.models.append(MLPTrajectoryModel(model_config))
        
        # Ensemble weighting (if using weighted ensemble)
        if self.ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        elif self.ensemble_method == 'attention':
            self.attention_weights = nn.Linear(self.models[0].output_dim, self.num_models)
    
    def forward(self, conditions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through ensemble."""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(conditions, **kwargs)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch, seq, action]
        
        # Combine predictions
        if self.ensemble_method == 'mean':
            output = torch.mean(predictions, dim=0)
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.ensemble_weights, dim=0)
            weights = weights.view(-1, 1, 1, 1)  # Broadcast weights
            output = torch.sum(predictions * weights, dim=0)
        elif self.ensemble_method == 'attention':
            # Use first model's prediction to compute attention weights
            base_pred = predictions[0].flatten(start_dim=1)  # [batch, seq*action]
            attention_scores = self.attention_weights(base_pred)  # [batch, num_models]
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, num_models]
            
            # Apply attention weights
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [batch, num_models, 1, 1]
            predictions_weighted = predictions.transpose(0, 1) * attention_weights  # [batch, num_models, seq, action]
            output = torch.sum(predictions_weighted, dim=1)  # [batch, seq, action]
        
        return output
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute ensemble loss with diversity regularization."""
        # Compute individual model losses
        individual_losses = []
        for model in self.models:
            loss_dict = model.compute_loss(batch)
            individual_losses.append(loss_dict['loss'])
        
        # Mean loss across ensemble
        mean_loss = torch.mean(torch.stack(individual_losses))
        
        # Diversity regularization (encourage model disagreement)
        if self.config.get('use_diversity_loss', False):
            conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
            predictions = []
            for model in self.models:
                pred = model(conditions)
                predictions.append(pred)
            
            predictions = torch.stack(predictions)  # [num_models, batch, seq, action]
            
            # Compute pairwise diversity
            diversity_loss = 0.0
            for i in range(self.num_models):
                for j in range(i + 1, self.num_models):
                    diversity_loss += F.mse_loss(predictions[i], predictions[j])
            
            diversity_loss = -diversity_loss / (self.num_models * (self.num_models - 1) / 2)
            total_loss = mean_loss + 0.01 * diversity_loss
        else:
            diversity_loss = torch.tensor(0.0)
            total_loss = mean_loss
        
        return {
            'loss': total_loss,
            'ensemble_loss': mean_loss,
            'diversity_loss': diversity_loss
        }


# Factory function for creating different MLP variants
def create_mlp_model(config: Dict[str, Any]) -> MLPTrajectoryModel:
    """
    Factory function to create appropriate MLP variant.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated MLP model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'standard':
        return MLPTrajectoryModel(config)
    elif model_type == 'residual':
        return ResidualMLPModel(config)
    elif model_type == 'conditional':
        return ConditionalMLPModel(config)
    elif model_type == 'ensemble':
        return EnsembleMLPModel(config)
    else:
        raise ValueError(f"Unknown MLP model type: {model_type}")


# Alias for backward compatibility
MLPModel = MLPTrajectoryModel