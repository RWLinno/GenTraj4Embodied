"""
Multi-Layer Perceptron (MLP) Model for 3D End-Effector Trajectory Generation

This module implements MLP-based approaches for robotic trajectory generation,
including standard MLP, conditional MLP, and residual MLP variants.

Key Features:
- Simple feedforward architecture for trajectory generation
- Conditional generation based on start/goal poses
- Residual connections for improved training
- Support for different activation functions and normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from .base_model import FundamentalArchitectureModel


class MLPModel(FundamentalArchitectureModel):
    """
    Multi-Layer Perceptron for trajectory generation
    
    This class implements a simple MLP architecture for generating robotic trajectories.
    It takes start and end poses as input and directly outputs trajectory waypoints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Architecture parameters
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        self.activation = config.get('activation', 'relu')
        self.use_batch_norm = config.get('use_batch_norm', False)
        self.use_residual = config.get('use_residual', False)
        
        # Input/output dimensions
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        
        # Condition dimensions (start + end pose)
        self.condition_dim = self.input_dim * 2
        
        # Output trajectory dimension
        self.trajectory_dim = self.max_seq_length * self.output_dim
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        
        # Input layer
        input_size = self.condition_dim
        self.layers.append(nn.Linear(input_size, self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            if self.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(self.hidden_dim))
            
            self.layers.append(self._get_activation())
            
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))
            
            # Add residual connection for middle layers
            if self.use_residual and i > 0:
                self.layers.append(ResidualBlock(self.hidden_dim, self.hidden_dim, self.activation))
            else:
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output layer
        if self.use_batch_norm:
            self.layers.append(nn.BatchNorm1d(self.hidden_dim))
        
        self.layers.append(self._get_activation())
        
        if self.dropout > 0:
            self.layers.append(nn.Dropout(self.dropout))
        
        self.layers.append(nn.Linear(self.hidden_dim, self.trajectory_dim))
        
        # Combine all layers
        self.mlp = nn.Sequential(*self.layers)
        
    def _get_activation(self):
        """Get activation function based on configuration"""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'swish':
            return nn.SiLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the MLP
        
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
        
        # Forward pass through MLP
        trajectory_flat = self.mlp(condition)
        
        # Reshape to trajectory format
        trajectory = trajectory_flat.view(batch_size, self.max_seq_length, self.output_dim)
        
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
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute training loss
        
        Args:
            predictions: Model predictions [batch_size, seq_length, output_dim]
            targets: Target trajectory [batch_size, seq_length, output_dim]
            
        Returns:
            Loss value
        """
        # Mean squared error loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Optional: Add smoothness regularization
        if self.config.get('use_smoothness_loss', False):
            smoothness_loss = self._compute_smoothness_loss(predictions)
            total_loss = mse_loss + 0.01 * smoothness_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def _compute_smoothness_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularization loss
        """
        if trajectories.shape[1] >= 3:
            # Second-order differences (acceleration)
            accel = trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2]
            return torch.mean(torch.norm(accel, dim=-1))
        return torch.tensor(0.0, device=trajectories.device)


class ResidualBlock(nn.Module):
    """
    Residual block for MLP
    """
    
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Skip connection projection if dimensions don't match
        if input_dim != output_dim:
            self.skip_projection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block
        """
        residual = self.skip_projection(x)
        
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class ConditionalMLPModel(MLPModel):
    """
    Conditional MLP with enhanced conditioning mechanisms
    
    This variant provides more sophisticated conditioning beyond simple pose pairs,
    including task embeddings and environmental context.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Modify condition dimension for enhanced conditioning
        self.task_embed_dim = config.get('task_embed_dim', 32)
        self.context_dim = config.get('context_dim', 16)
        
        # Update condition dimension
        original_condition_dim = config.get('input_dim', 7) * 2
        enhanced_condition_dim = original_condition_dim + self.task_embed_dim + self.context_dim
        
        # Temporarily store original condition_dim and update config
        config['condition_dim'] = enhanced_condition_dim
        
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
        
        # Update the first layer to handle enhanced conditioning
        self.condition_dim = enhanced_condition_dim
        
        # Rebuild first layer with correct input dimension
        self.layers[0] = nn.Linear(self.condition_dim, self.hidden_dim)
        self.mlp = nn.Sequential(*self.layers)
    
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
        
        # Forward pass through MLP
        trajectory_flat = self.mlp(condition)
        
        # Reshape to trajectory format
        trajectory = trajectory_flat.view(batch_size, self.max_seq_length, self.output_dim)
        
        return trajectory


class PhysicsConstrainedMLPModel(MLPModel):
    """
    Physics-constrained MLP model for trajectory generation
    
    This variant incorporates physical constraints and dynamics
    to ensure generated trajectories are physically feasible.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Physics parameters
        self.max_velocity = config.get('max_velocity', 1.0)
        self.max_acceleration = config.get('max_acceleration', 2.0)
        self.enforce_constraints = config.get('enforce_constraints', True)
        
        # Constraint penalty weights
        self.velocity_penalty = config.get('velocity_penalty', 1.0)
        self.acceleration_penalty = config.get('acceleration_penalty', 1.0)
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with physics constraints
        """
        # Generate trajectory using parent forward method
        trajectory = super().forward(start_pose, end_pose, context)
        
        # Apply physics constraints if enabled
        if self.enforce_constraints and not self.training:
            trajectory = self._apply_physics_constraints(trajectory)
        
        return trajectory
    
    def _apply_physics_constraints(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Apply physics constraints to trajectory
        """
        batch_size, seq_len, dim = trajectory.shape
        
        if seq_len < 2:
            return trajectory
        
        # Compute velocities
        velocities = torch.diff(trajectory, dim=1)
        
        # Clamp velocities to maximum allowed
        velocity_norms = torch.norm(velocities, dim=-1, keepdim=True)
        velocity_scale = torch.clamp(velocity_norms / self.max_velocity, min=1.0)
        velocities = velocities / velocity_scale
        
        # Reconstruct trajectory from constrained velocities
        constrained_trajectory = torch.zeros_like(trajectory)
        constrained_trajectory[:, 0] = trajectory[:, 0]  # Keep start pose
        
        for t in range(1, seq_len):
            constrained_trajectory[:, t] = constrained_trajectory[:, t-1] + velocities[:, t-1]
        
        return constrained_trajectory
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute loss with physics constraints
        """
        # Base MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics constraint losses
        constraint_loss = 0.0
        
        if predictions.shape[1] >= 2:
            # Velocity constraint
            pred_velocities = torch.diff(predictions, dim=1)
            velocity_norms = torch.norm(pred_velocities, dim=-1)
            velocity_violation = F.relu(velocity_norms - self.max_velocity)
            velocity_loss = torch.mean(velocity_violation)
            constraint_loss += self.velocity_penalty * velocity_loss
            
            # Acceleration constraint
            if predictions.shape[1] >= 3:
                pred_accelerations = torch.diff(pred_velocities, dim=1)
                accel_norms = torch.norm(pred_accelerations, dim=-1)
                accel_violation = F.relu(accel_norms - self.max_acceleration)
                accel_loss = torch.mean(accel_violation)
                constraint_loss += self.acceleration_penalty * accel_loss
        
        total_loss = mse_loss + constraint_loss
        
        return total_loss


# Factory function for creating MLP models
def create_mlp_model(config: Dict[str, Any]) -> Union[MLPModel, ConditionalMLPModel, PhysicsConstrainedMLPModel]:
    """
    Factory function to create MLP model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated MLP model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'conditional':
        return ConditionalMLPModel(config)
    elif model_type == 'physics_constrained':
        return PhysicsConstrainedMLPModel(config)
    else:
        return MLPModel(config)