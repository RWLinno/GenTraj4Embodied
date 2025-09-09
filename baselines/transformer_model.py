"""
Transformer Model for 3D End-Effector Trajectory Generation

This module implements Transformer-based approaches for robotic trajectory generation,
including standard Transformer, Decision Transformer, and conditional variants.

Key Features:
- Multi-head self-attention for capturing long-range dependencies
- Positional encoding for temporal structure
- Flexible conditioning on start/goal poses
- Support for autoregressive and non-autoregressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import math
from .base_model import SequentialModelingModel


class TransformerModel(SequentialModelingModel):
    """
    Transformer model for trajectory generation
    
    This class implements a Transformer architecture for generating robotic trajectories.
    It uses multi-head self-attention to capture long-range dependencies in trajectory
    sequences and can be conditioned on start and goal poses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Architecture parameters
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 1024)
        self.dropout = config.get('dropout', 0.1)
        
        # Input/output dimensions
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Condition embedding (for start/end poses)
        self.condition_dim = self.input_dim * 2  # start + end pose
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.condition_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer
        
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
        condition_emb = self.condition_embedding(condition)  # [batch_size, d_model]
        
        # Create initial trajectory sequence (linear interpolation as initialization)
        trajectory_seq = self._create_initial_sequence(start_pose, end_pose, self.max_seq_length)
        
        # Input embedding
        embedded = self.input_embedding(trajectory_seq)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Add condition embedding to each position
        condition_emb = condition_emb.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        embedded = embedded + condition_emb
        
        # Layer normalization
        embedded = self.layer_norm(embedded)
        
        # Transformer encoding
        transformer_output = self.transformer(embedded)
        
        # Output projection
        output = self.output_projection(transformer_output)
        
        return output
    
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
    
    def _create_initial_sequence(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                               seq_length: int) -> torch.Tensor:
        """
        Create initial trajectory sequence using linear interpolation
        """
        batch_size = start_pose.shape[0]
        
        # Create interpolation weights
        weights = torch.linspace(0, 1, seq_length, device=start_pose.device)
        weights = weights.view(1, seq_length, 1)
        
        # Linear interpolation
        start_expanded = start_pose.unsqueeze(1).expand(-1, seq_length, -1)
        end_expanded = end_pose.unsqueeze(1).expand(-1, seq_length, -1)
        
        trajectory = start_expanded * (1 - weights) + end_expanded * weights
        
        return trajectory
    
    def _compute_smoothness_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularization loss
        """
        if trajectories.shape[1] >= 3:
            # Second-order differences (acceleration)
            accel = trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2]
            return torch.mean(torch.norm(accel, dim=-1))
        return torch.tensor(0.0, device=trajectories.device)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class DecisionTransformerModel(SequentialModelingModel):
    """
    Decision Transformer for trajectory generation
    
    Based on "Decision Transformer: Reinforcement Learning via Sequence Modeling"
    This variant treats trajectory generation as a sequence modeling problem.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Architecture parameters
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dropout = config.get('dropout', 0.1)
        
        # Token embeddings
        self.state_embedding = nn.Linear(self.input_dim, self.d_model)
        self.action_embedding = nn.Linear(self.output_dim, self.d_model)
        self.return_embedding = nn.Linear(1, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length * 3)
        
        # Transformer
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_model * 4,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=self.num_layers
        )
        
        # Output heads
        self.action_head = nn.Linear(self.d_model, self.output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Decision Transformer
        """
        batch_size = start_pose.shape[0]
        
        # Create trajectory sequence using autoregressive generation
        trajectory = self._generate_autoregressive(start_pose, end_pose, self.max_seq_length)
        
        return trajectory
    
    def _generate_autoregressive(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                               seq_length: int) -> torch.Tensor:
        """
        Generate trajectory autoregressively
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # Initialize trajectory with start pose
        trajectory = torch.zeros(batch_size, seq_length, self.output_dim, device=device)
        trajectory[:, 0] = start_pose
        
        # Generate sequence autoregressively
        for t in range(1, seq_length):
            # Create input sequence up to current timestep
            current_seq = trajectory[:, :t]
            
            # Embed current sequence
            embedded = self.action_embedding(current_seq)
            embedded = self.pos_encoding(embedded)
            embedded = self.layer_norm(embedded)
            
            # Create memory (conditioning information)
            memory = self._create_memory(start_pose, end_pose, t)
            
            # Transformer forward pass
            output = self.transformer(embedded, memory)
            
            # Predict next action
            next_action = self.action_head(output[:, -1])
            trajectory[:, t] = next_action
        
        return trajectory
    
    def _create_memory(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                      timestep: int) -> torch.Tensor:
        """
        Create memory tensor for conditioning
        """
        batch_size = start_pose.shape[0]
        
        # Simple conditioning: concatenate start and end poses
        condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_emb = nn.Linear(condition.shape[-1], self.d_model).to(condition.device)(condition)
        
        # Expand to sequence length
        memory = condition_emb.unsqueeze(1).expand(-1, timestep, -1)
        
        return memory
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        Generate trajectory using Decision Transformer
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
        Compute training loss for Decision Transformer
        """
        return F.mse_loss(predictions, targets)


# Factory function for creating transformer models
def create_transformer_model(config: Dict[str, Any]) -> Union[TransformerModel, DecisionTransformerModel]:
    """
    Factory function to create transformer model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated transformer model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'decision_transformer':
        return DecisionTransformerModel(config)
    else:
        return TransformerModel(config)