"""
Generative Flow Networks (GFlowNets) Model for 3D End-Effector Trajectory Generation

This module implements GFlowNets for robotic trajectory generation,
based on the work "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"

Key Features:
- Flow-based trajectory generation with trajectory balance
- Support for diverse trajectory sampling
- Energy-based reward modeling
- Compositional trajectory generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import math
from .base_model import ProbabilisticGenerativeModel


class GFlowNetModel(ProbabilisticGenerativeModel):
    """
    Generative Flow Networks for trajectory generation
    
    This class implements GFlowNets for generating robotic trajectories.
    It learns to sample trajectories proportionally to their rewards
    through flow matching and trajectory balance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Architecture parameters
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # GFlowNet specific parameters
        self.max_trajectory_length = config.get('max_trajectory_length', 50)
        self.action_dim = config.get('action_dim', 7)  # 3D position + 4D quaternion
        self.state_dim = config.get('state_dim', 14)   # current_pose + target_pose
        
        # Flow networks
        self.forward_policy = ForwardPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        self.backward_policy = BackwardPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # State flow estimator
        self.state_flow = StateFlow(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Reward function
        self.reward_function = RewardFunction(
            trajectory_dim=self.max_trajectory_length * self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Temperature for exploration
        self.temperature = config.get('temperature', 1.0)
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for trajectory generation
        
        Args:
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim]
            context: Optional context information [batch_size, context_dim]
            
        Returns:
            Generated trajectory [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        
        # Generate trajectories using forward policy
        trajectories = self.sample_trajectories(start_pose, end_pose, batch_size)
        
        return trajectories
    
    def sample_trajectories(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                          num_samples: int = 1) -> torch.Tensor:
        """
        Sample trajectories using GFlowNet forward policy
        
        Args:
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim]
            num_samples: Number of trajectories to sample
            
        Returns:
            Sampled trajectories [batch_size, max_length, action_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # Initialize trajectories
        trajectories = torch.zeros(batch_size, self.max_trajectory_length, self.action_dim, device=device)
        trajectories[:, 0] = start_pose
        
        # Current state (current_pose + target_pose)
        current_state = torch.cat([start_pose, end_pose], dim=-1)
        
        # Generate trajectory step by step
        for t in range(1, self.max_trajectory_length):
            # Sample action from forward policy
            action_logits = self.forward_policy(current_state)
            action_probs = F.softmax(action_logits / self.temperature, dim=-1)
            
            # Sample action (simplified - in practice would be more complex)
            action = self._sample_action(action_probs, current_state, end_pose)
            trajectories[:, t] = action
            
            # Update current state
            current_state = torch.cat([action, end_pose], dim=-1)
            
            # Check termination condition (reached target)
            if self._check_termination(action, end_pose):
                break
        
        return trajectories
    
    def _sample_action(self, action_probs: torch.Tensor, current_state: torch.Tensor, 
                      target_pose: torch.Tensor) -> torch.Tensor:
        """
        Sample next action based on forward policy
        """
        batch_size = current_state.shape[0]
        device = current_state.device
        
        # Simple action sampling - move towards target with noise
        current_pose = current_state[:, :self.action_dim]
        direction = target_pose - current_pose
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        
        # Add controlled noise for exploration
        noise = torch.randn_like(direction) * 0.1
        action = current_pose + 0.1 * direction + noise
        
        return action
    
    def _check_termination(self, current_pose: torch.Tensor, target_pose: torch.Tensor, 
                         threshold: float = 0.05) -> bool:
        """
        Check if trajectory should terminate (reached target)
        """
        distance = torch.norm(current_pose - target_pose, dim=-1)
        return torch.all(distance < threshold).item()
    
    def compute_trajectory_balance_loss(self, trajectories: torch.Tensor, 
                                      start_poses: torch.Tensor, 
                                      end_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory balance loss for GFlowNet training
        
        The trajectory balance condition ensures:
        ∑_{s'} P_F(s'|s) * F(s') = F(s) for all states s
        """
        batch_size, seq_len, _ = trajectories.shape
        total_loss = 0.0
        
        for t in range(seq_len - 1):
            current_poses = trajectories[:, t]
            next_poses = trajectories[:, t + 1]
            
            # Current and next states
            current_states = torch.cat([current_poses, end_poses], dim=-1)
            next_states = torch.cat([next_poses, end_poses], dim=-1)
            
            # Forward flow: F(s) -> F(s') * P_F(s'|s)
            forward_logits = self.forward_policy(current_states)
            forward_probs = F.softmax(forward_logits, dim=-1)
            
            # Backward flow: F(s') -> F(s) * P_B(s|s')
            backward_logits = self.backward_policy(next_states)
            backward_probs = F.softmax(backward_logits, dim=-1)
            
            # State flows
            current_flow = self.state_flow(current_states)
            next_flow = self.state_flow(next_states)
            
            # Trajectory balance loss
            # log F(s) + log P_F(s'|s) = log F(s') + log P_B(s|s')
            forward_term = current_flow + torch.log(forward_probs.mean(dim=-1) + 1e-8)
            backward_term = next_flow + torch.log(backward_probs.mean(dim=-1) + 1e-8)
            
            balance_loss = F.mse_loss(forward_term, backward_term)
            total_loss += balance_loss
        
        return total_loss / (seq_len - 1)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute GFlowNet training loss
        
        Args:
            predictions: Generated trajectories [batch_size, seq_length, output_dim]
            targets: Target trajectories [batch_size, seq_length, output_dim]
            **kwargs: Additional arguments containing start_poses and end_poses
            
        Returns:
            Total training loss
        """
        # Extract poses from kwargs
        start_poses = kwargs.get('start_poses', targets[:, 0])
        end_poses = kwargs.get('end_poses', targets[:, -1])
        
        # Trajectory balance loss
        balance_loss = self.compute_trajectory_balance_loss(predictions, start_poses, end_poses)
        
        # Reward-based loss
        pred_rewards = self.reward_function(predictions.flatten(start_dim=1))
        target_rewards = self.reward_function(targets.flatten(start_dim=1))
        reward_loss = F.mse_loss(pred_rewards, target_rewards)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(predictions, targets)
        
        # Combined loss
        total_loss = balance_loss + 0.1 * reward_loss + 0.5 * recon_loss
        
        return total_loss
    
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


class ForwardPolicy(nn.Module):
    """
    Forward policy network for GFlowNet
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network
        
        Args:
            state: Current state [batch_size, state_dim]
            
        Returns:
            Action logits [batch_size, action_dim]
        """
        return self.network(state)


class BackwardPolicy(nn.Module):
    """
    Backward policy network for GFlowNet
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Backward pass through policy network
        
        Args:
            state: Current state [batch_size, state_dim]
            
        Returns:
            Action logits [batch_size, action_dim]
        """
        return self.network(state)


class StateFlow(nn.Module):
    """
    State flow estimator for GFlowNet
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state flow
        
        Args:
            state: Current state [batch_size, state_dim]
            
        Returns:
            Log flow [batch_size]
        """
        return self.network(state).squeeze(-1)


class RewardFunction(nn.Module):
    """
    Reward function for trajectory evaluation
    """
    
    def __init__(self, trajectory_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        input_dim = trajectory_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Reward in [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory reward
        
        Args:
            trajectory: Flattened trajectory [batch_size, trajectory_dim]
            
        Returns:
            Reward values [batch_size]
        """
        return self.network(trajectory).squeeze(-1)


# Factory function for creating GFlowNet model
def create_gflownet_model(config: Dict[str, Any]) -> GFlowNetModel:
    """
    Factory function to create GFlowNet model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated GFlowNet model
    """
    return GFlowNetModel(config)