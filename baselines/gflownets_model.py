"""
Generative Flow Networks (GFlowNets) Model for 3D End-Effector Trajectory Generation

This module implements GFlowNets-based approaches for robotic trajectory generation,
leveraging flow-based generative modeling to explore diverse trajectory solutions
with principled exploration and multi-modal generation capabilities. GFlowNets
are particularly well-suited for discovering multiple valid solutions to the
same trajectory generation problem.

Key Features:
- Flow-based generative modeling with trajectory balance objective
- Principled exploration of multi-modal trajectory distributions
- Forward and backward policy networks for bidirectional flow
- Robust handling of discrete and continuous action spaces
- Support for reward-guided trajectory generation
- Enhanced diversity in generated solutions

Mathematical Foundation:
- Flow Conservation: ∑_{s'∈Parents(s)} F(s'→s) = ∑_{s'∈Children(s)} F(s→s')
- Trajectory Balance: P_F(τ)R(τ) = ∏_{t=0}^{|τ|-2} F(s_t → s_{t+1})
- Training Objective: L_TB = E_τ[(log P_F(τ)R(τ) - log ∏F(s_t→s_{t+1}))²]

Authors: Research Team
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import math
from collections import deque


class GFlowNetTrajectoryModel(nn.Module):
    """
    Generative Flow Network for trajectory generation.
    
    This model learns to generate diverse trajectories by modeling the flow
    of probability mass through trajectory space. Unlike standard generative
    models, GFlowNets explicitly balance exploration and exploitation through
    the trajectory balance objective, making them particularly effective for
    discovering multiple valid solutions.
    
    Architecture:
    - State encoder: Maps partial trajectories to state representations
    - Forward policy: Predicts next actions given current state
    - Backward policy: Predicts previous actions for flow balance
    - Flow network: Estimates flow magnitude through states
    - Reward network: Evaluates trajectory quality (optional)
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GFlowNet trajectory model.
        
        Args:
            config: Model configuration containing:
                - architecture: Network architecture parameters
                - training: Training hyperparameters including flow balance settings
                - exploration: Exploration and sampling parameters
        """
        super().__init__()
        self.config = config
        
        # Add required attributes for compatibility
        self.architecture = config.get('architecture', 'gflownet')
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        
        # Extract configuration parameters
        arch_config = config.get('architecture', {}) if isinstance(config.get('architecture'), dict) else {}
        self.hidden_dim = arch_config.get('hidden_dim', 256)
        self.num_layers = arch_config.get('num_layers', 4)
        self.num_components = arch_config.get('num_components', 8)
        self.use_attention = arch_config.get('use_attention', False)
        
        # Trajectory and condition dimensions
        self.action_dim = 7  # 3D position (3) + quaternion orientation (4)
        self.condition_dim = 14  # start_pose (7) + end_pose (7)
        self.max_seq_len = arch_config.get('max_seq_len', 50)
        
        # Flow balance parameters
        self.flow_balance_loss_type = config.get('flow_balance_loss_type', 'trajectory_balance')
        self.exploration_bonus = config.get('exploration_bonus', 0.1)
        
        # State encoder for trajectory representations
        state_input_dim = self.max_seq_len * self.action_dim + self.condition_dim
        self.state_encoder = self._build_state_encoder(state_input_dim)
        
        # Forward policy network P_F(a|s)
        self.forward_policy = self._build_policy_network()
        
        # Backward policy network P_B(a|s)
        self.backward_policy = self._build_policy_network()
        
        # Flow network Z(s) - estimates total flow through state
        self.flow_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Reward network R(τ) - evaluates trajectory quality
        if config.get('use_learned_reward', False):
            self.reward_network = self._build_reward_network()
        else:
            self.reward_network = None
        
        # Action prediction network (auxiliary task)
        self.action_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # Optional: Attention mechanism for state encoding
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Experience replay buffer for training stability
        self.replay_buffer = deque(maxlen=config.get('replay_buffer_size', 10000))
    
    def _init_weights(self, module):
        """
        Initialize model weights using Xavier/He initialization.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _build_state_encoder(self, input_dim: int) -> nn.Module:
        """
        Build state encoder network.
        
        Args:
            input_dim: Input dimension for state encoding
            
        Returns:
            State encoder network
        """
        layers = []
        current_dim = input_dim
        
        # Multi-layer encoder with residual connections
        for i in range(self.num_layers):
            layers.extend([
                nn.Linear(current_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = self.hidden_dim
        
        # Final projection
        layers.append(nn.Linear(current_dim, self.hidden_dim))
        
        return nn.Sequential(*layers)
    
    def _build_policy_network(self) -> nn.Module:
        """
        Build policy network for action prediction.
        
        Returns:
            Policy network that outputs action distribution parameters
        """
        layers = []
        input_dim = self.hidden_dim
        
        # Multi-layer policy network
        for i in range(self.num_layers):
            layers.extend([
                nn.Linear(input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = self.hidden_dim
        
        # Output layer: mixture of Gaussians parameters
        # For each action dimension: [weight, mean, log_std] for each component
        output_dim = self.action_dim * self.num_components * 3
        layers.append(nn.Linear(input_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _build_reward_network(self) -> nn.Module:
        """Build learned reward network for trajectory evaluation."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()  # Reward in [0, 1]
        )
    
    def encode_state(self, partial_trajectory: torch.Tensor,
                    conditions: torch.Tensor,
                    trajectory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode current trajectory state for flow computation.
        
        Args:
            partial_trajectory: Partial trajectory [batch_size, current_len, action_dim]
            conditions: Conditioning information [batch_size, condition_dim]
            trajectory_mask: Optional mask for variable-length trajectories
            
        Returns:
            State encoding [batch_size, hidden_dim]
        """
        batch_size = partial_trajectory.shape[0]
        current_len = partial_trajectory.shape[1]
        
        # Pad trajectory to maximum length
        if current_len < self.max_seq_len:
            padding = torch.zeros(
                batch_size, self.max_seq_len - current_len, self.action_dim,
                device=partial_trajectory.device
            )
            padded_trajectory = torch.cat([partial_trajectory, padding], dim=1)
        else:
            padded_trajectory = partial_trajectory[:, :self.max_seq_len]
        
        # Flatten trajectory for encoding
        flat_trajectory = padded_trajectory.view(batch_size, -1)
        
        # Concatenate trajectory and conditions
        state_input = torch.cat([flat_trajectory, conditions], dim=-1)
        
        # Encode state
        state_encoding = self.state_encoder(state_input)
        
        # Optional attention mechanism
        if self.use_attention:
            # Reshape for attention (treat as sequence of waypoints)
            waypoint_encodings = state_encoding.view(batch_size, 1, -1)
            attended_encoding, _ = self.attention(
                waypoint_encodings, waypoint_encodings, waypoint_encodings
            )
            state_encoding = attended_encoding.squeeze(1)
        
        return state_encoding
    
    def forward_policy_distribution(self, state_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute forward policy distribution parameters.
        
        Args:
            state_encoding: State encoding [batch_size, hidden_dim]
            
        Returns:
            Dictionary containing mixture distribution parameters
        """
        policy_output = self.forward_policy(state_encoding)
        return self._parse_policy_output(policy_output)
    
    def backward_policy_distribution(self, state_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute backward policy distribution parameters.
        
        Args:
            state_encoding: State encoding [batch_size, hidden_dim]
            
        Returns:
            Dictionary containing mixture distribution parameters
        """
        policy_output = self.backward_policy(state_encoding)
        return self._parse_policy_output(policy_output)
    
    def _parse_policy_output(self, policy_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parse policy network output into mixture distribution parameters.
        
        Args:
            policy_output: Raw policy output [batch_size, output_dim]
            
        Returns:
            Dictionary with 'weights', 'means', 'log_stds' for mixture components
        """
        batch_size = policy_output.shape[0]
        
        # Reshape to [batch_size, action_dim, num_components, 3]
        reshaped = policy_output.view(batch_size, self.action_dim, self.num_components, 3)
        
        # Extract parameters
        weights = F.softmax(reshaped[:, :, :, 0], dim=-1)  # Component weights
        means = reshaped[:, :, :, 1]  # Component means
        log_stds = torch.clamp(reshaped[:, :, :, 2], min=-10, max=2)  # Component log stds
        
        return {
            'weights': weights,
            'means': means,
            'log_stds': log_stds
        }
    
    def compute_flow(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute flow magnitude through state.
        
        Args:
            state_encoding: State encoding [batch_size, hidden_dim]
            
        Returns:
            Flow values [batch_size, 1]
        """
        return self.flow_network(state_encoding)
    
    def compute_reward(self, trajectory: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory reward.
        
        Args:
            trajectory: Complete trajectory [batch_size, seq_len, action_dim]
            conditions: Conditioning information [batch_size, condition_dim]
            
        Returns:
            Reward values [batch_size, 1]
        """
        if self.reward_network is not None:
            # Use learned reward
            state_encoding = self.encode_state(trajectory, conditions)
            return self.reward_network(state_encoding)
        else:
            # Use hand-crafted reward based on trajectory quality
            return self._compute_handcrafted_reward(trajectory, conditions)
    
    def _compute_handcrafted_reward(self, trajectory: torch.Tensor,
                                  conditions: torch.Tensor) -> torch.Tensor:
        """
        Compute hand-crafted reward based on trajectory properties.
        
        Args:
            trajectory: Complete trajectory
            conditions: Conditioning information
            
        Returns:
            Reward values
        """
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        # Goal reaching reward
        goal_poses = conditions[:, 7:14]  # Second half is goal pose
        final_poses = trajectory[:, -1, :7]  # Last waypoint
        goal_distance = torch.norm(final_poses - goal_poses, dim=-1)
        goal_reward = torch.exp(-goal_distance)
        
        # Smoothness reward
        if trajectory.shape[1] >= 3:
            velocities = trajectory[:, 1:] - trajectory[:, :-1]
            accelerations = velocities[:, 1:] - velocities[:, :-1]
            smoothness_penalty = torch.mean(torch.norm(accelerations, dim=-1), dim=-1)
            smoothness_reward = torch.exp(-smoothness_penalty)
        else:
            smoothness_reward = torch.ones(batch_size, device=device)
        
        # Combine rewards
        total_reward = 0.7 * goal_reward + 0.3 * smoothness_reward
        
        return total_reward.unsqueeze(-1)
    
    def sample_action(self, policy_dist: Dict[str, torch.Tensor],
                     temperature: float = 1.0) -> torch.Tensor:
        """
        Sample action from mixture distribution.
        
        Args:
            policy_dist: Policy distribution parameters
            temperature: Sampling temperature
            
        Returns:
            Sampled actions [batch_size, action_dim]
        """
        batch_size = policy_dist['weights'].shape[0]
        device = policy_dist['weights'].device
        
        sampled_actions = torch.zeros(batch_size, self.action_dim, device=device)
        
        for d in range(self.action_dim):
            # Sample component for this dimension
            component_weights = policy_dist['weights'][:, d] / temperature
            component_probs = F.softmax(component_weights, dim=-1)
            component_indices = torch.multinomial(component_probs, 1).squeeze(-1)
            
            # Sample from selected Gaussian component
            selected_means = policy_dist['means'][:, d, :].gather(1, component_indices.unsqueeze(-1)).squeeze(-1)
            selected_log_stds = policy_dist['log_stds'][:, d, :].gather(1, component_indices.unsqueeze(-1)).squeeze(-1)
            selected_stds = torch.exp(selected_log_stds) * temperature
            
            # Sample from Gaussian
            noise = torch.randn_like(selected_means)
            sampled_actions[:, d] = selected_means + selected_stds * noise
        
        return sampled_actions
    
    def compute_action_log_prob(self, policy_dist: Dict[str, torch.Tensor],
                               actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of actions under policy distribution.
        
        Args:
            policy_dist: Policy distribution parameters
            actions: Actions to evaluate [batch_size, action_dim]
            
        Returns:
            Log probabilities [batch_size]
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        log_probs = torch.zeros(batch_size, device=device)
        
        for d in range(self.action_dim):
            # Compute log probability for each component
            means = policy_dist['means'][:, d, :]  # [batch, num_components]
            log_stds = policy_dist['log_stds'][:, d, :]
            weights = policy_dist['weights'][:, d, :]
            
            # Gaussian log probabilities
            action_expanded = actions[:, d].unsqueeze(-1)  # [batch, 1]
            diff_squared = (action_expanded - means) ** 2
            component_log_probs = (
                -0.5 * math.log(2 * math.pi) - log_stds - 
                0.5 * diff_squared / torch.exp(2 * log_stds)
            )
            
            # Weighted mixture log probability
            weighted_probs = weights * torch.exp(component_log_probs)
            mixture_prob = torch.sum(weighted_probs, dim=-1)
            log_probs += torch.log(mixture_prob + 1e-8)
        
        return log_probs
    
    def compute_trajectory_balance_loss(self, trajectories: torch.Tensor,
                                      conditions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory balance loss for GFlowNet training.
        
        Implements the trajectory balance objective:
        L_TB = E_τ[(log P_F(τ)R(τ) - log ∏F(s_t→s_{t+1}))²]
        
        Args:
            trajectories: Complete trajectories [batch_size, seq_len, action_dim]
            conditions: Conditioning information [batch_size, condition_dim]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size, seq_len, _ = trajectories.shape
        device = trajectories.device
        
        # Compute trajectory reward
        trajectory_rewards = self.compute_reward(trajectories, conditions)
        
        total_loss = 0.0
        forward_log_probs = 0.0
        backward_log_probs = 0.0
        
        # Iterate through trajectory steps
        for t in range(seq_len - 1):
            # Current and next states
            current_traj = trajectories[:, :t+1]
            next_traj = trajectories[:, :t+2]
            current_action = trajectories[:, t+1]
            
            # Encode states
            current_state = self.encode_state(current_traj, conditions)
            next_state = self.encode_state(next_traj, conditions)
            
            # Compute flows
            current_flow = self.compute_flow(current_state)
            next_flow = self.compute_flow(next_state)
            
            # Compute policy distributions
            forward_dist = self.forward_policy_distribution(current_state)
            backward_dist = self.backward_policy_distribution(next_state)
            
            # Compute action log probabilities
            forward_log_prob = self.compute_action_log_prob(forward_dist, current_action)
            backward_log_prob = self.compute_action_log_prob(backward_dist, current_action)
            
            # Accumulate log probabilities
            forward_log_probs += forward_log_prob
            backward_log_probs += backward_log_prob
            
            # Flow balance constraint
            if self.flow_balance_loss_type == 'detailed_balance':
                # Detailed balance: F(s→s') = F(s'→s) for each transition
                flow_balance = current_flow + forward_log_prob - next_flow - backward_log_prob
                step_loss = torch.mean(flow_balance ** 2)
            else:
                # Trajectory balance will be computed after the loop
                step_loss = torch.tensor(0.0, device=device)
            
            total_loss += step_loss
        
        # Trajectory balance loss
        if self.flow_balance_loss_type == 'trajectory_balance':
            # Initial state flow
            initial_state = self.encode_state(trajectories[:, :1], conditions)
            initial_flow = self.compute_flow(initial_state)
            
            # Trajectory balance constraint
            log_pf_r = forward_log_probs + torch.log(trajectory_rewards.squeeze(-1) + 1e-8)
            trajectory_balance = initial_flow.squeeze(-1) - log_pf_r
            trajectory_loss = torch.mean(trajectory_balance ** 2)
            total_loss += trajectory_loss
        
        # Average over trajectory steps
        if self.flow_balance_loss_type == 'detailed_balance':
            total_loss = total_loss / (seq_len - 1)
        
        return {
            'flow_balance_loss': total_loss,
            'avg_reward': torch.mean(trajectory_rewards),
            'avg_forward_log_prob': torch.mean(forward_log_probs),
            'avg_backward_log_prob': torch.mean(backward_log_probs)
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute complete training loss for GFlowNet.
        
        Args:
            batch: Training batch containing trajectories and conditions
            
        Returns:
            Dictionary containing all loss components
        """
        trajectories = batch['trajectory']
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        
        # Resize trajectories if necessary
        if trajectories.shape[1] != self.max_seq_len:
            trajectories = self._resize_trajectory(trajectories, self.max_seq_len)
        
        # Main GFlowNet loss
        gfn_losses = self.compute_trajectory_balance_loss(trajectories, conditions)
        
        # Auxiliary reconstruction loss
        final_state = self.encode_state(trajectories, conditions)
        predicted_action = self.action_predictor(final_state)
        
        # Use mean action as reconstruction target
        mean_action = torch.mean(trajectories, dim=1)
        recon_loss = F.mse_loss(predicted_action, mean_action)
        
        # Optional: Learned reward supervision
        reward_loss = torch.tensor(0.0, device=trajectories.device)
        if self.reward_network is not None and 'reward' in batch:
            predicted_rewards = self.compute_reward(trajectories, conditions)
            target_rewards = batch['reward'].unsqueeze(-1)
            reward_loss = F.mse_loss(predicted_rewards, target_rewards)
        
        # Combine losses
        total_loss = (gfn_losses['flow_balance_loss'] + 
                     0.1 * recon_loss + 
                     0.1 * reward_loss)
        
        return {
            'loss': total_loss,
            'flow_balance_loss': gfn_losses['flow_balance_loss'],
            'recon_loss': recon_loss,
            'reward_loss': reward_loss,
            'avg_reward': gfn_losses['avg_reward']
        }
    
    def _resize_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resize trajectory using linear interpolation."""
        batch_size, current_length, action_dim = trajectory.shape
        
        if current_length == target_length:
            return trajectory
        
        indices = torch.linspace(0, current_length - 1, target_length,
                               device=trajectory.device, dtype=torch.float32)
        
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
                max_steps: Optional[int] = None, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate trajectories using GFlowNet sampling.
        
        Args:
            conditions: Conditioning tensor [batch_size, condition_dim]
            num_samples: Number of samples per condition
            max_steps: Maximum generation steps (defaults to max_seq_len)
            temperature: Sampling temperature for exploration
            
        Returns:
            Generated trajectories [batch_size * num_samples, seq_len, action_dim]
        """
        if max_steps is None:
            max_steps = self.max_seq_len
        
        batch_size = conditions.shape[0]
        device = conditions.device
        
        # Expand conditions for multiple samples
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
            batch_size = batch_size * num_samples
        
        # Initialize trajectories with start poses
        start_poses = conditions[:, :7]  # First 7 dimensions are start poses
        trajectories = start_poses.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        # Sequential generation using forward policy
        for step in range(1, max_steps):
            # Encode current state
            current_state = self.encode_state(trajectories, conditions)
            
            # Get forward policy distribution
            forward_dist = self.forward_policy_distribution(current_state)
            
            # Sample next action
            next_action = self.sample_action(forward_dist, temperature=temperature)
            
            # Apply constraints to ensure valid actions
            next_action = self._apply_action_constraints(next_action)
            
            # Append to trajectory
            trajectories = torch.cat([trajectories, next_action.unsqueeze(1)], dim=1)
        
        return trajectories
    
    def _apply_action_constraints(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply physical constraints to actions."""
        # Clamp positions to workspace bounds
        actions[:, :3] = torch.clamp(actions[:, :3], -2.0, 2.0)
        
        # Normalize quaternions
        quats = actions[:, 3:7]
        actions[:, 3:7] = F.normalize(quats, p=2, dim=-1)
        
        return actions
    
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
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for GFlowNet trajectory generation.
        
        Args:
            start_pose: Starting poses [batch_size, 7]
            end_pose: Goal poses [batch_size, 7] 
            context: Optional context information
            
        Returns:
            Generated trajectories [batch_size, seq_len, action_dim]
        """
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        return self.generate(conditions, num_samples=1)
    
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
            conditions = torch.cat([start_tensor, end_tensor], dim=-1)
            trajectory = self.generate(conditions, num_samples=1)
            
        self.max_seq_len = original_seq_length
        
        return trajectory.squeeze(0).numpy()


class SimpleGFlowNet(GFlowNetTrajectoryModel):
    """
    Simplified GFlowNet for rapid prototyping and baseline comparison.
    
    This variant uses simplified flow computation and policy networks
    while maintaining the core GFlowNet principles. Useful for quick
    experiments and understanding GFlowNet behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Simplified policy network
        self.simple_policy = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # Simplified flow network
        self.simple_flow = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Simplified loss computation for rapid prototyping.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss dictionary with simplified components
        """
        trajectories = batch['trajectory']
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        
        # Resize trajectories
        if trajectories.shape[1] != self.max_seq_len:
            trajectories = self._resize_trajectory(trajectories, self.max_seq_len)
        
        # Encode final state
        state_encoding = self.encode_state(trajectories, conditions)
        
        # Simple reconstruction loss
        predicted_trajectory = self.simple_policy(state_encoding)
        target_trajectory = torch.mean(trajectories, dim=1)  # Use mean as target
        
        recon_loss = F.mse_loss(predicted_trajectory, target_trajectory)
        
        # Simple flow consistency loss
        flow_value = self.simple_flow(state_encoding)
        flow_target = torch.ones_like(flow_value)  # Target flow of 1
        flow_loss = F.mse_loss(flow_value, flow_target)
        
        total_loss = recon_loss + 0.1 * flow_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'flow_loss': flow_loss
        }
    
    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, num_samples: int = 1,
                max_steps: Optional[int] = None, **kwargs) -> torch.Tensor:
        """
        Simplified generation process.
        
        Args:
            conditions: Conditioning tensor
            num_samples: Number of samples
            max_steps: Maximum steps (unused in simple version)
            **kwargs: Additional arguments
            
        Returns:
            Generated trajectories
        """
        batch_size = conditions.shape[0]
        device = conditions.device
        
        # Expand conditions for multiple samples
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
            batch_size = batch_size * num_samples
        
        # Create dummy trajectory for state encoding
        dummy_trajectory = torch.zeros(batch_size, self.max_seq_len, self.action_dim,
                                     device=device)
        
        # Set start poses
        start_poses = conditions[:, :7]
        dummy_trajectory[:, 0] = start_poses
        
        # Encode state
        state_encoding = self.encode_state(dummy_trajectory, conditions)
        
        # Generate single action and repeat
        generated_action = self.simple_policy(state_encoding)
        
        # Create trajectory by interpolating between start and generated action
        trajectories = torch.zeros(batch_size, self.max_seq_len, self.action_dim,
                                 device=device)
        
        for t in range(self.max_seq_len):
            alpha = t / (self.max_seq_len - 1)
            trajectories[:, t] = (1 - alpha) * start_poses + alpha * generated_action
        
        return trajectories


class HierarchicalGFlowNet(GFlowNetTrajectoryModel):
    """
    Hierarchical GFlowNet for multi-scale trajectory generation.
    
    This variant decomposes trajectory generation into multiple levels:
    high-level waypoint planning and low-level trajectory interpolation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Hierarchical parameters
        self.num_waypoints = config['architecture'].get('num_waypoints', 5)
        self.waypoint_dim = self.action_dim
        
        # High-level waypoint policy
        self.waypoint_policy = self._build_policy_network()
        
        # Low-level interpolation network
        self.interpolation_network = nn.Sequential(
            nn.Linear(self.hidden_dim + self.waypoint_dim * self.num_waypoints, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.max_seq_len * self.action_dim)
        )
    
    @torch.no_grad()
    def generate_hierarchical(self, conditions: torch.Tensor,
                            num_samples: int = 1) -> torch.Tensor:
        """
        Generate trajectories using hierarchical approach.
        
        Args:
            conditions: Conditioning tensor
            num_samples: Number of samples
            
        Returns:
            Generated trajectories
        """
        batch_size = conditions.shape[0]
        device = conditions.device
        
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
            batch_size = batch_size * num_samples
        
        # Generate high-level waypoints
        dummy_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        waypoint_dist = self._parse_policy_output(self.waypoint_policy(dummy_state))
        
        waypoints = []
        for _ in range(self.num_waypoints):
            waypoint = self.sample_action(waypoint_dist)
            waypoints.append(waypoint)
        
        waypoints = torch.stack(waypoints, dim=1)  # [batch, num_waypoints, action_dim]
        
        # Generate dense trajectory from waypoints
        waypoints_flat = waypoints.view(batch_size, -1)
        interpolation_input = torch.cat([dummy_state, waypoints_flat], dim=-1)
        
        dense_trajectory = self.interpolation_network(interpolation_input)
        dense_trajectory = dense_trajectory.view(batch_size, self.max_seq_len, self.action_dim)
        
        return dense_trajectory


# Factory function for creating different GFlowNet variants
def create_gflownet_model(config: Dict[str, Any]) -> GFlowNetTrajectoryModel:
    """
    Factory function to create appropriate GFlowNet variant.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated GFlowNet model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'standard':
        return GFlowNetTrajectoryModel(config)
    elif model_type == 'simple':
        return SimpleGFlowNet(config)
    elif model_type == 'hierarchical':
        return HierarchicalGFlowNet(config)
    else:
        raise ValueError(f"Unknown GFlowNet model type: {model_type}")


# Alias for backward compatibility
GFlowNetModel = GFlowNetTrajectoryModel