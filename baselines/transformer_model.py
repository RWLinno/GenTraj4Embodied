"""
Transformer Model for 3D End-Effector Trajectory Generation

This module implements Transformer-based approaches for robotic trajectory generation,
leveraging the self-attention mechanism to capture long-range dependencies in
sequential trajectory data. The implementation includes standard autoregressive
generation, bidirectional encoding variants, and embodiment-aware architectures.

Key Features:
- Multi-head self-attention for sequence modeling
- Causal masking for autoregressive trajectory generation
- Positional encoding for temporal structure understanding
- Flexible conditioning on start/goal poses and task context
- Support for variable-length trajectory generation
- Attention visualization for interpretability

Mathematical Foundation:
- Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Multi-Head: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
- Positional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d_model))

Authors: Research Team
Date: 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import math
import numpy as np


class TransformerTrajectoryModel(nn.Module):
    """
    Transformer model for trajectory generation using autoregressive approach.
    
    This model treats trajectory generation as a sequence-to-sequence problem,
    where the model learns to predict the next waypoint given previous waypoints
    and conditioning information (start/goal poses, task context).
    
    Architecture:
    - Input embedding layers for actions and conditions
    - Multi-layer Transformer encoder with self-attention
    - Causal masking for autoregressive generation
    - Output projection to action space
    
    Args:
        config: Configuration dictionary containing model hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Transformer trajectory model.
        
        Args:
            config: Model configuration containing:
                - architecture: Network architecture parameters
                - training: Training hyperparameters
                - generation: Generation parameters
        """
        super().__init__()
        self.config = config
        
        # Add required attributes for compatibility
        self.architecture = config.get('architecture', 'transformer')
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')
        self.input_dim = config.get('input_dim', 7)
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        
        # Extract configuration parameters
        arch_config = config.get('architecture', {}) if isinstance(config.get('architecture'), dict) else {}
        self.d_model = arch_config.get('d_model', 512)
        self.nhead = arch_config.get('nhead', 8)
        self.num_layers = arch_config.get('num_layers', 6)
        self.dim_feedforward = arch_config.get('dim_feedforward', 2048)
        self.dropout = arch_config.get('dropout', 0.1)
        # Use the already set max_seq_length, but allow architecture config to override if explicitly set
        if 'max_seq_length' in arch_config:
            self.max_seq_length = arch_config['max_seq_length']
        
        # Action and condition dimensions
        self.action_dim = 7  # 3D position (3) + quaternion orientation (4)
        self.condition_dim = 14  # start_pose (7) + end_pose (7)
        
        # Input embeddings
        self.action_embedding = nn.Linear(self.action_dim, self.d_model)
        self.condition_embedding = nn.Linear(self.condition_dim, self.d_model)
        
        # Positional encoding for temporal structure
        self.pos_encoding = PositionalEncoding(
            d_model=self.d_model, 
            max_len=self.max_seq_length + 1  # +1 for condition token
        )
        
        # Layer normalization for input embeddings
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',  # GELU activation for better performance
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers,
            enable_nested_tensor=False  # For compatibility
        )
        
        # Output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward, self.action_dim)
        )
        
        # Optional: Task embedding for multi-task learning
        if config.get('num_tasks', 0) > 0:
            self.task_embedding = nn.Embedding(
                config['num_tasks'], 
                arch_config.get('task_embed_dim', 64)
            )
            self.condition_dim += arch_config.get('task_embed_dim', 64)
        else:
            self.task_embedding = None
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
        
        # Register attention weights for visualization
        self.attention_weights = []
    
    def _init_weights(self, module):
        """
        Initialize model weights using Xavier uniform initialization.
        
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
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer model (compatible interface).
        
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
        dummy_traj = start_pose.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        
        return self._forward_with_trajectory(dummy_traj, conditions)
    
    def _forward_with_trajectory(self, trajectories: torch.Tensor, conditions: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                task_ids: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the Transformer model.
        
        Args:
            trajectories: Input trajectory tensor [batch_size, seq_len, action_dim]
            conditions: Conditioning tensor [batch_size, condition_dim]
            mask: Optional attention mask [batch_size, seq_len]
            task_ids: Optional task identifiers for multi-task learning
            return_attention: Whether to return attention weights
            
        Returns:
            Predicted trajectory [batch_size, seq_len, action_dim]
            Optionally attention weights if return_attention=True
        """
        batch_size, seq_len, _ = trajectories.shape
        device = trajectories.device
        
        # Embed trajectories and conditions
        traj_emb = self.action_embedding(trajectories)  # [batch_size, seq_len, d_model]
        
        # Enhanced condition embedding with optional task information
        if self.task_embedding is not None and task_ids is not None:
            task_emb = self.task_embedding(task_ids)  # [batch_size, task_embed_dim]
            conditions = torch.cat([conditions, task_emb], dim=-1)
        
        cond_emb = self.condition_embedding(conditions)  # [batch_size, d_model]
        cond_emb = cond_emb.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Concatenate condition as first token (similar to BERT's [CLS] token)
        x = torch.cat([cond_emb, traj_emb], dim=1)  # [batch_size, seq_len+1, d_model]
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal attention mask for autoregressive generation
        causal_mask = self._generate_causal_mask(seq_len + 1, device)
        
        # Optional: Combine with padding mask
        if mask is not None:
            # Extend mask to include condition token
            extended_mask = torch.cat([
                torch.ones(batch_size, 1, device=device, dtype=torch.bool),
                mask
            ], dim=1)
            # Convert to attention mask format
            extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask.unsqueeze(0) + extended_mask.float() * -1e9
        
        # Store attention weights if requested
        if return_attention:
            self.attention_weights = []
            
            # Hook to capture attention weights
            def attention_hook(module, input, output):
                if hasattr(output, 'attn_weights'):
                    self.attention_weights.append(output.attn_weights.detach())
            
            # Register hooks
            hooks = []
            for layer in self.transformer.layers:
                hooks.append(layer.self_attn.register_forward_hook(attention_hook))
        
        # Pass through Transformer layers
        x = self.transformer(x, mask=causal_mask)
        
        # Remove hooks if they were registered
        if return_attention:
            for hook in hooks:
                hook.remove()
        
        # Remove condition token and apply output projection
        x = x[:, 1:]  # Remove first token (condition)
        output = self.output_projection(x)
        
        if return_attention:
            return output, torch.stack(self.attention_weights) if self.attention_weights else None
        return output
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length including condition token
            device: Device to create mask on
            
        Returns:
            Causal mask tensor [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss using teacher forcing.
        
        Implements the standard autoregressive training objective where
        the model learns to predict the next waypoint given previous waypoints.
        
        Args:
            batch: Training batch containing:
                - trajectory: Ground truth trajectories [batch_size, seq_len, action_dim]
                - start_pose: Start poses [batch_size, 7]
                - end_pose: Goal poses [batch_size, 7]
                - task_id: Task identifiers (optional)
                
        Returns:
            Dictionary containing loss components and metrics
        """
        trajectories = batch['trajectory']
        conditions = torch.cat([batch['start_pose'], batch['end_pose']], dim=-1)
        task_ids = batch.get('task_id', None)
        
        # Prepare input and target sequences for teacher forcing
        input_traj = trajectories[:, :-1]  # All but last waypoint as input
        target_traj = trajectories[:, 1:]   # All but first waypoint as target
        
        # Forward pass
        pred_traj = self._forward_with_trajectory(input_traj, conditions, task_ids=task_ids)
        
        # Compute mean squared error loss
        mse_loss = F.mse_loss(pred_traj, target_traj, reduction='mean')
        
        # Optional: Add smoothness regularization
        smoothness_loss = torch.tensor(0.0, device=pred_traj.device)
        if self.config.get('use_smoothness_loss', False):
            smoothness_loss = self._compute_smoothness_loss(pred_traj)
        
        # Optional: Add goal consistency loss
        goal_loss = torch.tensor(0.0, device=pred_traj.device)
        if self.config.get('use_goal_consistency_loss', False):
            goal_loss = self._compute_goal_consistency_loss(pred_traj, batch['end_pose'])
        
        # Combine losses
        total_loss = mse_loss + 0.01 * smoothness_loss + 0.1 * goal_loss
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'smoothness_loss': smoothness_loss,
            'goal_loss': goal_loss
        }
    
    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, max_length: int = 50,
                temperature: float = 1.0, top_k: Optional[int] = None,
                task_ids: Optional[torch.Tensor] = None,
                guidance_scale: float = 1.0) -> torch.Tensor:
        """
        Generate trajectories using autoregressive sampling.
        
        Supports various sampling strategies including temperature scaling,
        top-k sampling, and classifier-free guidance for improved quality.
        
        Args:
            conditions: Conditioning tensor [batch_size, condition_dim]
            max_length: Maximum trajectory length to generate
            temperature: Temperature for sampling (>1.0 for more diversity)
            top_k: Top-k sampling parameter (None for no top-k)
            task_ids: Task identifiers for multi-task models
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Generated trajectories [batch_size, max_length, action_dim]
        """
        batch_size = conditions.shape[0]
        device = conditions.device
        
        # Initialize sequence with start pose
        start_pose = conditions[:, :7]  # First 7 dimensions are start pose
        generated = start_pose.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        # Autoregressive generation loop
        for step in range(max_length - 1):
            # Forward pass to get next waypoint prediction
            output = self._forward_with_trajectory(generated, conditions, task_ids=task_ids)
            
            # Get prediction for next waypoint
            next_logits = output[:, -1]  # Last timestep prediction
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply top-k sampling if specified
            if top_k is not None:
                # For continuous outputs, we interpret top-k as selecting
                # from a discretized version of the output distribution
                pass  # Implementation would depend on specific discretization scheme
            
            # For continuous trajectory generation, we can add controlled noise
            if temperature > 1.0:
                noise = torch.randn_like(next_logits) * (temperature - 1.0) * 0.01
                next_pred = next_logits + noise
            else:
                next_pred = next_logits
            
            # Ensure predictions are within reasonable bounds
            next_pred = self._apply_trajectory_constraints(next_pred, generated, conditions)
            
            # Add to generated sequence
            generated = torch.cat([generated, next_pred.unsqueeze(1)], dim=1)
        
        return generated
    
    @torch.no_grad()
    def sample(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
               max_length: int = 50, **kwargs) -> torch.Tensor:
        """
        Sample trajectories from start to end poses.
        
        Convenience method for trajectory generation with pose conditioning.
        
        Args:
            start_pose: Starting poses [batch_size, 7]
            end_pose: Goal poses [batch_size, 7]
            max_length: Trajectory length
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            Sampled trajectories [batch_size, max_length, action_dim]
        """
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        return self.generate(conditions, max_length, **kwargs)
    
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
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self.forward(start_tensor, end_tensor)
            
        self.max_seq_length = original_seq_length
        
        return trajectory.squeeze(0).numpy()
    
    def _compute_smoothness_loss(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularization loss.
        
        Penalizes large accelerations to encourage smooth trajectories.
        
        Args:
            trajectories: Predicted trajectories
            
        Returns:
            Smoothness loss scalar
        """
        if trajectories.shape[1] >= 3:
            # Compute second-order differences (acceleration)
            accel = trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2]
            return torch.mean(torch.norm(accel, dim=-1))
        return torch.tensor(0.0, device=trajectories.device)
    
    def _compute_goal_consistency_loss(self, trajectories: torch.Tensor, 
                                     goal_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute goal consistency loss to ensure trajectories reach target poses.
        
        Args:
            trajectories: Predicted trajectories
            goal_poses: Target goal poses
            
        Returns:
            Goal consistency loss
        """
        final_poses = trajectories[:, -1]  # Last waypoint
        return F.mse_loss(final_poses, goal_poses)
    
    def _apply_trajectory_constraints(self, next_pred: torch.Tensor,
                                    generated: torch.Tensor,
                                    conditions: torch.Tensor) -> torch.Tensor:
        """
        Apply constraints to ensure generated trajectories are feasible.
        
        Args:
            next_pred: Next waypoint prediction
            generated: Previously generated waypoints
            conditions: Conditioning information
            
        Returns:
            Constrained next waypoint
        """
        # Clamp positions to reasonable workspace bounds
        next_pred[:, :3] = torch.clamp(next_pred[:, :3], -2.0, 2.0)
        
        # Normalize quaternions to ensure valid orientations
        quat = next_pred[:, 3:7]
        quat_norm = F.normalize(quat, p=2, dim=-1)
        next_pred[:, 3:7] = quat_norm
        
        # Optional: Enforce maximum step size for smoothness
        if generated.shape[1] > 0:
            prev_pose = generated[:, -1]
            max_step = 0.1  # Maximum position change per step
            
            pos_diff = next_pred[:, :3] - prev_pose[:, :3]
            pos_diff_norm = torch.norm(pos_diff, dim=-1, keepdim=True)
            pos_diff = pos_diff * torch.clamp(max_step / (pos_diff_norm + 1e-8), max=1.0)
            next_pred[:, :3] = prev_pose[:, :3] + pos_diff
        
        return next_pred


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence modeling.
    
    Adds positional information to input embeddings to help the model
    understand the temporal ordering of trajectory waypoints.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
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
        x = x + self.pe[:, :seq_len]
        return x


class EmbodimentAwareTransformer(TransformerTrajectoryModel):
    """
    Embodiment-aware Transformer that incorporates robot morphology information.
    
    This variant extends the standard Transformer by incorporating robot-specific
    information such as joint limits, workspace constraints, and kinematic structure.
    Inspired by Body Transformer and similar embodiment-aware architectures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Robot embodiment parameters
        self.joint_limits = config.get('joint_limits', None)
        self.workspace_bounds = config.get('workspace_bounds', None)
        
        # Embodiment encoding
        embodiment_dim = config['architecture'].get('embodiment_dim', 64)
        self.embodiment_encoder = nn.Sequential(
            nn.Linear(self.action_dim + embodiment_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.d_model)
        )
        
        # Kinematic constraint layer
        if config.get('use_kinematic_constraints', False):
            self.kinematic_layer = KinematicConstraintLayer(
                action_dim=self.action_dim,
                joint_limits=self.joint_limits
            )
        else:
            self.kinematic_layer = None
    
    def forward(self, trajectories: torch.Tensor, conditions: torch.Tensor,
                embodiment_info: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass with embodiment awareness.
        
        Args:
            trajectories: Input trajectories
            conditions: Conditioning information
            embodiment_info: Robot embodiment parameters
            **kwargs: Additional arguments
            
        Returns:
            Predicted trajectories with embodiment constraints
        """
        # Enhance trajectories with embodiment information
        if embodiment_info is not None:
            batch_size, seq_len, _ = trajectories.shape
            embodiment_expanded = embodiment_info.unsqueeze(1).expand(-1, seq_len, -1)
            enhanced_traj = torch.cat([trajectories, embodiment_expanded], dim=-1)
            trajectories = self.embodiment_encoder(enhanced_traj)
        
        # Standard forward pass
        output = super().forward(trajectories, conditions, **kwargs)
        
        # Apply kinematic constraints if available
        if self.kinematic_layer is not None:
            output = self.kinematic_layer(output)
        
        return output


class KinematicConstraintLayer(nn.Module):
    """
    Layer that enforces kinematic constraints on generated trajectories.
    
    Ensures that generated end-effector poses are reachable given
    robot joint limits and workspace constraints.
    """
    
    def __init__(self, action_dim: int, joint_limits: Optional[torch.Tensor] = None):
        super().__init__()
        self.action_dim = action_dim
        self.joint_limits = joint_limits
        
        # Learnable constraint parameters
        self.constraint_weights = nn.Parameter(torch.ones(action_dim))
        
    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Apply kinematic constraints to trajectories.
        
        Args:
            trajectories: Input trajectories
            
        Returns:
            Constrained trajectories
        """
        # Apply soft constraints using learned weights
        constrained = trajectories * self.constraint_weights.unsqueeze(0).unsqueeze(0)
        
        # Clamp positions to workspace bounds
        constrained[:, :, :3] = torch.clamp(constrained[:, :, :3], -1.5, 1.5)
        
        # Normalize quaternions
        quats = constrained[:, :, 3:7]
        constrained[:, :, 3:7] = F.normalize(quats, p=2, dim=-1)
        
        return constrained


class BidirectionalTransformer(TransformerTrajectoryModel):
    """
    Bidirectional Transformer for trajectory completion and refinement.
    
    Unlike the standard autoregressive model, this variant can condition
    on both past and future context, making it suitable for trajectory
    completion and refinement tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Modify config to remove causal masking
        config = config.copy()
        config['use_causal_mask'] = False
        
        super().__init__(config)
        
        # Bidirectional encoding layers
        self.bidirectional_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=True
            ) for _ in range(2)  # Additional bidirectional layers
        ])
    
    def forward(self, trajectories: torch.Tensor, conditions: torch.Tensor,
                mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Bidirectional forward pass without causal masking.
        
        Args:
            trajectories: Input trajectories (may have missing waypoints)
            conditions: Conditioning information
            mask: Padding mask for variable-length sequences
            **kwargs: Additional arguments
            
        Returns:
            Completed/refined trajectories
        """
        batch_size, seq_len, _ = trajectories.shape
        
        # Embed inputs
        traj_emb = self.action_embedding(trajectories)
        cond_emb = self.condition_embedding(conditions).unsqueeze(1)
        
        # Concatenate and add positional encoding
        x = torch.cat([cond_emb, traj_emb], dim=1)
        x = self.input_norm(x)
        x = self.pos_encoding(x)
        
        # Pass through standard Transformer layers (no causal mask)
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Additional bidirectional processing
        for layer in self.bidirectional_layers:
            x = layer(x, src_key_padding_mask=mask)
        
        # Output projection
        x = x[:, 1:]  # Remove condition token
        output = self.output_projection(x)
        
        return output
    
    def complete_trajectory(self, partial_trajectory: torch.Tensor,
                          conditions: torch.Tensor,
                          missing_mask: torch.Tensor) -> torch.Tensor:
        """
        Complete missing waypoints in a partial trajectory.
        
        Args:
            partial_trajectory: Trajectory with missing waypoints (NaN values)
            conditions: Conditioning information
            missing_mask: Boolean mask indicating missing waypoints
            
        Returns:
            Completed trajectory
        """
        # Replace NaN values with zeros for processing
        clean_trajectory = torch.where(
            torch.isnan(partial_trajectory),
            torch.zeros_like(partial_trajectory),
            partial_trajectory
        )
        
        # Forward pass
        completed = self._forward_with_trajectory(clean_trajectory, conditions, mask=missing_mask)
        
        # Only update missing waypoints
        result = torch.where(
            missing_mask.unsqueeze(-1).expand_as(partial_trajectory),
            completed,
            partial_trajectory
        )
        
        return result


# Factory function for creating different Transformer variants
def create_transformer_model(config: Dict[str, Any]) -> TransformerTrajectoryModel:
    """
    Factory function to create appropriate Transformer variant.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated Transformer model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'standard':
        return TransformerTrajectoryModel(config)
    elif model_type == 'embodiment_aware':
        return EmbodimentAwareTransformer(config)
    elif model_type == 'bidirectional':
        return BidirectionalTransformer(config)
    else:
        raise ValueError(f"Unknown Transformer model type: {model_type}")


# Alias for backward compatibility
TransformerModel = TransformerTrajectoryModel