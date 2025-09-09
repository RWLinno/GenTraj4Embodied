"""
Base classes for trajectory generation models
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import logging


class BaseTrajectoryModel(nn.Module, ABC):
    """
    Base class for all trajectory generation models
    Defines unified interface and common functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        self.input_dim = config.get('input_dim', 7)  # 3D position + 4D quaternion
        self.output_dim = config.get('output_dim', 7)
        self.max_seq_length = config.get('max_seq_length', 50)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            start_pose: Starting pose [batch_size, input_dim]
            end_pose: Ending pose [batch_size, input_dim] 
            context: Optional context information [batch_size, context_dim]
            
        Returns:
            Generated trajectory [batch_size, seq_length, output_dim]
        """
        pass
    
    @abstractmethod
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        Trajectory generation inference interface
        
        Args:
            start_pose: Starting pose [input_dim]
            end_pose: Ending pose [input_dim]
            num_points: Number of trajectory points
            
        Returns:
            Generated trajectory [num_points, output_dim]
        """
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        Compute loss function
        
        Args:
            predictions: Model predictions [batch_size, seq_length, output_dim]
            targets: Target trajectory [batch_size, seq_length, output_dim]
            
        Returns:
            Loss value
        """
        pass
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Training step
        
        Args:
            batch: Batch data
            
        Returns:
            Dictionary containing loss values
        """
        self.train()
        
        start_pose = batch['start_pose'].to(self.device)
        end_pose = batch['end_pose'].to(self.device)
        trajectory = batch['trajectory'].to(self.device)
        
        # Forward pass
        predictions = self.forward(start_pose, end_pose)
        
        # Compute loss
        loss = self.compute_loss(predictions, trajectory)
        
        return {'loss': loss.item()}
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluation step
        
        Args:
            batch: Batch data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            start_pose = batch['start_pose'].to(self.device)
            end_pose = batch['end_pose'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            
            # Forward pass
            predictions = self.forward(start_pose, end_pose)
            
            # Compute loss
            loss = self.compute_loss(predictions, trajectory)
            
            # Compute additional evaluation metrics
            metrics = self.compute_metrics(predictions, trajectory)
            metrics['loss'] = loss.item()
            
        return metrics
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            predictions: Model predictions [batch_size, seq_length, output_dim]
            targets: Target trajectory [batch_size, seq_length, output_dim]
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Mean squared error
        mse = torch.mean((predictions - targets) ** 2).item()
        metrics['mse'] = mse
        
        # Trajectory smoothness (variance of acceleration)
        pred_acc = torch.diff(predictions, n=2, dim=1)
        target_acc = torch.diff(targets, n=2, dim=1)
        
        pred_smoothness = torch.var(pred_acc.reshape(-1, pred_acc.size(-1)), dim=0).mean().item()
        target_smoothness = torch.var(target_acc.reshape(-1, target_acc.size(-1)), dim=0).mean().item()
        
        metrics['smoothness_ratio'] = pred_smoothness / (target_smoothness + 1e-8)
        
        # End point error
        end_error = torch.mean((predictions[:, -1] - targets[:, -1]) ** 2).item()
        metrics['end_error'] = end_error
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'max_seq_length': self.max_seq_length,
            'device': self.device
        }
    
    def to_device(self, device: str):
        """
        Move model to specified device
        
        Args:
            device: Target device
        """
        self.device = device
        self.to(device)
        return self


class ClassicalTrajectoryModel(BaseTrajectoryModel):
    """
    Base class for classical methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.method_type = config.get('method_type', 'interpolation')


class FundamentalArchitectureModel(BaseTrajectoryModel):
    """
    Base class for fundamental architecture models (MLP, CNN, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)


class ProbabilisticGenerativeModel(BaseTrajectoryModel):
    """
    Base class for probabilistic generative models
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.latent_dim = config.get('latent_dim', 64)
        self.num_samples = config.get('num_samples', 1)
        self.dropout = config.get('dropout', 0.1)


class SequentialModelingModel(BaseTrajectoryModel):
    """
    Base class for sequential modeling methods (RNN, LSTM, Transformer, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 6)
        self.dropout = config.get('dropout', 0.1)


class HybridHierarchicalModel(BaseTrajectoryModel):
    """
    Base class for hybrid and hierarchical models
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.action_dim = config.get('action_dim', self.output_dim)
        self.state_dim = config.get('state_dim', self.input_dim * 2)  # start + end pose
        self.reward_type = config.get('reward_type', 'sparse')
        self.dropout = config.get('dropout', 0.1)