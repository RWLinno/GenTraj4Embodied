"""
Base classes for trajectory generation models
所有轨迹生成模型的统一基类
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import logging


class BaseTrajectoryModel(nn.Module, ABC):
    """
    所有轨迹生成模型的基类
    定义统一的接口和通用功能
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
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim] 
            context: 可选的上下文信息 [batch_size, context_dim]
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        计算损失函数
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            
        Returns:
            损失值
        """
        pass
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            包含损失值的字典
        """
        self.train()
        
        start_pose = batch['start_pose'].to(self.device)
        end_pose = batch['end_pose'].to(self.device)
        trajectory = batch['trajectory'].to(self.device)
        
        # 前向传播
        predictions = self.forward(start_pose, end_pose)
        
        # 计算损失
        loss = self.compute_loss(predictions, trajectory)
        
        return {'loss': loss.item()}
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        评估步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            包含评估指标的字典
        """
        self.eval()
        
        with torch.no_grad():
            start_pose = batch['start_pose'].to(self.device)
            end_pose = batch['end_pose'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            
            # 前向传播
            predictions = self.forward(start_pose, end_pose)
            
            # 计算损失
            loss = self.compute_loss(predictions, trajectory)
            
            # 计算额外的评估指标
            metrics = self.compute_metrics(predictions, trajectory)
            metrics['loss'] = loss.item()
            
        return metrics
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 均方误差
        mse = torch.mean((predictions - targets) ** 2).item()
        metrics['mse'] = mse
        
        # 轨迹平滑度 (加速度的方差)
        pred_acc = torch.diff(predictions, n=2, dim=1)
        target_acc = torch.diff(targets, n=2, dim=1)
        
        pred_smoothness = torch.var(pred_acc.reshape(-1, pred_acc.size(-1)), dim=0).mean().item()
        target_smoothness = torch.var(target_acc.reshape(-1, target_acc.size(-1)), dim=0).mean().item()
        
        metrics['smoothness_ratio'] = pred_smoothness / (target_smoothness + 1e-8)
        
        # 终点误差
        end_error = torch.mean((predictions[:, -1] - targets[:, -1]) ** 2).item()
        metrics['end_error'] = end_error
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
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
        移动模型到指定设备
        
        Args:
            device: 目标设备
        """
        self.device = device
        self.to(device)
        return self


class ClassicalTrajectoryModel(BaseTrajectoryModel):
    """
    经典方法的基类
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.method_type = config.get('method_type', 'interpolation')


class LinearArchitectureModel(BaseTrajectoryModel):
    """
    线性架构模型的基类 (RNN, LSTM, GRU等)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)


class TransformerVariantModel(BaseTrajectoryModel):
    """
    Transformer变体模型的基类
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dropout = config.get('dropout', 0.1)


class DiffusionVariantModel(BaseTrajectoryModel):
    """
    扩散模型变体的基类
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.beta_schedule = config.get('beta_schedule', 'linear')
        self.noise_schedule = config.get('noise_schedule', 'cosine')
        self.dropout = config.get('dropout', 0.1)


class RLBasedModel(BaseTrajectoryModel):
    """
    强化学习模型的基类
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.action_dim = config.get('action_dim', self.output_dim)
        self.state_dim = config.get('state_dim', self.input_dim * 2)  # start + end pose
        self.reward_type = config.get('reward_type', 'sparse')
        self.dropout = config.get('dropout', 0.1)
