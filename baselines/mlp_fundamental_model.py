"""
Multi-Layer Perceptron (MLP) Trajectory Generation Model
多层感知机轨迹生成模型

使用全连接神经网络进行轨迹生成，作为基础架构组件。
MLP通过多层非线性变换学习从起点和终点到完整轨迹的映射关系。

Reference:
- Rumelhart, D. E., et al. "Learning representations by back-propagating errors." 
  nature 323.6088 (1986): 533-536.
- Hornik, K., et al. "Multilayer feedforward networks are universal approximators." 
  Neural networks 2.5 (1989): 359-366.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from ...base_model import BaseTrajectoryModel


class MLPBlock(nn.Module):
    """MLP基础块"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1, 
                 activation: str = 'relu', use_batch_norm: bool = True):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        
        if self.use_batch_norm:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class ResidualMLPBlock(nn.Module):
    """残差MLP块"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.activation(out)
        return out


class MLPTrajectoryModel(BaseTrajectoryModel):
    """
    MLP轨迹生成模型
    
    使用多层感知机学习从起点和终点到完整轨迹的映射：
    1. 输入编码：处理起点和终点信息
    2. 特征提取：通过多层MLP提取特征
    3. 轨迹解码：生成完整的轨迹序列
    4. 约束处理：确保起点和终点约束
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型参数
        self.input_dim = config.get('input_dim', 3)  # x, y, z
        self.output_dim = config.get('output_dim', 3)
        self.sequence_length = config.get('sequence_length', 50)
        self.hidden_dims = config.get('hidden_dims', [256, 512, 512, 256])
        self.dropout = config.get('dropout', 0.1)
        self.use_residual = config.get('use_residual', True)
        self.use_batch_norm = config.get('use_batch_norm', True)
        self.activation = config.get('activation', 'relu')
        
        # 输入处理
        self.condition_dim = self.input_dim * 2  # start + end pose
        self.output_size = self.sequence_length * self.output_dim
        
        # 输入编码器
        self.input_encoder = nn.Sequential(
            MLPBlock(self.condition_dim, self.hidden_dims[0], 
                    dropout=self.dropout, activation=self.activation, 
                    use_batch_norm=self.use_batch_norm),
            MLPBlock(self.hidden_dims[0], self.hidden_dims[0], 
                    dropout=self.dropout, activation=self.activation, 
                    use_batch_norm=self.use_batch_norm)
        )
        
        # 主干网络
        backbone_layers = []
        
        for i in range(len(self.hidden_dims) - 1):
            # 普通MLP层
            backbone_layers.append(
                MLPBlock(self.hidden_dims[i], self.hidden_dims[i + 1],
                        dropout=self.dropout, activation=self.activation,
                        use_batch_norm=self.use_batch_norm)
            )
            
            # 残差块
            if self.use_residual and i > 0:
                backbone_layers.append(
                    ResidualMLPBlock(self.hidden_dims[i + 1], dropout=self.dropout)
                )
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 输出解码器
        self.output_decoder = nn.Sequential(
            MLPBlock(self.hidden_dims[-1], self.hidden_dims[-1] // 2,
                    dropout=self.dropout, activation=self.activation,
                    use_batch_norm=self.use_batch_norm),
            MLPBlock(self.hidden_dims[-1] // 2, self.hidden_dims[-1] // 4,
                    dropout=self.dropout, activation=self.activation,
                    use_batch_norm=self.use_batch_norm),
            nn.Linear(self.hidden_dims[-1] // 4, self.output_size)
        )
        
        # 轨迹后处理层
        self.trajectory_refiner = nn.Sequential(
            nn.Linear(self.output_size, self.output_size),
            nn.Tanh()
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入条件 [batch_size, condition_dim] (start + end pose)
            
        Returns:
            trajectory: 输出轨迹 [batch_size, sequence_length, output_dim]
        """
        # 输入编码
        encoded = self.input_encoder(x)
        
        # 主干网络
        features = self.backbone(encoded)
        
        # 输出解码
        raw_trajectory = self.output_decoder(features)
        
        # 轨迹后处理
        refined_trajectory = self.trajectory_refiner(raw_trajectory)
        
        # 重塑为轨迹格式
        batch_size = x.shape[0]
        trajectory = refined_trajectory.view(batch_size, self.sequence_length, self.output_dim)
        
        return trajectory
    
    def generate_trajectory(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                          num_points: int = 50) -> torch.Tensor:
        """
        生成轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, 3] 或 [3]
            end_pose: 结束位姿 [batch_size, 3] 或 [3]
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 生成的轨迹 [batch_size, num_points, 3]
        """
        if start_pose.dim() == 1:
            start_pose = start_pose.unsqueeze(0)
        if end_pose.dim() == 1:
            end_pose = end_pose.unsqueeze(0)
        
        batch_size = start_pose.shape[0]
        
        # 拼接起点和终点作为条件
        conditions = torch.cat([start_pose, end_pose], dim=-1)
        
        # 前向传播生成轨迹
        self.eval()
        with torch.no_grad():
            trajectory = self.forward(conditions)
        
        # 如果需要的点数不同，进行插值调整
        if num_points != self.sequence_length:
            trajectory = self._interpolate_trajectory(trajectory, num_points)
        
        # 确保端点约束
        trajectory[:, 0] = start_pose
        trajectory[:, -1] = end_pose
        
        return trajectory
    
    def _interpolate_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """插值调整轨迹长度"""
        batch_size, current_length, dim = trajectory.shape
        
        if current_length == target_length:
            return trajectory
        
        # 使用线性插值
        old_indices = torch.linspace(0, current_length - 1, current_length, device=trajectory.device)
        new_indices = torch.linspace(0, current_length - 1, target_length, device=trajectory.device)
        
        interpolated_trajectory = []
        
        for b in range(batch_size):
            traj_b = trajectory[b]  # [current_length, dim]
            
            # 对每个维度进行插值
            interp_traj = []
            for d in range(dim):
                interp_values = torch.interp(new_indices, old_indices, traj_b[:, d])
                interp_traj.append(interp_values)
            
            interp_traj = torch.stack(interp_traj, dim=1)  # [target_length, dim]
            interpolated_trajectory.append(interp_traj)
        
        return torch.stack(interpolated_trajectory)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 预测轨迹 [batch_size, sequence_length, 3]
            targets: 目标轨迹 [batch_size, sequence_length, 3]
            
        Returns:
            loss: 总损失
        """
        # 重构损失
        reconstruction_loss = F.mse_loss(predictions, targets)
        
        # 平滑度损失
        if predictions.shape[1] > 2:
            pred_velocity = predictions[:, 1:] - predictions[:, :-1]
            target_velocity = targets[:, 1:] - targets[:, :-1]
            velocity_loss = F.mse_loss(pred_velocity, target_velocity)
            
            pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
            target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
            acceleration_loss = F.mse_loss(pred_acceleration, target_acceleration)
        else:
            velocity_loss = torch.tensor(0.0, device=predictions.device)
            acceleration_loss = torch.tensor(0.0, device=predictions.device)
        
        # 端点约束损失
        endpoint_loss = F.mse_loss(predictions[:, [0, -1]], targets[:, [0, -1]])
        
        # 路径长度正则化
        pred_lengths = torch.sum(torch.norm(predictions[:, 1:] - predictions[:, :-1], dim=-1), dim=1)
        target_lengths = torch.sum(torch.norm(targets[:, 1:] - targets[:, :-1], dim=-1), dim=1)
        length_loss = F.mse_loss(pred_lengths, target_lengths)
        
        # 总损失
        total_loss = (reconstruction_loss + 
                     0.1 * velocity_loss + 
                     0.05 * acceleration_loss + 
                     0.5 * endpoint_loss + 
                     0.02 * length_loss)
        
        return total_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        # 获取数据
        trajectories = batch.get('trajectory', torch.randn(4, self.sequence_length, 3))
        start_poses = batch.get('start_pose', trajectories[:, 0])
        end_poses = batch.get('end_pose', trajectories[:, -1])
        
        # 准备输入条件
        conditions = torch.cat([start_poses, end_poses], dim=-1)
        
        # 前向传播
        predictions = self.forward(conditions)
        
        # 计算损失
        loss = self.compute_loss(predictions, trajectories)
        
        # 计算指标
        mse = F.mse_loss(predictions, trajectories)
        endpoint_error = F.mse_loss(predictions[:, [0, -1]], trajectories[:, [0, -1]])
        
        # 计算平滑度指标
        if trajectories.shape[1] > 2:
            pred_smoothness = torch.mean(torch.norm(
                predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2], dim=-1
            ))
            target_smoothness = torch.mean(torch.norm(
                trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2], dim=-1
            ))
        else:
            pred_smoothness = torch.tensor(0.0)
            target_smoothness = torch.tensor(0.0)
        
        # 计算路径长度
        pred_length = torch.mean(torch.sum(torch.norm(predictions[:, 1:] - predictions[:, :-1], dim=-1), dim=1))
        target_length = torch.mean(torch.sum(torch.norm(trajectories[:, 1:] - trajectories[:, :-1], dim=-1), dim=1))
        
        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'endpoint_error': endpoint_error.item(),
            'pred_smoothness': pred_smoothness.item(),
            'target_smoothness': target_smoothness.item(),
            'pred_length': pred_length.item(),
            'target_length': target_length.item()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'MLP',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'sequence_length': self.sequence_length,
            'hidden_dims': self.hidden_dims,
            'use_residual': self.use_residual,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }