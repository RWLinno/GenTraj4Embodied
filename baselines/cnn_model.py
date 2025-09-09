"""
Convolutional Neural Network (CNN) Trajectory Generation Model
卷积神经网络轨迹生成模型

使用1D卷积神经网络处理轨迹序列数据，通过卷积层提取局部特征，
适用于具有空间相关性的轨迹生成任务。

Reference:
- LeCun, Y., et al. "Deep learning." nature 521.7553 (2015): 436-444.
- Krizhevsky, A., et al. "Imagenet classification with deep convolutional neural networks." 
  Communications of the ACM 60.6 (2017): 84-90.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from ...base_model import BaseTrajectoryModel


class Conv1DBlock(nn.Module):
    """1D卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class AttentionModule(nn.Module):
    """注意力模块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.size()
        
        # 计算注意力权重
        proj_query = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, length)
        proj_value = self.value(x).view(batch_size, -1, length)
        
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, length)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class CNNTrajectoryModel(BaseTrajectoryModel):
    """
    CNN轨迹生成模型
    
    使用1D卷积神经网络处理轨迹序列：
    1. 编码器：提取起点和终点特征
    2. 生成器：使用反卷积生成轨迹序列
    3. 注意力机制：增强特征表示
    4. 残差连接：改善梯度流动
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型参数
        self.input_dim = config.get('input_dim', 3)  # x, y, z
        self.output_dim = config.get('output_dim', 3)
        self.sequence_length = config.get('sequence_length', 50)
        self.hidden_channels = config.get('hidden_channels', 64)
        self.num_layers = config.get('num_layers', 4)
        self.kernel_size = config.get('kernel_size', 3)
        self.dropout = config.get('dropout', 0.1)
        self.use_attention = config.get('use_attention', True)
        self.use_residual = config.get('use_residual', True)
        
        # 条件编码器（处理起点和终点）
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, 128),  # start + end
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_channels * 4)
        )
        
        # 编码器（下采样）
        encoder_layers = []
        in_channels = self.input_dim
        
        for i in range(self.num_layers):
            out_channels = self.hidden_channels * (2 ** i)
            encoder_layers.append(Conv1DBlock(
                in_channels, out_channels, 
                kernel_size=self.kernel_size,
                stride=2 if i > 0 else 1,
                dropout=self.dropout
            ))
            
            # 添加残差块
            if self.use_residual and i > 0:
                encoder_layers.append(ResidualBlock(out_channels, dropout=self.dropout))
            
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 注意力模块
        if self.use_attention:
            self.attention = AttentionModule(in_channels)
        
        # 解码器（上采样）
        decoder_layers = []
        
        for i in range(self.num_layers):
            out_channels = self.hidden_channels * (2 ** (self.num_layers - i - 2)) if i < self.num_layers - 1 else self.output_dim
            
            if i == 0:
                # 第一层：融合条件信息
                decoder_layers.append(nn.ConvTranspose1d(
                    in_channels + self.hidden_channels * 4, 
                    self.hidden_channels * (2 ** (self.num_layers - 2)),
                    kernel_size=4, stride=2, padding=1
                ))
                in_channels = self.hidden_channels * (2 ** (self.num_layers - 2))
            else:
                decoder_layers.append(nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1
                ))
                in_channels = out_channels
            
            if i < self.num_layers - 1:
                decoder_layers.append(nn.BatchNorm1d(out_channels))
                decoder_layers.append(nn.ReLU(inplace=True))
                decoder_layers.append(nn.Dropout(self.dropout))
                
                # 添加残差块
                if self.use_residual:
                    decoder_layers.append(ResidualBlock(out_channels, dropout=self.dropout))
        
        # 最后一层使用tanh激活
        decoder_layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 输出调整层
        self.output_projection = nn.Conv1d(self.output_dim, self.output_dim, 1)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def encode_conditions(self, start_pose: torch.Tensor, end_pose: torch.Tensor) -> torch.Tensor:
        """编码起点和终点条件"""
        # 拼接起点和终点
        conditions = torch.cat([start_pose, end_pose], dim=-1)  # [batch_size, 6]
        
        # 编码条件
        encoded = self.condition_encoder(conditions)  # [batch_size, hidden_channels * 4]
        
        return encoded
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim, sequence_length]
            
        Returns:
            output: 输出张量 [batch_size, output_dim, sequence_length]
        """
        # 编码
        encoded = self.encoder(x)
        
        # 注意力
        if self.use_attention:
            encoded = self.attention(encoded)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 输出投影
        output = self.output_projection(decoded)
        
        return output
    
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
        
        # 编码条件
        conditions = self.encode_conditions(start_pose, end_pose)
        
        # 创建初始噪声序列
        noise = torch.randn(batch_size, self.input_dim, num_points, device=self.device)
        
        # 编码噪声
        encoded_noise = self.encoder(noise)
        
        # 注意力
        if self.use_attention:
            encoded_noise = self.attention(encoded_noise)
        
        # 融合条件信息
        # 将条件信息扩展到序列长度
        seq_len = encoded_noise.shape[-1]
        conditions_expanded = conditions.unsqueeze(-1).expand(-1, -1, seq_len)
        
        # 拼接特征
        fused_features = torch.cat([encoded_noise, conditions_expanded], dim=1)
        
        # 解码生成轨迹
        trajectory = self.decoder(fused_features)
        
        # 调整输出
        trajectory = self.output_projection(trajectory)
        
        # 转置为 [batch_size, num_points, 3]
        trajectory = trajectory.transpose(1, 2)
        
        # 确保起点和终点约束
        trajectory[:, 0] = start_pose
        trajectory[:, -1] = end_pose
        
        return trajectory
    
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
            pred_diff2 = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
            target_diff2 = targets[:, 2:] - 2 * targets[:, 1:-1] + targets[:, :-2]
            smoothness_loss = F.mse_loss(pred_diff2, target_diff2)
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        
        # 端点约束损失
        endpoint_loss = F.mse_loss(predictions[:, [0, -1]], targets[:, [0, -1]])
        
        # 总损失
        total_loss = reconstruction_loss + 0.1 * smoothness_loss + 0.5 * endpoint_loss
        
        return total_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        # 获取数据
        trajectories = batch.get('trajectory', torch.randn(4, self.sequence_length, 3))
        start_poses = batch.get('start_pose', trajectories[:, 0])
        end_poses = batch.get('end_pose', trajectories[:, -1])
        
        # 转换为CNN输入格式 [batch_size, channels, length]
        input_trajectories = trajectories.transpose(1, 2)
        
        # 前向传播
        predictions = self.forward(input_trajectories)
        
        # 转换回 [batch_size, length, channels]
        predictions = predictions.transpose(1, 2)
        
        # 计算损失
        loss = self.compute_loss(predictions, trajectories)
        
        # 计算指标
        mse = F.mse_loss(predictions, trajectories)
        
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
        
        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'pred_smoothness': pred_smoothness.item(),
            'target_smoothness': target_smoothness.item(),
            'endpoint_error': F.mse_loss(predictions[:, [0, -1]], trajectories[:, [0, -1]]).item()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CNN',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'sequence_length': self.sequence_length,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }