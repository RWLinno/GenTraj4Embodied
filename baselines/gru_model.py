"""
GRU Model for Trajectory Generation
GRU轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from .base_model import LinearArchitectureModel


class GRUTrajectoryModel(LinearArchitectureModel):
    """
    GRU轨迹生成模型
    使用GRU网络进行序列到序列的轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bidirectional = config.get('bidirectional', False)
        self.use_residual = config.get('use_residual', True)
        self.use_layer_norm = config.get('use_layer_norm', True)
        
        # 输入编码器
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # GRU层
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # 层归一化
        gru_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(gru_output_dim)
        
        # 残差连接投影
        if self.use_residual:
            self.residual_projection = nn.Linear(self.hidden_dim, gru_output_dim)
        
        # 输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(gru_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        ])
        
        # 时间嵌入
        self.time_embedding = nn.Embedding(self.max_seq_length, self.hidden_dim)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码输入
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        encoded_input = self.input_encoder(combined_pose)  # [batch_size, hidden_dim]
        
        # 创建序列输入
        input_sequence = encoded_input.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        
        # 添加时间嵌入
        time_indices = torch.arange(self.max_seq_length, device=device)
        time_emb = self.time_embedding(time_indices).unsqueeze(0).expand(batch_size, -1, -1)
        input_sequence = input_sequence + time_emb
        
        # GRU前向传播
        gru_output, hidden = self.gru(input_sequence)
        
        # 层归一化
        if self.use_layer_norm:
            gru_output = self.layer_norm(gru_output)
        
        # 残差连接
        if self.use_residual:
            residual = self.residual_projection(input_sequence)
            gru_output = gru_output + residual
        
        # 通过输出层
        output = gru_output
        for layer in self.output_layers:
            output = layer(output)
        
        # 强制边界条件
        output = self._enforce_boundary_conditions(output, start_pose, end_pose)
        
        return output
    
    def _enforce_boundary_conditions(self, trajectory: torch.Tensor,
                                   start_pose: torch.Tensor, 
                                   end_pose: torch.Tensor) -> torch.Tensor:
        """
        强制执行边界条件
        """
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        return trajectory
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
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
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        计算损失函数
        """
        # 基础重建损失
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 速度一致性损失
        velocity_weight = self.config.get('velocity_weight', 0.1)
        if velocity_weight > 0:
            pred_vel = torch.diff(predictions, dim=1)
            target_vel = torch.diff(targets, dim=1)
            velocity_loss = nn.MSELoss()(pred_vel, target_vel)
            
            total_loss = mse_loss + velocity_weight * velocity_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'bidirectional': self.bidirectional,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm,
            'model_category': 'Linear Architecture'
        })
        return info


class MultiScaleGRUModel(LinearArchitectureModel):
    """
    多尺度GRU轨迹生成模型
    使用不同时间尺度的GRU捕获不同频率的运动模式
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_scales = config.get('num_scales', 3)
        self.scale_factors = config.get('scale_factors', [1, 2, 4])
        
        # 输入编码器
        self.input_encoder = nn.Linear(self.input_dim * 2, self.hidden_dim)
        
        # 多尺度GRU
        self.scale_grus = nn.ModuleList()
        for scale in self.scale_factors:
            gru = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim // len(self.scale_factors),
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
            self.scale_grus.append(gru)
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        多尺度前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码输入
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        encoded_input = self.input_encoder(combined_pose)
        
        # 多尺度处理
        scale_outputs = []
        
        for scale_factor, gru in zip(self.scale_factors, self.scale_grus):
            # 创建不同尺度的输入序列
            scale_seq_length = max(1, self.max_seq_length // scale_factor)
            scale_input = encoded_input.unsqueeze(1).expand(-1, scale_seq_length, -1)
            
            # GRU处理
            scale_output, _ = gru(scale_input)
            
            # 上采样到原始长度
            if scale_seq_length != self.max_seq_length:
                scale_output = self._upsample_sequence(scale_output, self.max_seq_length)
            
            scale_outputs.append(scale_output)
        
        # 融合多尺度特征
        fused_features = torch.cat(scale_outputs, dim=-1)
        fused_output = self.fusion_layer(fused_features)
        
        # 输出投影
        trajectory = self.output_projection(fused_output)
        
        # 强制边界条件
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        
        return trajectory
    
    def _upsample_sequence(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        序列上采样
        """
        batch_size, seq_length, feature_dim = sequence.shape
        device = sequence.device
        
        # 线性插值上采样
        old_indices = torch.linspace(0, seq_length - 1, seq_length, device=device)
        new_indices = torch.linspace(0, seq_length - 1, target_length, device=device)
        
        upsampled = torch.zeros(batch_size, target_length, feature_dim, device=device)
        
        for b in range(batch_size):
            for d in range(feature_dim):
                upsampled[b, :, d] = torch.interp(new_indices, old_indices, sequence[b, :, d])
        
        return upsampled


class ConditionalGRUModel(LinearArchitectureModel):
    """
    条件GRU轨迹生成模型
    支持基于任务条件的轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.condition_dim = config.get('condition_dim', 32)
        self.use_condition_attention = config.get('use_condition_attention', True)
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.condition_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 条件注意力
        if self.use_condition_attention:
            self.condition_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,
                dropout=self.dropout,
                batch_first=True
            )
        
        # 输入编码器（包含条件）
        self.input_encoder = nn.Linear(
            self.input_dim * 2 + self.condition_dim, 
            self.hidden_dim
        )
        
        # GRU层
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        条件前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition = self.condition_encoder(combined_pose)
        
        # 创建条件化输入
        condition_expanded = condition.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        pose_expanded = combined_pose.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        
        conditioned_input = torch.cat([pose_expanded, condition_expanded], dim=-1)
        encoded_input = self.input_encoder(conditioned_input)
        
        # GRU处理
        gru_output, _ = self.gru(encoded_input)
        
        # 条件注意力
        if self.use_condition_attention:
            condition_key = condition.unsqueeze(1).expand(-1, self.max_seq_length, -1)
            attended_output, _ = self.condition_attention(
                gru_output, condition_key, condition_key
            )
            gru_output = gru_output + attended_output
        
        # 输出投影
        trajectory = self.output_projection(gru_output)
        
        # 强制边界条件
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        
        return trajectory