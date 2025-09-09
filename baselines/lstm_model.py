"""
LSTM Model for Trajectory Generation
LSTM轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import LinearArchitectureModel


class LSTMTrajectoryModel(LinearArchitectureModel):
    """
    LSTM轨迹生成模型
    使用LSTM网络进行序列到序列的轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bidirectional = config.get('bidirectional', False)
        self.use_attention = config.get('use_attention', False)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        
        # 输入编码器
        self.input_encoder = nn.Linear(self.input_dim * 2, self.hidden_dim)  # start + end pose
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # 输出投影层
        lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        
        if self.use_attention:
            # 注意力机制
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=self.dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_dim)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # 位置编码
        self.use_positional_encoding = config.get('use_positional_encoding', True)
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(self.hidden_dim, max_len=self.max_seq_length)
    
    def _create_input_sequence(self, start_pose: torch.Tensor, 
                             end_pose: torch.Tensor) -> torch.Tensor:
        """
        创建输入序列
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            输入序列 [batch_size, seq_length, hidden_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 连接起始和终止位姿
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)  # [batch_size, input_dim*2]
        
        # 编码到隐藏维度
        encoded_input = self.input_encoder(combined_pose)  # [batch_size, hidden_dim]
        
        # 复制到序列长度
        input_sequence = encoded_input.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        
        # 添加位置编码
        if self.use_positional_encoding:
            input_sequence = self.pos_encoder(input_sequence)
        
        return input_sequence
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                target_trajectory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            target_trajectory: 目标轨迹（用于teacher forcing）[batch_size, seq_length, output_dim]
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 创建输入序列
        input_sequence = self._create_input_sequence(start_pose, end_pose)
        
        # LSTM前向传播
        lstm_output, (hidden, cell) = self.lstm(input_sequence)
        
        # 应用注意力机制（如果启用）
        if self.use_attention:
            attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
            lstm_output = self.attention_norm(lstm_output + attended_output)
        
        # 输出投影
        trajectory = self.output_projection(lstm_output)
        
        # 确保边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _enforce_boundary_conditions(self, trajectory: torch.Tensor,
                                   start_pose: torch.Tensor, 
                                   end_pose: torch.Tensor) -> torch.Tensor:
        """
        强制执行边界条件
        
        Args:
            trajectory: 原始轨迹 [batch_size, seq_length, output_dim]
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            修正后的轨迹 [batch_size, seq_length, output_dim]
        """
        # 直接设置首末点
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        
        return trajectory
    
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
        
        # 转换为tensor
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        # 临时设置序列长度
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self.forward(start_tensor, end_tensor)
            
        # 恢复原始序列长度
        self.max_seq_length = original_seq_length
        
        return trajectory.squeeze(0).numpy()
    
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
        # 基础重建损失
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 平滑度损失
        smoothness_weight = self.config.get('smoothness_weight', 0.1)
        if smoothness_weight > 0:
            # 计算速度和加速度的平滑度
            pred_vel = torch.diff(predictions, dim=1)
            target_vel = torch.diff(targets, dim=1)
            
            pred_acc = torch.diff(pred_vel, dim=1)
            target_acc = torch.diff(target_vel, dim=1)
            
            vel_loss = nn.MSELoss()(pred_vel, target_vel)
            acc_loss = nn.MSELoss()(pred_acc, target_acc)
            
            smoothness_loss = vel_loss + acc_loss
            total_loss = mse_loss + smoothness_weight * smoothness_loss
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
            'use_attention': self.use_attention,
            'use_positional_encoding': self.use_positional_encoding,
            'teacher_forcing_ratio': self.teacher_forcing_ratio,
            'model_category': 'Linear Architecture'
        })
        return info


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, d_model]
        Returns:
            [batch_size, seq_length, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class BidirectionalLSTMModel(LSTMTrajectoryModel):
    """
    双向LSTM轨迹生成模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        config['bidirectional'] = True
        super().__init__(config)


class AttentionLSTMModel(LSTMTrajectoryModel):
    """
    带注意力机制的LSTM轨迹生成模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        config['use_attention'] = True
        super().__init__(config)


class HierarchicalLSTMModel(LinearArchitectureModel):
    """
    分层LSTM轨迹生成模型
    使用多层次的LSTM进行粗到细的轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_levels = config.get('num_levels', 2)
        self.level_seq_lengths = config.get('level_seq_lengths', [10, 50])
        
        # 每个层次的LSTM
        self.level_lstms = nn.ModuleList()
        self.level_projections = nn.ModuleList()
        
        for i, seq_len in enumerate(self.level_seq_lengths):
            # LSTM层
            lstm = nn.LSTM(
                input_size=self.hidden_dim if i == 0 else self.hidden_dim + self.output_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
            self.level_lstms.append(lstm)
            
            # 输出投影
            projection = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
            self.level_projections.append(projection)
        
        # 输入编码器
        self.input_encoder = nn.Linear(self.input_dim * 2, self.hidden_dim)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        分层前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码输入
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        encoded_input = self.input_encoder(combined_pose)
        
        # 逐层生成轨迹
        prev_trajectory = None
        
        for level, (seq_len, lstm, projection) in enumerate(
            zip(self.level_seq_lengths, self.level_lstms, self.level_projections)
        ):
            # 创建当前层的输入序列
            if level == 0:
                # 第一层：只使用编码的输入
                level_input = encoded_input.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # 后续层：上采样前一层的输出并连接
                upsampled_prev = self._upsample_trajectory(prev_trajectory, seq_len)
                level_input_base = encoded_input.unsqueeze(1).expand(-1, seq_len, -1)
                level_input = torch.cat([level_input_base, upsampled_prev], dim=-1)
            
            # LSTM前向传播
            lstm_output, _ = lstm(level_input)
            
            # 输出投影
            level_trajectory = projection(lstm_output)
            
            # 强制边界条件
            level_trajectory = self._enforce_boundary_conditions_hierarchical(
                level_trajectory, start_pose, end_pose
            )
            
            prev_trajectory = level_trajectory
        
        return prev_trajectory
    
    def _upsample_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        上采样轨迹到目标长度
        
        Args:
            trajectory: 输入轨迹 [batch_size, seq_length, output_dim]
            target_length: 目标长度
            
        Returns:
            上采样后的轨迹 [batch_size, target_length, output_dim]
        """
        # 使用线性插值进行上采样
        batch_size, seq_length, output_dim = trajectory.shape
        device = trajectory.device
        
        # 原始时间索引
        old_indices = torch.linspace(0, seq_length - 1, seq_length, device=device)
        # 新的时间索引
        new_indices = torch.linspace(0, seq_length - 1, target_length, device=device)
        
        # 对每个维度进行插值
        upsampled = torch.zeros(batch_size, target_length, output_dim, device=device)
        
        for b in range(batch_size):
            for d in range(output_dim):
                upsampled[b, :, d] = torch.interp(new_indices, old_indices, trajectory[b, :, d])
        
        return upsampled
    
    def _enforce_boundary_conditions_hierarchical(self, trajectory: torch.Tensor,
                                                start_pose: torch.Tensor, 
                                                end_pose: torch.Tensor) -> torch.Tensor:
        """
        分层模型的边界条件强制执行
        """
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        return trajectory