"""
Positional Transformer Model for Trajectory Generation
位置增强Transformer轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import TransformerVariantModel


class PositionalTransformerModel(TransformerVariantModel):
    """
    位置增强Transformer轨迹生成模型
    使用多种位置编码策略增强空间-时间建模能力
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pos_encoding_type = config.get('pos_encoding_type', 'learned')  # 'sinusoidal', 'learned', 'relative', 'rotary'
        self.use_spatial_encoding = config.get('use_spatial_encoding', True)
        self.use_temporal_encoding = config.get('use_temporal_encoding', True)
        self.relative_attention = config.get('relative_attention', False)
        
        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.d_model),  # start + end pose
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # 轨迹点嵌入
        self.trajectory_embedding = nn.Sequential(
            nn.Linear(self.output_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # 位置编码
        if self.pos_encoding_type == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(self.d_model, self.max_seq_length)
        elif self.pos_encoding_type == 'learned':
            self.pos_encoder = LearnedPositionalEncoding(self.d_model, self.max_seq_length)
        elif self.pos_encoding_type == 'relative':
            self.pos_encoder = RelativePositionalEncoding(self.d_model, self.max_seq_length)
        elif self.pos_encoding_type == 'rotary':
            self.pos_encoder = RotaryPositionalEncoding(self.d_model)
        
        # 空间位置编码（用于3D位置）
        if self.use_spatial_encoding:
            self.spatial_encoder = SpatialPositionalEncoding(self.d_model)
        
        # 时间位置编码
        if self.use_temporal_encoding:
            self.temporal_encoder = TemporalPositionalEncoding(self.d_model, self.max_seq_length)
        
        # Transformer编码器
        if self.relative_attention:
            # 使用相对位置注意力的Transformer
            self.transformer = RelativeTransformer(
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.d_model * 4,
                dropout=self.dropout,
                max_len=self.max_seq_length
            )
        else:
            # 标准Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_model * 4,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.output_dim)
        )
        
        # 可学习的轨迹查询
        self.trajectory_queries = nn.Parameter(
            torch.randn(self.max_seq_length, self.d_model) * 0.02
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                target_trajectory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            target_trajectory: 目标轨迹（用于训练时的teacher forcing）
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件信息
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition_embed = self.input_embedding(combined_pose)  # [batch_size, d_model]
        
        # 准备序列输入
        if target_trajectory is not None and self.training:
            # 训练时使用目标轨迹
            sequence_input = self.trajectory_embedding(target_trajectory)  # [batch_size, seq_length, d_model]
        else:
            # 推理时使用可学习的查询
            sequence_input = self.trajectory_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 添加条件信息到每个时间步
        condition_expanded = condition_embed.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        sequence_input = sequence_input + condition_expanded
        
        # 应用位置编码
        sequence_input = self._apply_positional_encodings(sequence_input, start_pose, end_pose)
        
        # Transformer编码
        if self.relative_attention:
            transformer_output = self.transformer(sequence_input)
        else:
            transformer_output = self.transformer(sequence_input)
        
        # 输出投影
        trajectory = self.output_projection(transformer_output)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _apply_positional_encodings(self, sequence_input: torch.Tensor,
                                  start_pose: torch.Tensor, 
                                  end_pose: torch.Tensor) -> torch.Tensor:
        """
        应用多种位置编码
        
        Args:
            sequence_input: 序列输入 [batch_size, seq_length, d_model]
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            编码后的序列 [batch_size, seq_length, d_model]
        """
        # 基础位置编码
        if self.pos_encoding_type in ['sinusoidal', 'learned']:
            sequence_input = self.pos_encoder(sequence_input)
        elif self.pos_encoding_type == 'rotary':
            sequence_input = self.pos_encoder(sequence_input)
        
        # 空间位置编码
        if self.use_spatial_encoding:
            # 生成线性插值的3D位置
            batch_size, seq_length, _ = sequence_input.shape
            device = sequence_input.device
            
            t = torch.linspace(0, 1, seq_length, device=device)
            t = t.unsqueeze(0).unsqueeze(-1)  # [1, seq_length, 1]
            
            # 假设前3维是3D位置
            start_pos_3d = start_pose[:, :3].unsqueeze(1)  # [batch_size, 1, 3]
            end_pos_3d = end_pose[:, :3].unsqueeze(1)      # [batch_size, 1, 3]
            
            interpolated_positions = start_pos_3d + t * (end_pos_3d - start_pos_3d)  # [batch_size, seq_length, 3]
            
            spatial_encoding = self.spatial_encoder(interpolated_positions)
            sequence_input = sequence_input + spatial_encoding
        
        # 时间位置编码
        if self.use_temporal_encoding:
            temporal_encoding = self.temporal_encoder(sequence_input)
            sequence_input = sequence_input + temporal_encoding
        
        return sequence_input
    
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
        
        # 位置一致性损失
        position_weight = self.config.get('position_weight', 0.1)
        if position_weight > 0:
            # 计算位置误差（假设前3维是位置）
            pred_pos = predictions[:, :, :3]
            target_pos = targets[:, :, :3]
            position_loss = nn.MSELoss()(pred_pos, target_pos)
            
            total_loss = mse_loss + position_weight * position_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'pos_encoding_type': self.pos_encoding_type,
            'use_spatial_encoding': self.use_spatial_encoding,
            'use_temporal_encoding': self.use_temporal_encoding,
            'relative_attention': self.relative_attention,
            'model_category': 'Transformer Architecture'
        })
        return info


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """
    可学习位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embedding(positions)
        
        return x + pos_embeddings


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 相对位置嵌入
        self.relative_positions = nn.Embedding(2 * max_len - 1, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # 计算相对位置矩阵
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + self.max_len - 1  # 偏移到正数范围
        
        # 获取相对位置嵌入
        rel_pos_embeddings = self.relative_positions(relative_positions)
        
        # 这里简化处理，实际应该在注意力计算中使用
        # 为了兼容，我们只返回原始输入加上平均相对位置编码
        avg_rel_pos = torch.mean(rel_pos_embeddings, dim=1)  # [seq_len, d_model]
        
        return x + avg_rel_pos.unsqueeze(0).expand(batch_size, -1, -1)


class RotaryPositionalEncoding(nn.Module):
    """
    旋转位置编码 (RoPE)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 计算旋转角度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算旋转矩阵
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        
        # 获取cos和sin值
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # [1, seq_len, d_model//2]
        sin = self.sin_cached[:seq_len].unsqueeze(0)  # [1, seq_len, d_model//2]
        
        # 分离x的偶数和奇数维度
        x_even = x[:, :, 0::2]  # [batch_size, seq_len, d_model//2]
        x_odd = x[:, :, 1::2]   # [batch_size, seq_len, d_model//2]
        
        # 应用旋转
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # 重新组合
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, 0::2] = x_rotated_even
        x_rotated[:, :, 1::2] = x_rotated_odd
        
        return x_rotated


class SpatialPositionalEncoding(nn.Module):
    """
    空间位置编码（用于3D位置）
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 3D位置编码网络
        self.position_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: 3D位置 [batch_size, seq_length, 3]
        Returns:
            空间编码 [batch_size, seq_length, d_model]
        """
        return self.position_mlp(positions)


class TemporalPositionalEncoding(nn.Module):
    """
    时间位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 时间编码网络
        self.temporal_mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [batch_size, seq_length, d_model]
        Returns:
            时间编码 [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = x.shape
        device = x.device
        
        # 生成时间步
        time_steps = torch.linspace(0, 1, seq_length, device=device)
        time_steps = time_steps.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        return self.temporal_mlp(time_steps)


class RelativeTransformer(nn.Module):
    """
    使用相对位置注意力的Transformer
    """
    
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTransformerLayer(d_model, nhead, dim_feedforward, dropout, max_len)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RelativeTransformerLayer(nn.Module):
    """
    相对位置Transformer层
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout, max_len)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class RelativeMultiHeadAttention(nn.Module):
    """
    相对位置多头注意力
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # 相对位置嵌入
        self.relative_positions_embeddings = nn.Embedding(2 * max_len - 1, self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # 线性变换
        Q = self.q_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 添加相对位置偏置
        relative_positions = self._get_relative_positions(seq_len, x.device)
        relative_embeddings = self.relative_positions_embeddings(relative_positions)
        
        # 简化的相对位置注意力计算
        relative_scores = torch.einsum('bhid,jkd->bhijk', Q, relative_embeddings)
        relative_scores = relative_scores.mean(dim=-1)  # 简化处理
        
        scores = scores + relative_scores
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 重新组合头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.out_linear(attn_output)
    
    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取相对位置矩阵
        """
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + seq_len - 1  # 偏移到正数范围
        
        return relative_positions


class AdaptivePositionalTransformer(PositionalTransformerModel):
    """
    自适应位置Transformer
    根据轨迹特征自动选择最佳位置编码策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adaptive_encoding = config.get('adaptive_encoding', True)
        
        # 位置编码选择网络
        if self.adaptive_encoding:
            self.encoding_selector = nn.Sequential(
                nn.Linear(self.input_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 4),  # 4种编码类型
                nn.Softmax(dim=-1)
            )
            
            # 多种位置编码器
            self.encoders = nn.ModuleDict({
                'sinusoidal': SinusoidalPositionalEncoding(self.d_model, self.max_seq_length),
                'learned': LearnedPositionalEncoding(self.d_model, self.max_seq_length),
                'rotary': RotaryPositionalEncoding(self.d_model),
                'none': nn.Identity()
            })
    
    def _apply_adaptive_encoding(self, sequence_input: torch.Tensor,
                               start_pose: torch.Tensor, 
                               end_pose: torch.Tensor) -> torch.Tensor:
        """
        自适应应用位置编码
        """
        if not self.adaptive_encoding:
            return self._apply_positional_encodings(sequence_input, start_pose, end_pose)
        
        # 选择编码策略
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        encoding_weights = self.encoding_selector(combined_pose)  # [batch_size, 4]
        
        # 应用加权组合的位置编码
        encoded_outputs = []
        encoding_names = ['sinusoidal', 'learned', 'rotary', 'none']
        
        for i, name in enumerate(encoding_names):
            if name == 'none':
                encoded = sequence_input
            else:
                encoded = self.encoders[name](sequence_input)
            encoded_outputs.append(encoded)
        
        # 加权组合
        weighted_output = torch.zeros_like(sequence_input)
        for i, encoded in enumerate(encoded_outputs):
            weight = encoding_weights[:, i].unsqueeze(1).unsqueeze(2)
            weighted_output += weight * encoded
        
        # 应用空间和时间编码
        if self.use_spatial_encoding or self.use_temporal_encoding:
            weighted_output = self._apply_positional_encodings(weighted_output, start_pose, end_pose)
        
        return weighted_output