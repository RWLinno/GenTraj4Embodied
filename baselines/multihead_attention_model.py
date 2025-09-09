"""
Multi-Head Attention Trajectory Model for Trajectory Generation
多头注意力轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import TransformerVariantModel


class MultiHeadAttentionTrajectoryModel(TransformerVariantModel):
    """
    多头注意力轨迹生成模型
    专注于注意力机制的设计和优化
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.attention_type = config.get('attention_type', 'scaled_dot_product')  # 'scaled_dot_product', 'additive', 'multiplicative'
        self.use_cross_attention = config.get('use_cross_attention', True)
        self.use_self_attention = config.get('use_self_attention', True)
        self.attention_dropout = config.get('attention_dropout', 0.1)
        self.num_attention_layers = config.get('num_attention_layers', 3)
        
        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 轨迹查询嵌入
        self.trajectory_queries = nn.Parameter(
            torch.randn(self.max_seq_length, self.d_model) * 0.02
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # 多层注意力模块
        self.attention_layers = nn.ModuleList()
        
        for i in range(self.num_attention_layers):
            layer = MultiHeadAttentionLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dropout=self.attention_dropout,
                attention_type=self.attention_type,
                use_cross_attention=self.use_cross_attention,
                use_self_attention=self.use_self_attention
            )
            self.attention_layers.append(layer)
        
        # 最终输出层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.output_dim)
        )
        
        # 条件融合模块
        self.condition_fusion = ConditionFusion(self.d_model, self.dropout)
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            return_attention: 是否返回注意力权重
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件信息
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition_embed = self.input_embedding(combined_pose)  # [batch_size, d_model]
        
        # 准备查询序列
        queries = self.trajectory_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_length, d_model]
        
        # 添加位置编码
        queries = self.pos_encoder(queries)
        
        # 条件融合
        queries = self.condition_fusion(queries, condition_embed)
        
        # 多层注意力处理
        attention_weights_list = []
        
        for attention_layer in self.attention_layers:
            queries, attention_weights = attention_layer(
                queries, condition_embed, return_attention=return_attention
            )
            
            if return_attention:
                attention_weights_list.append(attention_weights)
        
        # 输出投影
        trajectory = self.output_projection(queries)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        if return_attention:
            return trajectory, attention_weights_list
        else:
            return trajectory
    
    def _enforce_boundary_conditions(self, trajectory: torch.Tensor,
                                   start_pose: torch.Tensor, 
                                   end_pose: torch.Tensor) -> torch.Tensor:
        """
        强制执行边界条件
        """
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        return trajectory
    
    def visualize_attention(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        可视化注意力权重
        
        Args:
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 生成的轨迹 [num_points, output_dim]
            attention_weights: 各层注意力权重列表
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory, attention_weights_list = self.forward(
                start_tensor, end_tensor, return_attention=True
            )
            
        self.max_seq_length = original_seq_length
        
        # 转换为numpy
        trajectory_np = trajectory.squeeze(0).numpy()
        attention_weights_np = [
            {k: v.squeeze(0).numpy() if v is not None else None for k, v in attn.items()}
            for attn in attention_weights_list
        ]
        
        return trajectory_np, attention_weights_np
    
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
        
        # 注意力正则化损失
        attention_reg_weight = self.config.get('attention_regularization', 0.01)
        if attention_reg_weight > 0:
            # 计算注意力权重的熵，鼓励注意力分散
            _, attention_weights_list = self.forward(
                kwargs.get('start_poses'), kwargs.get('end_poses'), 
                return_attention=True
            )
            
            attention_entropy = 0.0
            for attention_weights in attention_weights_list:
                if 'self_attention' in attention_weights and attention_weights['self_attention'] is not None:
                    attn = attention_weights['self_attention']  # [batch_size, nhead, seq_len, seq_len]
                    # 计算熵
                    entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
                    attention_entropy += torch.mean(entropy)
            
            # 鼓励适度的注意力熵
            target_entropy = math.log(self.max_seq_length) * 0.5  # 目标熵
            entropy_loss = torch.abs(attention_entropy - target_entropy)
            
            total_loss = mse_loss + attention_reg_weight * entropy_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'attention_type': self.attention_type,
            'use_cross_attention': self.use_cross_attention,
            'use_self_attention': self.use_self_attention,
            'attention_dropout': self.attention_dropout,
            'num_attention_layers': self.num_attention_layers,
            'model_category': 'Transformer Architecture'
        })
        return info


class MultiHeadAttentionLayer(nn.Module):
    """
    多头注意力层
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 attention_type: str = 'scaled_dot_product',
                 use_cross_attention: bool = True,
                 use_self_attention: bool = True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention
        
        # 自注意力
        if use_self_attention:
            if attention_type == 'scaled_dot_product':
                self.self_attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                )
            elif attention_type == 'additive':
                self.self_attention = AdditiveAttention(d_model, nhead, dropout)
            elif attention_type == 'multiplicative':
                self.self_attention = MultiplicativeAttention(d_model, nhead, dropout)
        
        # 交叉注意力
        if use_cross_attention:
            if attention_type == 'scaled_dot_product':
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                )
            elif attention_type == 'additive':
                self.cross_attention = AdditiveAttention(d_model, nhead, dropout)
            elif attention_type == 'multiplicative':
                self.cross_attention = MultiplicativeAttention(d_model, nhead, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries: torch.Tensor, condition: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            queries: 查询序列 [batch_size, seq_length, d_model]
            condition: 条件信息 [batch_size, d_model]
            return_attention: 是否返回注意力权重
            
        Returns:
            output: 输出序列 [batch_size, seq_length, d_model]
            attention_weights: 注意力权重字典
        """
        attention_weights = {}
        
        # 自注意力
        if self.use_self_attention:
            if self.attention_type == 'scaled_dot_product':
                attn_output, attn_weights = self.self_attention(queries, queries, queries)
                if return_attention:
                    attention_weights['self_attention'] = attn_weights
            else:
                attn_output, attn_weights = self.self_attention(queries, queries, queries)
                if return_attention:
                    attention_weights['self_attention'] = attn_weights
            
            queries = self.norm1(queries + self.dropout(attn_output))
        
        # 交叉注意力
        if self.use_cross_attention:
            # 扩展条件信息作为键和值
            condition_expanded = condition.unsqueeze(1)  # [batch_size, 1, d_model]
            
            if self.attention_type == 'scaled_dot_product':
                cross_attn_output, cross_attn_weights = self.cross_attention(
                    queries, condition_expanded, condition_expanded
                )
                if return_attention:
                    attention_weights['cross_attention'] = cross_attn_weights
            else:
                cross_attn_output, cross_attn_weights = self.cross_attention(
                    queries, condition_expanded, condition_expanded
                )
                if return_attention:
                    attention_weights['cross_attention'] = cross_attn_weights
            
            queries = self.norm2(queries + self.dropout(cross_attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(queries)
        queries = self.norm3(queries + self.dropout(ff_output))
        
        return queries, attention_weights


class AdditiveAttention(nn.Module):
    """
    加性注意力机制
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # 加性注意力的参数
        self.attention_weights = nn.Linear(self.head_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加性注意力前向传播
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 线性变换
        Q = self.query_projection(query).view(batch_size, seq_len_q, self.nhead, self.head_dim)
        K = self.key_projection(key).view(batch_size, seq_len_k, self.nhead, self.head_dim)
        V = self.value_projection(value).view(batch_size, seq_len_k, self.nhead, self.head_dim)
        
        # 计算加性注意力分数
        # Q: [batch_size, seq_len_q, nhead, head_dim]
        # K: [batch_size, seq_len_k, nhead, head_dim]
        Q_expanded = Q.unsqueeze(2)  # [batch_size, seq_len_q, 1, nhead, head_dim]
        K_expanded = K.unsqueeze(1)  # [batch_size, 1, seq_len_k, nhead, head_dim]
        
        # 加性注意力：tanh(Q + K)
        combined = self.tanh(Q_expanded + K_expanded)  # [batch_size, seq_len_q, seq_len_k, nhead, head_dim]
        
        # 计算注意力权重
        attention_scores = self.attention_weights(combined).squeeze(-1)  # [batch_size, seq_len_q, seq_len_k, nhead]
        attention_weights = torch.softmax(attention_scores, dim=2)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        # V: [batch_size, seq_len_k, nhead, head_dim]
        # attention_weights: [batch_size, seq_len_q, seq_len_k, nhead]
        V_expanded = V.unsqueeze(1).expand(-1, seq_len_q, -1, -1, -1)  # [batch_size, seq_len_q, seq_len_k, nhead, head_dim]
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch_size, seq_len_q, seq_len_k, nhead, 1]
        
        attended_values = torch.sum(V_expanded * attention_weights_expanded, dim=2)  # [batch_size, seq_len_q, nhead, head_dim]
        
        # 重新组合头
        attended_values = attended_values.contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 输出投影
        output = self.output_projection(attended_values)
        
        # 返回平均注意力权重用于可视化
        avg_attention_weights = torch.mean(attention_weights, dim=-1)  # [batch_size, seq_len_q, seq_len_k]
        
        return output, avg_attention_weights


class MultiplicativeAttention(nn.Module):
    """
    乘性注意力机制
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # 乘性注意力的权重矩阵
        self.attention_matrix = nn.Parameter(torch.randn(self.head_dim, self.head_dim) * 0.02)
        
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        乘性注意力前向传播
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 线性变换
        Q = self.query_projection(query).view(batch_size, seq_len_q, self.nhead, self.head_dim)
        K = self.key_projection(key).view(batch_size, seq_len_k, self.nhead, self.head_dim)
        V = self.value_projection(value).view(batch_size, seq_len_k, self.nhead, self.head_dim)
        
        # 转置以便计算注意力
        Q = Q.transpose(1, 2)  # [batch_size, nhead, seq_len_q, head_dim]
        K = K.transpose(1, 2)  # [batch_size, nhead, seq_len_k, head_dim]
        V = V.transpose(1, 2)  # [batch_size, nhead, seq_len_k, head_dim]
        
        # 乘性注意力：Q * W * K^T
        # 首先计算 K * W^T
        K_weighted = torch.matmul(K, self.attention_matrix.T)  # [batch_size, nhead, seq_len_k, head_dim]
        
        # 然后计算 Q * (K * W^T)^T
        attention_scores = torch.matmul(Q, K_weighted.transpose(-2, -1)) / self.scale
        
        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, V)  # [batch_size, nhead, seq_len_q, head_dim]
        
        # 重新组合头
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # 输出投影
        output = self.output_projection(attended_values)
        
        # 返回平均注意力权重用于可视化
        avg_attention_weights = torch.mean(attention_weights, dim=1)  # [batch_size, seq_len_q, seq_len_k]
        
        return output, avg_attention_weights


class ConditionFusion(nn.Module):
    """
    条件融合模块
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 条件变换
        self.condition_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, queries: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        条件融合
        
        Args:
            queries: 查询序列 [batch_size, seq_length, d_model]
            condition: 条件信息 [batch_size, d_model]
            
        Returns:
            融合后的查询序列 [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = queries.shape
        
        # 扩展条件信息
        condition_expanded = condition.unsqueeze(1).expand(-1, seq_length, -1)
        
        # 计算门控权重
        combined = torch.cat([queries, condition_expanded], dim=-1)
        gate_weights = self.gate(combined)
        
        # 变换条件信息
        transformed_condition = self.condition_transform(condition_expanded)
        
        # 门控融合
        fused_queries = queries + gate_weights * transformed_condition
        
        return fused_queries


class PositionalEncoding(nn.Module):
    """
    位置编码模块
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


class SparseAttentionModel(MultiHeadAttentionTrajectoryModel):
    """
    稀疏注意力模型
    使用稀疏注意力模式提高效率
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.attention_pattern = config.get('attention_pattern', 'local')  # 'local', 'strided', 'random'
        self.local_window_size = config.get('local_window_size', 8)
        self.stride = config.get('stride', 4)
        self.random_ratio = config.get('random_ratio', 0.1)
        
    def _create_sparse_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        创建稀疏注意力掩码
        
        Args:
            seq_length: 序列长度
            device: 设备
            
        Returns:
            稀疏掩码 [seq_length, seq_length]
        """
        mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)
        
        if self.attention_pattern == 'local':
            # 局部注意力
            for i in range(seq_length):
                start = max(0, i - self.local_window_size // 2)
                end = min(seq_length, i + self.local_window_size // 2 + 1)
                mask[i, start:end] = True
                
        elif self.attention_pattern == 'strided':
            # 步长注意力
            for i in range(seq_length):
                # 局部窗口
                start = max(0, i - self.local_window_size // 2)
                end = min(seq_length, i + self.local_window_size // 2 + 1)
                mask[i, start:end] = True
                
                # 步长连接
                for j in range(0, seq_length, self.stride):
                    mask[i, j] = True
                    
        elif self.attention_pattern == 'random':
            # 随机注意力
            for i in range(seq_length):
                # 局部窗口
                start = max(0, i - self.local_window_size // 2)
                end = min(seq_length, i + self.local_window_size // 2 + 1)
                mask[i, start:end] = True
                
                # 随机连接
                num_random = int(seq_length * self.random_ratio)
                random_indices = torch.randperm(seq_length, device=device)[:num_random]
                mask[i, random_indices] = True
        
        return mask


class HierarchicalAttentionModel(MultiHeadAttentionTrajectoryModel):
    """
    分层注意力模型
    使用多层次的注意力机制
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_hierarchy_levels = config.get('num_hierarchy_levels', 2)
        self.level_seq_lengths = config.get('level_seq_lengths', [10, 50])
        
        # 多层次的注意力模块
        self.hierarchy_attention_layers = nn.ModuleList()
        
        for i, seq_len in enumerate(self.level_seq_lengths):
            level_layers = nn.ModuleList([
                MultiHeadAttentionLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dropout=self.attention_dropout,
                    attention_type=self.attention_type,
                    use_cross_attention=self.use_cross_attention,
                    use_self_attention=self.use_self_attention
                ) for _ in range(self.num_attention_layers // len(self.level_seq_lengths))
            ])
            self.hierarchy_attention_layers.append(level_layers)
        
        # 层间融合
        self.level_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
                nn.Dropout(self.dropout)
            ) for _ in range(len(self.level_seq_lengths) - 1)
        ])
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        分层注意力前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition_embed = self.input_embedding(combined_pose)
        
        prev_output = None
        
        for level, (seq_len, attention_layers) in enumerate(
            zip(self.level_seq_lengths, self.hierarchy_attention_layers)
        ):
            # 当前层的查询
            if level == 0:
                level_queries = self.trajectory_queries[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # 上采样前一层的输出
                upsampled_prev = self._upsample_features(prev_output, seq_len)
                base_queries = self.trajectory_queries[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                
                # 融合前一层信息
                combined_features = torch.cat([base_queries, upsampled_prev], dim=-1)
                level_queries = self.level_fusion[level-1](combined_features)
            
            # 位置编码
            level_queries = self.pos_encoder(level_queries)
            
            # 条件融合
            level_queries = self.condition_fusion(level_queries, condition_embed)
            
            # 多层注意力处理
            for attention_layer in attention_layers:
                level_queries, _ = attention_layer(level_queries, condition_embed)
            
            prev_output = level_queries
        
        # 最终输出投影
        trajectory = self.output_projection(prev_output)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _upsample_features(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        上采样特征到目标长度
        """
        batch_size, seq_length, feature_dim = features.shape
        device = features.device
        
        if seq_length == target_length:
            return features
        
        # 线性插值上采样
        old_indices = torch.linspace(0, seq_length - 1, seq_length, device=device)
        new_indices = torch.linspace(0, seq_length - 1, target_length, device=device)
        
        upsampled = torch.zeros(batch_size, target_length, feature_dim, device=device)
        
        for b in range(batch_size):
            for d in range(feature_dim):
                upsampled[b, :, d] = torch.interp(new_indices, old_indices, features[b, :, d])
        
        return upsampled