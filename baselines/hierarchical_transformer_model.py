"""
Hierarchical Transformer Model for Trajectory Generation
分层Transformer轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import TransformerVariantModel


class HierarchicalTransformerModel(TransformerVariantModel):
    """
    分层Transformer轨迹生成模型
    使用多层次的Transformer进行粗到细的轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_levels = config.get('num_levels', 3)
        self.level_seq_lengths = config.get('level_seq_lengths', [5, 15, 50])
        self.level_d_models = config.get('level_d_models', [128, 256, 512])
        self.level_num_layers = config.get('level_num_layers', [2, 3, 4])
        self.use_cross_level_attention = config.get('use_cross_level_attention', True)
        self.progressive_refinement = config.get('progressive_refinement', True)
        
        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 多层次的Transformer编码器
        self.level_transformers = nn.ModuleList()
        self.level_embeddings = nn.ModuleList()
        self.level_projections = nn.ModuleList()
        self.level_pos_encoders = nn.ModuleList()
        
        for i, (seq_len, d_model, num_layers) in enumerate(
            zip(self.level_seq_lengths, self.level_d_models, self.level_num_layers)
        ):
            # 层级嵌入
            level_embedding = nn.Sequential(
                nn.Linear(self.d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
            self.level_embeddings.append(level_embedding)
            
            # 位置编码
            pos_encoder = PositionalEncoding(d_model, seq_len)
            self.level_pos_encoders.append(pos_encoder)
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=min(self.nhead, d_model // 64),  # 确保head数量合理
                dim_feedforward=d_model * 4,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.level_transformers.append(transformer)
            
            # 输出投影
            projection = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.output_dim)
            )
            self.level_projections.append(projection)
        
        # 跨层注意力模块
        if self.use_cross_level_attention:
            self.cross_level_attention = nn.ModuleList()
            for i in range(1, self.num_levels):
                cross_attn = CrossLevelAttention(
                    d_model_prev=self.level_d_models[i-1],
                    d_model_curr=self.level_d_models[i],
                    nhead=min(self.nhead, min(self.level_d_models[i-1], self.level_d_models[i]) // 64),
                    dropout=self.dropout
                )
                self.cross_level_attention.append(cross_attn)
        
        # 层级融合模块
        self.level_fusion = nn.ModuleList()
        for i in range(1, self.num_levels):
            fusion = LevelFusion(
                d_model_prev=self.level_d_models[i-1],
                d_model_curr=self.level_d_models[i],
                output_dim=self.output_dim,
                dropout=self.dropout
            )
            self.level_fusion.append(fusion)
        
        # 可学习的层级查询
        self.level_queries = nn.ParameterList([
            nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
            for seq_len, d_model in zip(self.level_seq_lengths, self.level_d_models)
        ])
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                return_intermediate: bool = False) -> torch.Tensor:
        """
        分层前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            return_intermediate: 是否返回中间结果
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件信息
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition_embed = self.input_embedding(combined_pose)  # [batch_size, d_model]
        
        # 存储中间结果
        intermediate_results = []
        prev_trajectory = None
        prev_features = None
        
        # 逐层处理
        for level in range(self.num_levels):
            seq_len = self.level_seq_lengths[level]
            d_model = self.level_d_models[level]
            
            # 准备当前层的查询
            level_queries = self.level_queries[level].unsqueeze(0).expand(batch_size, -1, -1)
            
            # 如果当前层的序列长度与level_queries不匹配，进行调整
            if level_queries.shape[1] != seq_len:
                # 使用插值调整level_queries的长度
                level_queries_adjusted = torch.nn.functional.interpolate(
                    level_queries.transpose(1, 2),  # [batch_size, d_model, original_seq_len]
                    size=seq_len,
                    mode='linear',
                    align_corners=True
                ).transpose(1, 2)  # [batch_size, seq_len, d_model]
                level_queries = level_queries_adjusted
            
            # 层级嵌入
            # 先将condition_embed转换到当前层的维度
            condition_embed_level = self.level_embeddings[level](condition_embed)
            # 然后与level_queries相加
            level_queries = condition_embed_level.unsqueeze(1).expand(-1, seq_len, -1) + level_queries
            
            # 位置编码
            level_queries = self.level_pos_encoders[level](level_queries)
            
            # 跨层注意力（如果不是第一层）
            if level > 0 and self.use_cross_level_attention and prev_features is not None:
                level_queries = self.cross_level_attention[level-1](
                    level_queries, prev_features
                )
            
            # Transformer编码
            level_features = self.level_transformers[level](level_queries)
            
            # 输出投影
            level_trajectory = self.level_projections[level](level_features)
            
            # 层级融合（如果不是第一层）
            if level > 0 and prev_trajectory is not None:
                level_trajectory = self.level_fusion[level-1](
                    prev_trajectory, level_trajectory, level_features
                )
            
            # 强制边界条件
            level_trajectory = self._enforce_boundary_conditions(
                level_trajectory, start_pose, end_pose
            )
            
            # 渐进式细化
            if self.progressive_refinement and prev_trajectory is not None:
                # 上采样前一层轨迹
                upsampled_prev = self._upsample_trajectory(prev_trajectory, seq_len)
                # 残差连接
                level_trajectory = level_trajectory + upsampled_prev
                # 再次强制边界条件
                level_trajectory = self._enforce_boundary_conditions(
                    level_trajectory, start_pose, end_pose
                )
            
            # 存储中间结果
            if return_intermediate:
                intermediate_results.append(level_trajectory.clone())
            
            prev_trajectory = level_trajectory
            prev_features = level_features
        
        if return_intermediate:
            return prev_trajectory, intermediate_results
        else:
            return prev_trajectory
    
    def _enforce_boundary_conditions(self, trajectory: torch.Tensor,
                                   start_pose: torch.Tensor, 
                                   end_pose: torch.Tensor) -> torch.Tensor:
        """
        强制执行边界条件
        """
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        return trajectory
    
    def _upsample_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        上采样轨迹到目标长度
        """
        batch_size, seq_length, output_dim = trajectory.shape
        device = trajectory.device
        
        if seq_length == target_length:
            return trajectory
        
        # 线性插值上采样
        old_indices = torch.linspace(0, seq_length - 1, seq_length, device=device)
        new_indices = torch.linspace(0, seq_length - 1, target_length, device=device)
        
        upsampled = torch.zeros(batch_size, target_length, output_dim, device=device)
        
        for b in range(batch_size):
            for d in range(output_dim):
                # 使用线性插值替代torch.interp
                upsampled[b, :, d] = torch.nn.functional.interpolate(
                    trajectory[b, :, d].unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode='linear',
                    align_corners=True
                ).squeeze()
        
        return upsampled
    
    def generate_trajectory_progressive(self, start_pose: np.ndarray, end_pose: np.ndarray,
                                     num_points: int = 50, return_all_levels: bool = False) -> np.ndarray:
        """
        渐进式生成轨迹
        
        Args:
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            return_all_levels: 是否返回所有层级的结果
            
        Returns:
            生成的轨迹或所有层级的轨迹列表
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        # 临时设置最高层的序列长度
        original_seq_length = self.level_seq_lengths[-1]
        self.level_seq_lengths[-1] = num_points
        
        with torch.no_grad():
            if return_all_levels:
                final_trajectory, intermediate_results = self.forward(
                    start_tensor, end_tensor, return_intermediate=True
                )
                
                # 转换为numpy
                results = []
                for traj in intermediate_results:
                    results.append(traj.squeeze(0).numpy())
                results.append(final_trajectory.squeeze(0).numpy())
                
                self.level_seq_lengths[-1] = original_seq_length
                return results
            else:
                trajectory = self.forward(start_tensor, end_tensor)
                
                # 如果生成的轨迹长度与期望不符，进行插值调整
                if trajectory.shape[1] != num_points:
                    trajectory = self._downsample_trajectory(trajectory, num_points)
                
        self.level_seq_lengths[-1] = original_seq_length
        
        return trajectory.squeeze(0).numpy()
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        """
        return self.generate_trajectory_progressive(
            start_pose, end_pose, num_points, return_all_levels=False
        )
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    start_poses: torch.Tensor, end_poses: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算分层损失函数
        """
        # 获取所有层级的输出
        _, intermediate_results = self.forward(
            start_poses, end_poses, return_intermediate=True
        )
        
        total_loss = 0.0
        loss_weights = kwargs.get('level_loss_weights', [0.1, 0.3, 1.0])  # 高层级权重更大
        
        # 计算每个层级的损失
        for i, (level_pred, weight) in enumerate(zip(intermediate_results, loss_weights)):
            # 下采样目标轨迹到当前层级的长度
            target_length = level_pred.shape[1]
            downsampled_targets = self._downsample_trajectory(targets, target_length)
            
            # 计算损失
            level_loss = nn.MSELoss()(level_pred, downsampled_targets)
            total_loss += weight * level_loss
        
        # 多尺度一致性损失
        consistency_weight = self.config.get('consistency_weight', 0.1)
        if consistency_weight > 0 and len(intermediate_results) > 1:
            consistency_loss = 0.0
            
            for i in range(len(intermediate_results) - 1):
                # 上采样低层级结果
                low_level = intermediate_results[i]
                high_level = intermediate_results[i + 1]
                
                upsampled_low = self._upsample_trajectory(low_level, high_level.shape[1])
                
                # 计算一致性损失
                consistency_loss += nn.MSELoss()(upsampled_low, high_level)
            
            total_loss += consistency_weight * consistency_loss
        
        return total_loss
    
    def _downsample_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        下采样轨迹到目标长度
        """
        batch_size, seq_length, output_dim = trajectory.shape
        device = trajectory.device
        
        if seq_length == target_length:
            return trajectory
        
        # 线性插值下采样
        old_indices = torch.linspace(0, seq_length - 1, seq_length, device=device)
        new_indices = torch.linspace(0, seq_length - 1, target_length, device=device)
        
        downsampled = torch.zeros(batch_size, target_length, output_dim, device=device)
        
        for b in range(batch_size):
            for d in range(output_dim):
                # 使用线性插值替代torch.interp
                downsampled[b, :, d] = torch.nn.functional.interpolate(
                    trajectory[b, :, d].unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode='linear',
                    align_corners=True
                ).squeeze()
        
        return downsampled
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'num_levels': self.num_levels,
            'level_seq_lengths': self.level_seq_lengths,
            'level_d_models': self.level_d_models,
            'level_num_layers': self.level_num_layers,
            'use_cross_level_attention': self.use_cross_level_attention,
            'progressive_refinement': self.progressive_refinement,
            'model_category': 'Transformer Architecture'
        })
        return info


class CrossLevelAttention(nn.Module):
    """
    跨层注意力模块
    """
    
    def __init__(self, d_model_prev: int, d_model_curr: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model_prev = d_model_prev
        self.d_model_curr = d_model_curr
        
        # 维度对齐
        if d_model_prev != d_model_curr:
            self.prev_projection = nn.Linear(d_model_prev, d_model_curr)
        else:
            self.prev_projection = nn.Identity()
        
        # 跨层注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model_curr,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model_curr)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, curr_features: torch.Tensor, prev_features: torch.Tensor) -> torch.Tensor:
        """
        跨层注意力前向传播
        
        Args:
            curr_features: 当前层特征 [batch_size, curr_seq_len, d_model_curr]
            prev_features: 前一层特征 [batch_size, prev_seq_len, d_model_prev]
            
        Returns:
            增强的当前层特征 [batch_size, curr_seq_len, d_model_curr]
        """
        # 维度对齐
        aligned_prev_features = self.prev_projection(prev_features)
        
        # 跨层注意力
        attn_output, _ = self.cross_attention(
            query=curr_features,
            key=aligned_prev_features,
            value=aligned_prev_features
        )
        
        # 残差连接和层归一化
        output = self.norm(curr_features + self.dropout(attn_output))
        
        return output


class LevelFusion(nn.Module):
    """
    层级融合模块
    """
    
    def __init__(self, d_model_prev: int, d_model_curr: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model_prev = d_model_prev
        self.d_model_curr = d_model_curr
        self.output_dim = output_dim
        
        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(output_dim * 2 + d_model_curr, d_model_curr),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model_curr, d_model_curr // 2),
            nn.GELU(),
            nn.Linear(d_model_curr // 2, output_dim)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, prev_trajectory: torch.Tensor, curr_trajectory: torch.Tensor,
                curr_features: torch.Tensor) -> torch.Tensor:
        """
        层级融合前向传播
        
        Args:
            prev_trajectory: 前一层轨迹 [batch_size, prev_seq_len, output_dim]
            curr_trajectory: 当前层轨迹 [batch_size, curr_seq_len, output_dim]
            curr_features: 当前层特征 [batch_size, curr_seq_len, d_model_curr]
            
        Returns:
            融合后的轨迹 [batch_size, curr_seq_len, output_dim]
        """
        batch_size, curr_seq_len, _ = curr_trajectory.shape
        
        # 上采样前一层轨迹
        upsampled_prev = self._upsample_trajectory(prev_trajectory, curr_seq_len)
        
        # 计算门控权重
        gate_input = torch.cat([upsampled_prev, curr_trajectory], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # 特征融合
        fusion_input = torch.cat([upsampled_prev, curr_trajectory, curr_features], dim=-1)
        fusion_output = self.fusion_network(fusion_input)
        
        # 门控融合
        fused_trajectory = gate_weights * upsampled_prev + (1 - gate_weights) * fusion_output
        
        return fused_trajectory
    
    def _upsample_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        上采样轨迹到目标长度
        """
        batch_size, seq_length, output_dim = trajectory.shape
        device = trajectory.device
        
        if seq_length == target_length:
            return trajectory
        
        # 线性插值上采样
        old_indices = torch.linspace(0, seq_length - 1, seq_length, device=device)
        new_indices = torch.linspace(0, seq_length - 1, target_length, device=device)
        
        upsampled = torch.zeros(batch_size, target_length, output_dim, device=device)
        
        for b in range(batch_size):
            for d in range(output_dim):
                # 使用线性插值替代torch.interp
                upsampled[b, :, d] = torch.nn.functional.interpolate(
                    trajectory[b, :, d].unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode='linear',
                    align_corners=True
                ).squeeze()
        
        return upsampled


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


class AdaptiveHierarchicalTransformer(HierarchicalTransformerModel):
    """
    自适应分层Transformer
    根据任务复杂度自动调整层级结构
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adaptive_levels = config.get('adaptive_levels', True)
        self.min_levels = config.get('min_levels', 2)
        self.max_levels = config.get('max_levels', 4)
        
        # 复杂度评估网络
        if self.adaptive_levels:
            self.complexity_estimator = nn.Sequential(
                nn.Linear(self.input_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def _estimate_complexity(self, start_pose: torch.Tensor, end_pose: torch.Tensor) -> float:
        """
        估计轨迹复杂度
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            复杂度分数 [0, 1]
        """
        if not self.adaptive_levels:
            return 0.5
        
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        complexity = self.complexity_estimator(combined_pose).mean().item()
        
        return complexity
    
    def _select_active_levels(self, complexity: float) -> List[int]:
        """
        根据复杂度选择激活的层级
        
        Args:
            complexity: 复杂度分数
            
        Returns:
            激活层级的索引列表
        """
        # 根据复杂度确定层级数量
        num_active_levels = int(
            self.min_levels + complexity * (self.max_levels - self.min_levels)
        )
        num_active_levels = max(self.min_levels, min(self.max_levels, num_active_levels))
        
        # 选择层级（从粗到细）
        if num_active_levels >= self.num_levels:
            return list(range(self.num_levels))
        else:
            # 均匀分布选择层级
            step = self.num_levels / num_active_levels
            active_levels = [int(i * step) for i in range(num_active_levels)]
            # 确保包含最后一层
            if active_levels[-1] != self.num_levels - 1:
                active_levels[-1] = self.num_levels - 1
            
            return active_levels
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        自适应分层前向传播
        """
        if not self.adaptive_levels:
            return super().forward(start_pose, end_pose, context)
        
        # 估计复杂度
        complexity = self._estimate_complexity(start_pose, end_pose)
        
        # 选择激活层级
        active_levels = self._select_active_levels(complexity)
        
        # 执行选择的层级
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件信息
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition_embed = self.input_embedding(combined_pose)
        
        prev_trajectory = None
        prev_features = None
        
        for i, level in enumerate(active_levels):
            seq_len = self.level_seq_lengths[level]
            d_model = self.level_d_models[level]
            
            # 准备查询
            level_queries = self.level_queries[level].unsqueeze(0).expand(batch_size, -1, -1)
            level_queries = self.level_embeddings[level](
                condition_embed.unsqueeze(1).expand(-1, seq_len, -1) + level_queries
            )
            level_queries = self.level_pos_encoders[level](level_queries)
            
            # 跨层注意力
            if i > 0 and self.use_cross_level_attention and prev_features is not None:
                # 使用前一个激活层级的特征
                prev_level_idx = active_levels[i-1]
                level_queries = self.cross_level_attention[level-1](
                    level_queries, prev_features
                )
            
            # Transformer编码
            level_features = self.level_transformers[level](level_queries)
            
            # 输出投影
            level_trajectory = self.level_projections[level](level_features)
            
            # 层级融合
            if i > 0 and prev_trajectory is not None:
                prev_level_idx = active_levels[i-1]
                level_trajectory = self.level_fusion[level-1](
                    prev_trajectory, level_trajectory, level_features
                )
            
            # 强制边界条件
            level_trajectory = self._enforce_boundary_conditions(
                level_trajectory, start_pose, end_pose
            )
            
            # 渐进式细化
            if self.progressive_refinement and prev_trajectory is not None:
                upsampled_prev = self._upsample_trajectory(prev_trajectory, seq_len)
                level_trajectory = level_trajectory + upsampled_prev
                level_trajectory = self._enforce_boundary_conditions(
                    level_trajectory, start_pose, end_pose
                )
            
            prev_trajectory = level_trajectory
            prev_features = level_features
        
        return prev_trajectory


class MultiTaskHierarchicalTransformer(HierarchicalTransformerModel):
    """
    多任务分层Transformer
    支持多种轨迹生成任务
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_tasks = config.get('num_tasks', 3)
        self.task_embedding_dim = config.get('task_embedding_dim', 64)
        
        # 任务嵌入
        self.task_embedding = nn.Embedding(self.num_tasks, self.task_embedding_dim)
        
        # 任务特定的输出头
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.level_d_models[-1]),
                nn.Linear(self.level_d_models[-1], self.level_d_models[-1] // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.level_d_models[-1] // 2, self.output_dim)
            ) for _ in range(self.num_tasks)
        ])
        
        # 任务融合网络
        self.task_fusion = nn.Sequential(
            nn.Linear(self.d_model + self.task_embedding_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                task_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        多任务前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            task_id: 任务ID [batch_size] (0到num_tasks-1)
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 默认任务ID
        if task_id is None:
            task_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 编码条件信息
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition_embed = self.input_embedding(combined_pose)
        
        # 任务嵌入
        task_embed = self.task_embedding(task_id)  # [batch_size, task_embedding_dim]
        
        # 融合任务信息
        task_condition = torch.cat([condition_embed, task_embed], dim=-1)
        fused_condition = self.task_fusion(task_condition)
        
        # 使用融合后的条件进行分层处理
        prev_trajectory = None
        prev_features = None
        
        for level in range(self.num_levels):
            seq_len = self.level_seq_lengths[level]
            
            # 准备查询
            level_queries = self.level_queries[level].unsqueeze(0).expand(batch_size, -1, -1)
            level_queries = self.level_embeddings[level](
                fused_condition.unsqueeze(1).expand(-1, seq_len, -1) + level_queries
            )
            level_queries = self.level_pos_encoders[level](level_queries)
            
            # 跨层注意力
            if level > 0 and self.use_cross_level_attention and prev_features is not None:
                level_queries = self.cross_level_attention[level-1](
                    level_queries, prev_features
                )
            
            # Transformer编码
            level_features = self.level_transformers[level](level_queries)
            
            # 任务特定输出（仅最后一层）
            if level == self.num_levels - 1:
                # 使用任务特定的输出头
                level_trajectories = []
                for b in range(batch_size):
                    task_head = self.task_heads[task_id[b].item()]
                    traj = task_head(level_features[b:b+1])
                    level_trajectories.append(traj)
                level_trajectory = torch.cat(level_trajectories, dim=0)
            else:
                level_trajectory = self.level_projections[level](level_features)
            
            # 层级融合
            if level > 0 and prev_trajectory is not None:
                level_trajectory = self.level_fusion[level-1](
                    prev_trajectory, level_trajectory, level_features
                )
            
            # 强制边界条件
            level_trajectory = self._enforce_boundary_conditions(
                level_trajectory, start_pose, end_pose
            )
            
            # 渐进式细化
            if self.progressive_refinement and prev_trajectory is not None:
                upsampled_prev = self._upsample_trajectory(prev_trajectory, seq_len)
                level_trajectory = level_trajectory + upsampled_prev
                level_trajectory = self._enforce_boundary_conditions(
                    level_trajectory, start_pose, end_pose
                )
            
            prev_trajectory = level_trajectory
            prev_features = level_features
        
        return prev_trajectory