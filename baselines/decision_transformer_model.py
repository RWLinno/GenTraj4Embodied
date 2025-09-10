"""
Decision Transformer for Trajectory Generation
基于决策变换器的轨迹生成方法

Decision Transformer将轨迹生成建模为序列建模问题，
使用Transformer架构直接从历史轨迹数据中学习策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import math

from .base_model import BaseTrajectoryModel


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class DecisionTransformerModel(BaseTrajectoryModel):
    """
    Decision Transformer轨迹生成模型
    
    将轨迹生成建模为条件序列生成问题，
    使用Transformer架构学习从起始点到终点的轨迹策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Transformer参数
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dropout = config.get('dropout', 0.1)
        self.max_ep_len = config.get('max_ep_len', 1000)
        
        # 嵌入层
        self.state_embed = nn.Linear(self.input_dim, self.d_model)
        self.action_embed = nn.Linear(self.output_dim, self.d_model)
        self.reward_embed = nn.Linear(1, self.d_model)
        self.timestep_embed = nn.Embedding(self.max_ep_len, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_length * 3)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # 输出层
        self.action_head = nn.Linear(self.d_model, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decision Transformer前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 上下文信息 (可选)
            
        Returns:
            生成的轨迹 [batch_size, max_seq_length, output_dim]
        """
        batch_size = start_pose.size(0)
        device = start_pose.device
        
        # 生成轨迹序列
        trajectories = []
        
        for i in range(batch_size):
            trajectory = self._generate_autoregressive(
                start_pose[i:i+1], 
                end_pose[i:i+1]
            )
            trajectories.append(trajectory)
        
        return torch.cat(trajectories, dim=0)
    
    def _generate_autoregressive(self, start_pose: torch.Tensor, end_pose: torch.Tensor) -> torch.Tensor:
        """
        自回归生成轨迹
        
        Args:
            start_pose: 起始位姿 [1, input_dim]
            end_pose: 终止位姿 [1, input_dim]
            
        Returns:
            生成的轨迹 [1, max_seq_length, output_dim]
        """
        device = start_pose.device
        
        # 初始化序列
        states = [start_pose.squeeze(0)]
        actions = []
        rewards = [torch.tensor([1.0], device=device)]  # 简化的奖励
        
        current_state = start_pose.squeeze(0)
        
        for step in range(self.max_seq_length - 1):
            # 构建输入序列
            seq_len = len(states)
            
            # 状态、动作、奖励嵌入
            state_embeds = torch.stack([self.state_embed(s) for s in states])  # [seq_len, d_model]
            
            if actions:
                action_embeds = torch.stack([self.action_embed(a) for a in actions])  # [seq_len-1, d_model]
            else:
                action_embeds = torch.empty(0, self.d_model, device=device)
            
            reward_embeds = torch.stack([self.reward_embed(r.unsqueeze(0)) for r in rewards])  # [seq_len, d_model]
            
            # 时间步嵌入
            timesteps = torch.arange(seq_len, device=device)
            time_embeds = self.timestep_embed(timesteps)
            
            # 交错排列：state, action, reward
            sequence = []
            for i in range(seq_len):
                sequence.append(state_embeds[i] + time_embeds[i])
                if i < len(action_embeds):
                    sequence.append(action_embeds[i] + time_embeds[i])
                sequence.append(reward_embeds[i] + time_embeds[i])
            
            # 转换为张量
            if sequence:
                input_seq = torch.stack(sequence).unsqueeze(0)  # [1, seq_len*3, d_model]
                
                # 位置编码
                input_seq = self.pos_encoder(input_seq.transpose(0, 1)).transpose(0, 1)
                
                # Transformer编码
                encoded = self.transformer(input_seq)  # [1, seq_len*3, d_model]
                
                # 获取最后一个状态对应的动作预测
                last_state_idx = len(sequence) - 2  # 倒数第二个是最后的状态
                if last_state_idx >= 0:
                    last_state_encoding = encoded[0, last_state_idx]
                else:
                    last_state_encoding = encoded[0, -1]
                
                # 预测下一个动作
                next_action = self.action_head(self.layer_norm(last_state_encoding))
            else:
                # 如果序列为空，使用简单的线性插值
                alpha = step / (self.max_seq_length - 1)
                next_action = (1 - alpha) * start_pose.squeeze(0) + alpha * end_pose.squeeze(0)
            
            # 更新状态 (简化的动力学模型)
            next_state = current_state + next_action * 0.1  # 简单的积分
            
            # 添加到序列
            actions.append(next_action)
            states.append(next_state)
            rewards.append(torch.tensor([1.0], device=device))
            
            current_state = next_state
            
            # 检查是否接近目标
            if torch.norm(current_state - end_pose.squeeze(0)) < 0.01:
                break
        
        # 填充到指定长度
        while len(states) < self.max_seq_length:
            states.append(states[-1])
        
        trajectory = torch.stack(states[:self.max_seq_length]).unsqueeze(0)
        return trajectory
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [output_dim]
            end_pose: 终止位姿 [output_dim]
            num_points: 轨迹点数量
            
        Returns:
            生成的轨迹 [num_points, output_dim]
        """
        self.eval()
        
        # 临时调整序列长度
        original_max_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        try:
            with torch.no_grad():
                start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0).to(self.device)
                end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0).to(self.device)
                
                trajectory = self._generate_autoregressive(start_tensor, end_tensor)
                
                return trajectory.squeeze(0).cpu().numpy()
        
        finally:
            # 恢复原始序列长度
            self.max_seq_length = original_max_seq_length
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """
        计算Decision Transformer损失函数
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            
        Returns:
            损失值
        """
        # 轨迹重构损失
        reconstruction_loss = F.mse_loss(predictions, targets)
        
        # 平滑度损失
        pred_diff = torch.diff(predictions, dim=1)
        target_diff = torch.diff(targets, dim=1)
        smoothness_loss = F.mse_loss(pred_diff, target_diff)
        
        # 终点损失
        end_loss = F.mse_loss(predictions[:, -1], targets[:, -1])
        
        # 总损失
        total_loss = reconstruction_loss + 0.1 * smoothness_loss + 2.0 * end_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'model_type': 'Decision Transformer',
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_ep_len': self.max_ep_len
        })
        return info