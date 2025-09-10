"""
Mixture Density Network (MDN) Model for Trajectory Generation
混合密度网络轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from .base_model import LinearArchitectureModel


class MDNTrajectoryModel(LinearArchitectureModel):
    """
    混合密度网络(MDN)轨迹生成模型
    结合LSTM和MDN，支持多模态轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_mixtures = config.get('num_mixtures', 5)
        self.use_lstm = config.get('use_lstm', True)
        self.use_attention = config.get('use_attention', False)
        
        # 输入编码器
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 序列建模层
        if self.use_lstm:
            self.sequence_model = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.sequence_model = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        
        # 注意力机制（可选）
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=self.dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(self.hidden_dim)
        
        # MDN输出层
        # 每个混合成分需要：均值(output_dim) + 方差(output_dim) + 权重(1)
        mdn_output_dim = self.num_mixtures * (2 * self.output_dim + 1)
        
        self.mdn_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, mdn_output_dim)
        )
        
        # 位置编码
        self.position_encoding = nn.Parameter(
            torch.randn(self.max_seq_length, self.hidden_dim) * 0.1
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                return_parameters: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            return_parameters: 是否返回MDN参数
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim] 或 MDN参数
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码输入
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        encoded_input = self.input_encoder(combined_pose)
        
        # 创建序列输入
        input_sequence = encoded_input.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        
        # 添加位置编码
        input_sequence = input_sequence + self.position_encoding.unsqueeze(0)
        
        # 序列建模
        if self.use_lstm:
            sequence_output, _ = self.sequence_model(input_sequence)
        else:
            sequence_output, _ = self.sequence_model(input_sequence)
        
        # 注意力机制
        if self.use_attention:
            attended_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
            sequence_output = self.attention_norm(sequence_output + attended_output)
        
        # MDN输出
        mdn_output = self.mdn_layer(sequence_output)  # [batch_size, seq_length, mdn_output_dim]
        
        # 解析MDN参数
        pi, mu, sigma = self._parse_mdn_parameters(mdn_output)
        
        if return_parameters:
            return pi, mu, sigma
        
        # 从MDN采样轨迹
        trajectory = self._sample_from_mdn(pi, mu, sigma)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _parse_mdn_parameters(self, mdn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解析MDN参数
        
        Args:
            mdn_output: MDN原始输出 [batch_size, seq_length, mdn_output_dim]
            
        Returns:
            pi: 混合权重 [batch_size, seq_length, num_mixtures]
            mu: 均值 [batch_size, seq_length, num_mixtures, output_dim]
            sigma: 标准差 [batch_size, seq_length, num_mixtures, output_dim]
        """
        batch_size, seq_length, _ = mdn_output.shape
        
        # 分离参数
        pi_logits = mdn_output[:, :, :self.num_mixtures]
        mu_params = mdn_output[:, :, self.num_mixtures:self.num_mixtures*(1+self.output_dim)]
        sigma_params = mdn_output[:, :, self.num_mixtures*(1+self.output_dim):]
        
        # 混合权重（softmax归一化）
        pi = torch.softmax(pi_logits, dim=-1)
        
        # 均值（重塑形状）
        mu = mu_params.view(batch_size, seq_length, self.num_mixtures, self.output_dim)
        
        # 标准差（指数函数确保正值）
        sigma = torch.exp(sigma_params.view(batch_size, seq_length, self.num_mixtures, self.output_dim)) + 1e-6
        
        return pi, mu, sigma
    
    def _sample_from_mdn(self, pi: torch.Tensor, mu: torch.Tensor, 
                        sigma: torch.Tensor) -> torch.Tensor:
        """
        从MDN采样轨迹
        
        Args:
            pi: 混合权重 [batch_size, seq_length, num_mixtures]
            mu: 均值 [batch_size, seq_length, num_mixtures, output_dim]
            sigma: 标准差 [batch_size, seq_length, num_mixtures, output_dim]
            
        Returns:
            采样的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size, seq_length, num_mixtures, output_dim = mu.shape
        device = mu.device
        
        # 为每个时间步采样
        trajectory = torch.zeros(batch_size, seq_length, output_dim, device=device)
        
        for b in range(batch_size):
            for t in range(seq_length):
                # 选择混合成分
                mixture_idx = torch.multinomial(pi[b, t], 1).item()
                
                # 从选择的高斯分布采样
                eps = torch.randn(output_dim, device=device)
                sample = mu[b, t, mixture_idx] + sigma[b, t, mixture_idx] * eps
                
                trajectory[b, t] = sample
        
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
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, num_samples: int = 1, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            num_samples: 采样数量
            
        Returns:
            生成的轨迹 [num_samples, num_points, output_dim] 或 [num_points, output_dim]
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0).expand(num_samples, -1)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0).expand(num_samples, -1)
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectories = self.forward(start_tensor, end_tensor)
            
        self.max_seq_length = original_seq_length
        
        result = trajectories.numpy()
        return result.squeeze(0) if num_samples == 1 else result
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    start_poses: torch.Tensor, end_poses: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算MDN损失函数
        
        Args:
            predictions: 模型预测（实际上不使用，因为我们需要重新计算MDN参数）
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            start_poses: 起始位姿 [batch_size, input_dim]
            end_poses: 终止位姿 [batch_size, input_dim]
            
        Returns:
            负对数似然损失
        """
        # 获取MDN参数
        pi, mu, sigma = self.forward(start_poses, end_poses, return_parameters=True)
        
        # 计算负对数似然
        nll_loss = self._compute_nll_loss(targets, pi, mu, sigma)
        
        # 正则化项
        regularization_weight = self.config.get('regularization_weight', 0.01)
        if regularization_weight > 0:
            # L2正则化
            l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
            total_loss = nll_loss + regularization_weight * l2_reg
        else:
            total_loss = nll_loss
        
        return total_loss
    
    def _compute_nll_loss(self, targets: torch.Tensor, pi: torch.Tensor, 
                         mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算负对数似然损失
        
        Args:
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            pi: 混合权重 [batch_size, seq_length, num_mixtures]
            mu: 均值 [batch_size, seq_length, num_mixtures, output_dim]
            sigma: 标准差 [batch_size, seq_length, num_mixtures, output_dim]
            
        Returns:
            负对数似然损失
        """
        batch_size, seq_length, output_dim = targets.shape
        
        # 扩展目标维度以匹配混合成分
        targets_expanded = targets.unsqueeze(2).expand(-1, -1, self.num_mixtures, -1)
        
        # 计算每个混合成分的概率密度
        # p(x|μ,σ) = (2πσ²)^(-d/2) * exp(-||x-μ||²/(2σ²))
        diff = targets_expanded - mu  # [batch_size, seq_length, num_mixtures, output_dim]
        
        # 计算指数项
        exp_term = -0.5 * torch.sum((diff / sigma) ** 2, dim=-1)  # [batch_size, seq_length, num_mixtures]
        
        # 计算归一化常数
        log_norm = -0.5 * output_dim * torch.log(2 * torch.tensor(np.pi)) - torch.sum(torch.log(sigma), dim=-1)
        
        # 对数概率密度
        log_prob = log_norm + exp_term  # [batch_size, seq_length, num_mixtures]
        
        # 加权对数概率（log-sum-exp技巧）
        log_pi = torch.log(pi + 1e-8)
        weighted_log_prob = log_pi + log_prob
        
        # 计算混合分布的对数似然
        max_log_prob = torch.max(weighted_log_prob, dim=-1, keepdim=True)[0]
        log_likelihood = max_log_prob + torch.log(
            torch.sum(torch.exp(weighted_log_prob - max_log_prob), dim=-1, keepdim=True)
        )
        
        # 负对数似然
        nll = -torch.mean(log_likelihood)
        
        return nll
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'num_mixtures': self.num_mixtures,
            'use_lstm': self.use_lstm,
            'use_attention': self.use_attention,
            'model_category': 'Linear Architecture'
        })
        return info


class ConditionalMDNModel(MDNTrajectoryModel):
    """
    条件MDN模型
    支持基于任务条件的多模态轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.condition_dim = config.get('condition_dim', 64)
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.condition_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 条件融合层
        self.condition_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                return_parameters: bool = False) -> torch.Tensor:
        """
        条件前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition = self.condition_encoder(combined_pose)
        
        # 编码输入
        encoded_input = self.input_encoder(combined_pose)
        
        # 创建序列输入
        input_sequence = encoded_input.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        input_sequence = input_sequence + self.position_encoding.unsqueeze(0)
        
        # 序列建模
        if self.use_lstm:
            sequence_output, _ = self.sequence_model(input_sequence)
        else:
            sequence_output, _ = self.sequence_model(input_sequence)
        
        # 条件融合
        condition_expanded = condition.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        fused_features = torch.cat([sequence_output, condition_expanded], dim=-1)
        conditioned_output = self.condition_fusion(fused_features)
        
        # 注意力机制
        if self.use_attention:
            attended_output, _ = self.attention(conditioned_output, conditioned_output, conditioned_output)
            conditioned_output = self.attention_norm(conditioned_output + attended_output)
        
        # MDN输出
        mdn_output = self.mdn_layer(conditioned_output)
        
        # 解析MDN参数
        pi, mu, sigma = self._parse_mdn_parameters(mdn_output)
        
        if return_parameters:
            return pi, mu, sigma
        
        # 从MDN采样轨迹
        trajectory = self._sample_from_mdn(pi, mu, sigma)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory


class HierarchicalMDNModel(MDNTrajectoryModel):
    """
    分层MDN模型
    使用多层次的MDN进行粗到细的轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_levels = config.get('num_levels', 2)
        self.level_seq_lengths = config.get('level_seq_lengths', [10, 50])
        self.level_mixtures = config.get('level_mixtures', [3, 5])
        
        # 每个层次的MDN
        self.level_mdns = nn.ModuleList()
        
        for i, (seq_len, num_mix) in enumerate(zip(self.level_seq_lengths, self.level_mixtures)):
            # 序列模型
            if self.use_lstm:
                sequence_model = nn.LSTM(
                    input_size=self.hidden_dim if i == 0 else self.hidden_dim + self.output_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout if self.num_layers > 1 else 0,
                    batch_first=True
                )
            else:
                sequence_model = nn.GRU(
                    input_size=self.hidden_dim if i == 0 else self.hidden_dim + self.output_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout if self.num_layers > 1 else 0,
                    batch_first=True
                )
            
            # MDN层
            mdn_output_dim = num_mix * (2 * self.output_dim + 1)
            mdn_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, mdn_output_dim)
            )
            
            self.level_mdns.append(nn.ModuleDict({
                'sequence_model': sequence_model,
                'mdn_layer': mdn_layer,
                'num_mixtures': num_mix
            }))
    
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
        
        prev_trajectory = None
        
        for level, (seq_len, level_mdn) in enumerate(zip(self.level_seq_lengths, self.level_mdns)):
            # 创建当前层的输入
            if level == 0:
                level_input = encoded_input.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # 上采样前一层的轨迹
                upsampled_prev = self._upsample_trajectory(prev_trajectory, seq_len)
                level_input_base = encoded_input.unsqueeze(1).expand(-1, seq_len, -1)
                level_input = torch.cat([level_input_base, upsampled_prev], dim=-1)
            
            # 序列建模
            if self.use_lstm:
                sequence_output, _ = level_mdn['sequence_model'](level_input)
            else:
                sequence_output, _ = level_mdn['sequence_model'](level_input)
            
            # MDN输出
            mdn_output = level_mdn['mdn_layer'](sequence_output)
            
            # 解析参数（需要动态设置混合数量）
            original_num_mixtures = self.num_mixtures
            self.num_mixtures = level_mdn['num_mixtures']
            pi, mu, sigma = self._parse_mdn_parameters(mdn_output)
            self.num_mixtures = original_num_mixtures
            
            # 采样轨迹
            level_trajectory = self._sample_from_mdn(pi, mu, sigma)
            
            # 强制边界条件
            level_trajectory = self._enforce_boundary_conditions(level_trajectory, start_pose, end_pose)
            
            prev_trajectory = level_trajectory
        
        return prev_trajectory
    
    def _upsample_trajectory(self, trajectory: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        上采样轨迹到目标长度
        """
        batch_size, seq_length, output_dim = trajectory.shape
        device = trajectory.device
        
        old_indices = torch.linspace(0, seq_length - 1, seq_length, device=device)
        new_indices = torch.linspace(0, seq_length - 1, target_length, device=device)
        
        upsampled = torch.zeros(batch_size, target_length, output_dim, device=device)
        
        for b in range(batch_size):
            for d in range(output_dim):
                upsampled[b, :, d] = torch.interp(new_indices, old_indices, trajectory[b, :, d])
        
        return upsampled