"""
Normalizing Flows for Trajectory Generation
基于标准化流的轨迹生成方法

Normalizing Flows通过可逆变换将简单分布映射到复杂分布，
能够精确计算概率密度并生成高质量的轨迹样本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math

from .base_model import BaseTrajectoryModel


class CouplingLayer(nn.Module):
    """耦合层 - Real NVP的核心组件"""
    
    def __init__(self, input_dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.mask = mask
        
        # 尺度和平移网络
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向或反向变换
        
        Args:
            x: 输入张量
            reverse: 是否反向变换
            
        Returns:
            变换后的张量和对数雅可比行列式
        """
        if not reverse:
            # 前向变换: x -> z
            x_masked = x * self.mask
            s = self.scale_net(x_masked)
            t = self.translate_net(x_masked)
            
            z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
            log_det = torch.sum((1 - self.mask) * s, dim=-1)
            
            return z, log_det
        else:
            # 反向变换: z -> x
            z_masked = x * self.mask
            s = self.scale_net(z_masked)
            t = self.translate_net(z_masked)
            
            x = z_masked + (1 - self.mask) * (x - t) * torch.exp(-s)
            log_det = -torch.sum((1 - self.mask) * s, dim=-1)
            
            return x, log_det


class ActNorm(nn.Module):
    """激活标准化层"""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.scale = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.initialized = False
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        激活标准化变换
        
        Args:
            x: 输入张量 [batch_size, num_features]
            reverse: 是否反向变换
            
        Returns:
            变换后的张量和对数雅可比行列式
        """
        if not self.initialized and not reverse:
            # 数据驱动的初始化
            with torch.no_grad():
                mean = torch.mean(x, dim=0)
                std = torch.std(x, dim=0) + 1e-6
                self.bias.data = -mean
                self.scale.data = 1.0 / std
                self.initialized = True
        
        if not reverse:
            # 前向变换
            z = (x + self.bias) * self.scale
            log_det = torch.sum(torch.log(torch.abs(self.scale))) * torch.ones(x.size(0), device=x.device)
        else:
            # 反向变换
            z = x / self.scale - self.bias
            log_det = -torch.sum(torch.log(torch.abs(self.scale))) * torch.ones(x.size(0), device=x.device)
        
        return z, log_det


class NormalizingFlowsModel(BaseTrajectoryModel):
    """
    Normalizing Flows轨迹生成模型
    
    使用可逆神经网络将简单的高斯分布映射到复杂的轨迹分布
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 流模型参数
        self.num_flows = config.get('num_flows', 8)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.condition_dim = config.get('condition_dim', self.input_dim * 2)  # start + end
        
        # 轨迹维度 (展平的轨迹)
        self.trajectory_dim = self.max_seq_length * self.output_dim
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.trajectory_dim)
        )
        
        # 构建流模型
        self.flows = nn.ModuleList()
        
        for i in range(self.num_flows):
            # 激活标准化
            self.flows.append(ActNorm(self.trajectory_dim))
            
            # 耦合层
            mask = self._create_mask(self.trajectory_dim, i % 2)
            coupling = CouplingLayer(self.trajectory_dim, self.hidden_dim, mask)
            self.flows.append(coupling)
        
        # 基础分布 (标准高斯)
        self.register_buffer('base_mean', torch.zeros(self.trajectory_dim))
        self.register_buffer('base_std', torch.ones(self.trajectory_dim))
    
    def _create_mask(self, dim: int, parity: int) -> torch.Tensor:
        """创建耦合层的掩码"""
        mask = torch.zeros(dim)
        mask[parity::2] = 1
        return mask
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Normalizing Flows前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 上下文信息 (未使用)
            
        Returns:
            生成的轨迹 [batch_size, max_seq_length, output_dim]
        """
        batch_size = start_pose.size(0)
        
        # 编码条件信息
        condition = torch.cat([start_pose, end_pose], dim=1)
        condition_encoded = self.condition_encoder(condition)
        
        # 从基础分布采样
        z = torch.randn(batch_size, self.trajectory_dim, device=start_pose.device)
        
        # 添加条件信息
        z = z + condition_encoded
        
        # 通过流模型反向变换
        x, _ = self._inverse_transform(z)
        
        # 重塑为轨迹格式
        trajectory = x.view(batch_size, self.max_seq_length, self.output_dim)
        
        return trajectory
    
    def _forward_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向变换: x -> z
        
        Args:
            x: 数据样本 [batch_size, trajectory_dim]
            
        Returns:
            潜在变量和对数雅可比行列式
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for flow in self.flows:
            x, log_det = flow(x, reverse=False)
            log_det_total += log_det
        
        return x, log_det_total
    
    def _inverse_transform(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反向变换: z -> x
        
        Args:
            z: 潜在变量 [batch_size, trajectory_dim]
            
        Returns:
            数据样本和对数雅可比行列式
        """
        log_det_total = torch.zeros(z.size(0), device=z.device)
        
        for flow in reversed(self.flows):
            z, log_det = flow(z, reverse=True)
            log_det_total += log_det
        
        return z, log_det_total
    
    def log_prob(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        计算对数概率密度
        
        Args:
            x: 轨迹样本 [batch_size, max_seq_length, output_dim]
            condition: 条件信息 [batch_size, condition_dim]
            
        Returns:
            对数概率密度 [batch_size]
        """
        batch_size = x.size(0)
        
        # 展平轨迹
        x_flat = x.view(batch_size, -1)
        
        # 编码条件信息
        condition_encoded = self.condition_encoder(condition)
        
        # 前向变换
        z, log_det = self._forward_transform(x_flat)
        
        # 减去条件编码
        z = z - condition_encoded
        
        # 基础分布的对数概率
        log_prob_base = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * self.trajectory_dim * math.log(2 * math.pi)
        
        # 总对数概率
        log_prob_total = log_prob_base + log_det
        
        return log_prob_total
    
    def sample(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        条件采样
        
        Args:
            condition: 条件信息 [batch_size, condition_dim]
            num_samples: 采样数量
            
        Returns:
            采样的轨迹 [batch_size * num_samples, max_seq_length, output_dim]
        """
        batch_size = condition.size(0)
        
        # 编码条件信息
        condition_encoded = self.condition_encoder(condition)
        
        # 重复条件编码
        if num_samples > 1:
            condition_encoded = condition_encoded.repeat(num_samples, 1)
        
        # 从基础分布采样
        z = torch.randn(batch_size * num_samples, self.trajectory_dim, device=condition.device)
        
        # 添加条件信息
        z = z + condition_encoded
        
        # 反向变换
        x, _ = self._inverse_transform(z)
        
        # 重塑为轨迹格式
        trajectory = x.view(batch_size * num_samples, self.max_seq_length, self.output_dim)
        
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
        original_trajectory_dim = self.trajectory_dim
        
        self.max_seq_length = num_points
        self.trajectory_dim = num_points * self.output_dim
        
        try:
            with torch.no_grad():
                # 构建条件
                condition = np.concatenate([start_pose, end_pose])
                condition_tensor = torch.from_numpy(condition).float().unsqueeze(0).to(self.device)
                
                # 采样轨迹
                trajectory = self.sample(condition_tensor, num_samples=1)
                
                return trajectory.squeeze(0).cpu().numpy()
        
        finally:
            # 恢复原始参数
            self.max_seq_length = original_max_seq_length
            self.trajectory_dim = original_trajectory_dim
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    start_pose: torch.Tensor, end_pose: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算Normalizing Flows损失函数
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            损失值
        """
        # 构建条件
        condition = torch.cat([start_pose, end_pose], dim=1)
        
        # 负对数似然损失
        log_prob = self.log_prob(targets, condition)
        nll_loss = -torch.mean(log_prob)
        
        return nll_loss
    
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
        
        # 计算损失
        loss = self.compute_loss(None, trajectory, start_pose, end_pose)
        
        return {'loss': loss.item()}
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'model_type': 'Normalizing Flows',
            'num_flows': self.num_flows,
            'hidden_dim': self.hidden_dim,
            'condition_dim': self.condition_dim,
            'trajectory_dim': self.trajectory_dim
        })
        return info