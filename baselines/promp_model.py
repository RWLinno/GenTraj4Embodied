"""
Probabilistic Movement Primitives (ProMP) Model for Trajectory Generation
概率运动基元轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import ClassicalTrajectoryModel


class ProMPTrajectoryModel(ClassicalTrajectoryModel):
    """
    概率运动基元(ProMP)轨迹生成模型
    基于概率分布的轨迹表示，支持不确定性建模
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_basis = config.get('num_basis', 20)  # 基函数数量
        self.sigma_basis = config.get('sigma_basis', 0.05)  # 基函数宽度
        self.regularization = config.get('regularization', 1e-6)
        self.condition_via_points = config.get('condition_via_points', True)
        
        # 可学习的权重均值和协方差
        self.weight_mean = nn.Parameter(
            torch.zeros(self.output_dim, self.num_basis)
        )
        
        # 协方差矩阵的对数对角元素（确保正定性）
        self.log_weight_var = nn.Parameter(
            torch.ones(self.output_dim, self.num_basis) * (-2.0)
        )
        
        # 观测噪声
        self.log_obs_noise = nn.Parameter(
            torch.ones(self.output_dim) * (-4.0)
        )
        
        # 基函数中心
        self.register_buffer('basis_centers', self._compute_basis_centers())
        
    def _compute_basis_centers(self) -> torch.Tensor:
        """
        计算基函数中心
        
        Returns:
            基函数中心 [num_basis]
        """
        centers = torch.linspace(0, 1, self.num_basis)
        return centers
    
    def _compute_basis_functions(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算基函数值
        
        Args:
            t: 时间步 [seq_length] 或 [batch_size, seq_length]
            
        Returns:
            基函数值 [seq_length, num_basis] 或 [batch_size, seq_length, num_basis]
        """
        if t.dim() == 1:
            t = t.unsqueeze(0)  # [1, seq_length]
        
        # 扩展维度进行广播
        t_expanded = t.unsqueeze(-1)  # [batch_size, seq_length, 1]
        centers_expanded = self.basis_centers.unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis]
        
        # 高斯基函数
        psi = torch.exp(-0.5 * ((t_expanded - centers_expanded) / self.sigma_basis) ** 2)
        
        # 归一化
        psi_sum = torch.sum(psi, dim=-1, keepdim=True) + 1e-8
        psi_normalized = psi / psi_sum
        
        return psi_normalized.squeeze(0) if t.shape[0] == 1 else psi_normalized
    
    def _sample_weights(self, batch_size: int = 1) -> torch.Tensor:
        """
        从权重分布中采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            采样的权重 [batch_size, output_dim, num_basis]
        """
        device = self.weight_mean.device
        
        # 权重方差
        weight_var = torch.exp(self.log_weight_var)
        
        # 从正态分布采样
        eps = torch.randn(batch_size, self.output_dim, self.num_basis, device=device)
        weights = self.weight_mean.unsqueeze(0) + torch.sqrt(weight_var).unsqueeze(0) * eps
        
        return weights
    
    def _condition_on_boundary(self, weights: torch.Tensor, 
                             start_pose: torch.Tensor, 
                             end_pose: torch.Tensor) -> torch.Tensor:
        """
        在边界条件上进行条件化
        
        Args:
            weights: 权重 [batch_size, output_dim, num_basis]
            start_pose: 起始位姿 [batch_size, output_dim]
            end_pose: 终止位姿 [batch_size, output_dim]
            
        Returns:
            条件化后的权重 [batch_size, output_dim, num_basis]
        """
        if not self.condition_via_points:
            return weights
        
        batch_size = weights.shape[0]
        device = weights.device
        
        # 边界时间点
        t_start = torch.zeros(1, device=device)
        t_end = torch.ones(1, device=device)
        
        # 计算边界基函数
        psi_start = self._compute_basis_functions(t_start).squeeze(0)  # [num_basis]
        psi_end = self._compute_basis_functions(t_end).squeeze(0)      # [num_basis]
        
        # 构建观测矩阵
        H = torch.stack([psi_start, psi_end], dim=0)  # [2, num_basis]
        
        # 观测值
        y = torch.stack([start_pose, end_pose], dim=1)  # [batch_size, 2, output_dim]
        
        # 权重先验
        weight_var = torch.exp(self.log_weight_var)
        Sigma_w = torch.diag_embed(weight_var)  # [output_dim, num_basis, num_basis]
        
        # 观测噪声
        obs_var = torch.exp(self.log_obs_noise)
        Sigma_y = torch.diag_embed(obs_var.unsqueeze(0).expand(2, -1))  # [2, output_dim, output_dim]
        
        conditioned_weights = []
        
        for b in range(batch_size):
            batch_weights = []
            for d in range(self.output_dim):
                # 当前维度的参数
                mu_w = self.weight_mean[d]  # [num_basis]
                Sigma_w_d = Sigma_w[d]      # [num_basis, num_basis]
                y_d = y[b, :, d]            # [2]
                Sigma_y_d = Sigma_y[:, d, d]  # [2]
                
                # 贝叶斯条件化
                # Sigma_y_inv = diag(1/Sigma_y_d)
                Sigma_y_inv = torch.diag(1.0 / (Sigma_y_d + self.regularization))
                
                # 后验协方差
                Sigma_w_post_inv = torch.inverse(Sigma_w_d + self.regularization * torch.eye(self.num_basis, device=device)) + H.T @ Sigma_y_inv @ H
                Sigma_w_post = torch.inverse(Sigma_w_post_inv + self.regularization * torch.eye(self.num_basis, device=device))
                
                # 后验均值
                mu_w_post = Sigma_w_post @ (torch.inverse(Sigma_w_d + self.regularization * torch.eye(self.num_basis, device=device)) @ mu_w + H.T @ Sigma_y_inv @ y_d)
                
                batch_weights.append(mu_w_post)
            
            conditioned_weights.append(torch.stack(batch_weights, dim=0))
        
        return torch.stack(conditioned_weights, dim=0)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 生成ProMP轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 采样权重
        weights = self._sample_weights(batch_size)
        
        # 在边界条件上进行条件化
        weights = self._condition_on_boundary(weights, start_pose, end_pose)
        
        # 时间序列
        t = torch.linspace(0, 1, self.max_seq_length, device=device)
        
        # 计算基函数
        psi = self._compute_basis_functions(t)  # [seq_length, num_basis]
        
        # 生成轨迹
        trajectories = []
        for b in range(batch_size):
            # 轨迹 = 基函数 × 权重
            trajectory = torch.matmul(psi, weights[b].T)  # [seq_length, output_dim]
            trajectories.append(trajectory)
        
        return torch.stack(trajectories, dim=0)
    
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
        
        # 转换为tensor
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0).expand(num_samples, -1)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0).expand(num_samples, -1)
        
        # 临时设置序列长度
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectories = self.forward(start_tensor, end_tensor)
            
        # 恢复原始序列长度
        self.max_seq_length = original_seq_length
        
        result = trajectories.numpy()
        return result.squeeze(0) if num_samples == 1 else result
    
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
        # 负对数似然损失
        obs_var = torch.exp(self.log_obs_noise)
        
        # 重建损失（负对数似然）
        diff = predictions - targets
        nll_loss = 0.5 * torch.sum(diff ** 2 / obs_var.unsqueeze(0).unsqueeze(0)) + \
                  0.5 * predictions.numel() * torch.sum(self.log_obs_noise)
        
        # 权重先验损失
        weight_var = torch.exp(self.log_weight_var)
        prior_loss = 0.5 * torch.sum(self.weight_mean ** 2 / weight_var) + \
                    0.5 * torch.sum(self.log_weight_var)
        
        # KL散度正则化
        kl_weight = self.config.get('kl_weight', 0.01)
        total_loss = nll_loss + kl_weight * prior_loss
        
        return total_loss / predictions.shape[0]  # 平均到批次
    
    def compute_uncertainty(self, start_pose: torch.Tensor, 
                          end_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算轨迹不确定性
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            轨迹均值和方差 ([batch_size, seq_length, output_dim], [batch_size, seq_length, output_dim])
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 时间序列
        t = torch.linspace(0, 1, self.max_seq_length, device=device)
        psi = self._compute_basis_functions(t)  # [seq_length, num_basis]
        
        # 权重分布参数
        weight_var = torch.exp(self.log_weight_var)
        obs_var = torch.exp(self.log_obs_noise)
        
        # 计算轨迹均值
        mean_trajectories = []
        var_trajectories = []
        
        for b in range(batch_size):
            # 条件化权重均值
            conditioned_mean = self._condition_on_boundary(
                self.weight_mean.unsqueeze(0), 
                start_pose[b:b+1], 
                end_pose[b:b+1]
            ).squeeze(0)
            
            # 轨迹均值
            traj_mean = torch.matmul(psi, conditioned_mean.T)  # [seq_length, output_dim]
            
            # 轨迹方差 (简化计算)
            traj_var = torch.matmul(psi ** 2, weight_var.T) + obs_var.unsqueeze(0)
            
            mean_trajectories.append(traj_mean)
            var_trajectories.append(traj_var)
        
        return torch.stack(mean_trajectories), torch.stack(var_trajectories)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'num_basis': self.num_basis,
            'sigma_basis': self.sigma_basis,
            'condition_via_points': self.condition_via_points,
            'regularization': self.regularization,
            'model_category': 'Classical Methods'
        })
        return info


class HierarchicalProMPModel(ProMPTrajectoryModel):
    """
    分层ProMP模型
    支持多层次的运动基元表示
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_levels = config.get('num_levels', 2)
        self.level_basis = config.get('level_basis', [10, 20])
        
        # 多层权重
        self.level_weights = nn.ModuleList([
            nn.Parameter(torch.zeros(self.output_dim, nb))
            for nb in self.level_basis
        ])
        
        self.level_log_vars = nn.ModuleList([
            nn.Parameter(torch.ones(self.output_dim, nb) * (-2.0))
            for nb in self.level_basis
        ])
    
    def _compute_hierarchical_trajectory(self, start_pose: torch.Tensor, 
                                       end_pose: torch.Tensor) -> torch.Tensor:
        """
        计算分层轨迹
        """
        device = start_pose.device
        t = torch.linspace(0, 1, self.max_seq_length, device=device)
        
        trajectory = torch.zeros(self.max_seq_length, self.output_dim, device=device)
        
        for level, (weights, nb) in enumerate(zip(self.level_weights, self.level_basis)):
            # 计算当前层的基函数
            centers = torch.linspace(0, 1, nb, device=device)
            sigma = self.sigma_basis * (2 ** level)  # 不同层使用不同尺度
            
            t_expanded = t.unsqueeze(-1)
            centers_expanded = centers.unsqueeze(0)
            
            psi = torch.exp(-0.5 * ((t_expanded - centers_expanded) / sigma) ** 2)
            psi = psi / (torch.sum(psi, dim=-1, keepdim=True) + 1e-8)
            
            # 添加当前层的贡献
            trajectory += torch.matmul(psi, weights.T)
        
        return trajectory