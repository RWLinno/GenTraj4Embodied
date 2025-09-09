"""
Gaussian Mixture Model (GMM) for Trajectory Generation
高斯混合模型轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.mixture import GaussianMixture
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import ClassicalTrajectoryModel


class GMMTrajectoryModel(ClassicalTrajectoryModel):
    """
    高斯混合模型(GMM)轨迹生成模型
    使用GMM建模轨迹分布，支持多模态轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_components = config.get('num_components', 5)
        self.covariance_type = config.get('covariance_type', 'full')  # 'full', 'diag', 'tied', 'spherical'
        self.reg_covar = config.get('reg_covar', 1e-6)
        self.max_iter = config.get('max_iter', 100)
        self.tol = config.get('tol', 1e-3)
        
        # 可学习的GMM参数
        self.mixture_weights = nn.Parameter(
            torch.ones(self.num_components) / self.num_components
        )
        
        # 均值参数 [num_components, seq_length, output_dim]
        self.means = nn.Parameter(
            torch.randn(self.num_components, self.max_seq_length, self.output_dim) * 0.1
        )
        
        # 协方差参数（对数形式确保正定性）
        if self.covariance_type == 'full':
            # 完整协方差矩阵
            self.log_covars = nn.Parameter(
                torch.zeros(self.num_components, self.max_seq_length, self.output_dim, self.output_dim)
            )
        elif self.covariance_type == 'diag':
            # 对角协方差矩阵
            self.log_covars = nn.Parameter(
                torch.zeros(self.num_components, self.max_seq_length, self.output_dim)
            )
        elif self.covariance_type == 'tied':
            # 共享协方差矩阵
            self.log_covars = nn.Parameter(
                torch.zeros(self.max_seq_length, self.output_dim, self.output_dim)
            )
        else:  # spherical
            # 球形协方差矩阵
            self.log_covars = nn.Parameter(
                torch.zeros(self.num_components, self.max_seq_length)
            )
        
        # 条件化网络（用于根据起点终点调整参数）
        self.condition_net = nn.Sequential(
            nn.Linear(self.input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_components * 2),  # 权重调整 + 均值偏移
            nn.Tanh()
        )
    
    def _get_covariance_matrices(self) -> torch.Tensor:
        """
        获取协方差矩阵
        
        Returns:
            协方差矩阵 [num_components, seq_length, output_dim, output_dim]
        """
        if self.covariance_type == 'full':
            # 确保正定性：Σ = L @ L.T + εI
            L = torch.tril(self.log_covars)  # 下三角矩阵
            covars = torch.matmul(L, L.transpose(-2, -1))
            # 添加正则化项
            eye = torch.eye(self.output_dim, device=self.log_covars.device)
            covars = covars + self.reg_covar * eye.unsqueeze(0).unsqueeze(0)
            
        elif self.covariance_type == 'diag':
            # 对角协方差
            diag_vals = torch.exp(self.log_covars) + self.reg_covar
            covars = torch.diag_embed(diag_vals)
            
        elif self.covariance_type == 'tied':
            # 共享协方差
            L = torch.tril(self.log_covars)
            shared_covar = torch.matmul(L, L.transpose(-2, -1))
            eye = torch.eye(self.output_dim, device=self.log_covars.device)
            shared_covar = shared_covar + self.reg_covar * eye.unsqueeze(0)
            covars = shared_covar.unsqueeze(0).expand(self.num_components, -1, -1, -1)
            
        else:  # spherical
            # 球形协方差
            var_vals = torch.exp(self.log_covars) + self.reg_covar
            eye = torch.eye(self.output_dim, device=self.log_covars.device)
            covars = var_vals.unsqueeze(-1).unsqueeze(-1) * eye.unsqueeze(0).unsqueeze(0)
        
        return covars
    
    def _condition_parameters(self, start_pose: torch.Tensor, 
                            end_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据起点终点条件化GMM参数
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            条件化的权重和均值
        """
        batch_size = start_pose.shape[0]
        
        # 连接起始和终止位姿
        combined_input = torch.cat([start_pose, end_pose], dim=-1)
        
        # 通过网络获取调整参数
        adjustments = self.condition_net(combined_input)  # [batch_size, num_components * 2]
        
        # 分离权重调整和均值偏移
        weight_adj = adjustments[:, :self.num_components]  # [batch_size, num_components]
        mean_shift = adjustments[:, self.num_components:]   # [batch_size, num_components]
        
        # 调整混合权重
        adjusted_weights = torch.softmax(
            torch.log(self.mixture_weights + 1e-8).unsqueeze(0) + weight_adj, 
            dim=-1
        )
        
        # 调整均值（线性插值约束）
        t = torch.linspace(0, 1, self.max_seq_length, device=start_pose.device)
        t = t.unsqueeze(0).unsqueeze(-1)  # [1, seq_length, 1]
        
        adjusted_means = []
        for b in range(batch_size):
            # 基础线性插值
            base_trajectory = start_pose[b].unsqueeze(0) + t * (
                end_pose[b].unsqueeze(0) - start_pose[b].unsqueeze(0)
            )  # [1, seq_length, output_dim]
            
            # 为每个组件添加偏移
            batch_means = []
            for k in range(self.num_components):
                # 基础轨迹 + 可学习偏移 + 条件偏移
                component_mean = (
                    base_trajectory.squeeze(0) + 
                    self.means[k] + 
                    mean_shift[b, k] * torch.randn_like(self.means[k]) * 0.1
                )
                batch_means.append(component_mean)
            
            adjusted_means.append(torch.stack(batch_means, dim=0))
        
        return adjusted_weights, torch.stack(adjusted_means, dim=0)
    
    def _sample_from_gmm(self, weights: torch.Tensor, means: torch.Tensor, 
                        covars: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        从GMM中采样轨迹
        
        Args:
            weights: 混合权重 [num_components]
            means: 均值 [num_components, seq_length, output_dim]
            covars: 协方差 [num_components, seq_length, output_dim, output_dim]
            num_samples: 采样数量
            
        Returns:
            采样的轨迹 [num_samples, seq_length, output_dim]
        """
        device = weights.device
        samples = []
        
        for _ in range(num_samples):
            # 选择组件
            component_idx = torch.multinomial(weights, 1).item()
            
            # 从选择的组件采样
            component_mean = means[component_idx]  # [seq_length, output_dim]
            component_covar = covars[component_idx]  # [seq_length, output_dim, output_dim]
            
            # 为每个时间步采样
            trajectory = []
            for t in range(self.max_seq_length):
                # 多元正态分布采样
                mean_t = component_mean[t]
                covar_t = component_covar[t]
                
                # 使用Cholesky分解采样
                try:
                    L = torch.linalg.cholesky(covar_t)
                    eps = torch.randn(self.output_dim, device=device)
                    sample_t = mean_t + L @ eps
                except:
                    # 如果Cholesky分解失败，使用对角近似
                    eps = torch.randn(self.output_dim, device=device)
                    std_t = torch.sqrt(torch.diag(covar_t))
                    sample_t = mean_t + std_t * eps
                
                trajectory.append(sample_t)
            
            samples.append(torch.stack(trajectory, dim=0))
        
        return torch.stack(samples, dim=0)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 生成GMM轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        
        # 获取协方差矩阵
        covars = self._get_covariance_matrices()
        
        # 条件化参数
        weights, means = self._condition_parameters(start_pose, end_pose)
        
        # 为每个批次样本生成轨迹
        trajectories = []
        for b in range(batch_size):
            trajectory = self._sample_from_gmm(
                weights[b], means[b], covars, num_samples=1
            ).squeeze(0)
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
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        # 临时设置序列长度
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            # 获取参数
            covars = self._get_covariance_matrices()
            weights, means = self._condition_parameters(start_tensor, end_tensor)
            
            # 生成多个样本
            trajectories = self._sample_from_gmm(
                weights[0], means[0], covars, num_samples=num_samples
            )
            
        # 恢复原始序列长度
        self.max_seq_length = original_seq_length
        
        result = trajectories.numpy()
        return result.squeeze(0) if num_samples == 1 else result
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    start_poses: Optional[torch.Tensor] = None,
                    end_poses: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        计算损失函数 - GMM负对数似然
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            start_poses: 起始位姿 [batch_size, input_dim]
            end_poses: 终止位姿 [batch_size, input_dim]
            
        Returns:
            损失值
        """
        batch_size = targets.shape[0]
        
        if start_poses is None or end_poses is None:
            # 如果没有提供边界条件，使用轨迹的首末点
            start_poses = targets[:, 0, :]
            end_poses = targets[:, -1, :]
        
        # 获取参数
        covars = self._get_covariance_matrices()
        weights, means = self._condition_parameters(start_poses, end_poses)
        
        # 计算负对数似然
        total_nll = 0.0
        
        for b in range(batch_size):
            trajectory = targets[b]  # [seq_length, output_dim]
            
            # 计算每个组件的似然
            component_likelihoods = []
            
            for k in range(self.num_components):
                # 计算多元正态分布的对数似然
                diff = trajectory - means[b, k]  # [seq_length, output_dim]
                
                log_likelihood = 0.0
                for t in range(self.max_seq_length):
                    # 每个时间步的似然
                    covar_t = covars[k, t]
                    diff_t = diff[t].unsqueeze(0)  # [1, output_dim]
                    
                    # 计算log p(x|μ,Σ) = -0.5 * (x-μ)^T Σ^-1 (x-μ) - 0.5 * log|Σ| - 0.5 * d * log(2π)
                    try:
                        covar_inv = torch.inverse(covar_t + self.reg_covar * torch.eye(self.output_dim, device=covar_t.device))
                        quad_form = torch.matmul(torch.matmul(diff_t, covar_inv), diff_t.T)
                        log_det = torch.logdet(covar_t + self.reg_covar * torch.eye(self.output_dim, device=covar_t.device))
                        
                        log_likelihood += -0.5 * (quad_form + log_det + self.output_dim * np.log(2 * np.pi))
                    except:
                        # 数值稳定性处理
                        log_likelihood += -0.5 * torch.sum(diff_t ** 2) / self.reg_covar
                
                component_likelihoods.append(log_likelihood)
            
            # 组合所有组件的似然（log-sum-exp技巧）
            component_likelihoods = torch.stack(component_likelihoods)
            log_weights = torch.log(weights[b] + 1e-8)
            
            # log-sum-exp
            max_ll = torch.max(component_likelihoods)
            log_mixture_likelihood = max_ll + torch.log(
                torch.sum(torch.exp(component_likelihoods - max_ll + log_weights))
            )
            
            total_nll += -log_mixture_likelihood
        
        return total_nll / batch_size
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'num_components': self.num_components,
            'covariance_type': self.covariance_type,
            'reg_covar': self.reg_covar,
            'max_iter': self.max_iter,
            'model_category': 'Classical Methods'
        })
        return info


class AdaptiveGMMModel(GMMTrajectoryModel):
    """
    自适应GMM模型
    根据数据复杂度自动调整组件数量
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_components = config.get('min_components', 2)
        self.max_components = config.get('max_components', 10)
        
        # 组件数量选择网络
        self.component_selector = nn.Sequential(
            nn.Linear(self.input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _select_num_components(self, start_pose: torch.Tensor, 
                             end_pose: torch.Tensor) -> int:
        """
        自适应选择组件数量
        
        Args:
            start_pose: 起始位姿
            end_pose: 终止位姿
            
        Returns:
            选择的组件数量
        """
        combined_input = torch.cat([start_pose, end_pose], dim=-1)
        complexity = self.component_selector(combined_input).item()
        
        # 线性映射到组件数量范围
        num_components = int(
            self.min_components + complexity * (self.max_components - self.min_components)
        )
        
        return max(self.min_components, min(self.max_components, num_components))