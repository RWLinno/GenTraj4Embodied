"""
Dynamic Movement Primitives (DMP) Model for Trajectory Generation
动态运动基元轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import ClassicalTrajectoryModel


class DMPTrajectoryModel(ClassicalTrajectoryModel):
    """
    动态运动基元(DMP)轨迹生成模型
    基于经典DMP理论，使用可学习的基函数权重
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_basis = config.get('num_basis', 25)  # 基函数数量
        self.alpha_y = config.get('alpha_y', 25.0)    # 临界阻尼系数
        self.beta_y = config.get('beta_y', 6.25)      # 弹簧常数
        self.alpha_x = config.get('alpha_x', 1.0)     # 相位衰减系数
        self.dt = config.get('dt', 0.02)              # 时间步长
        self.tau = config.get('tau', 1.0)             # 时间缩放因子
        
        # 可学习的基函数权重
        self.weights = nn.Parameter(
            torch.randn(self.output_dim, self.num_basis) * 0.1
        )
        
        # 基函数中心和宽度
        self.register_buffer('centers', self._compute_centers())
        self.register_buffer('widths', self._compute_widths())
        
    def _compute_centers(self) -> torch.Tensor:
        """
        计算基函数中心
        
        Returns:
            基函数中心 [num_basis]
        """
        # 在相位空间中均匀分布
        centers = torch.exp(-self.alpha_x * torch.linspace(0, 1, self.num_basis))
        return centers
    
    def _compute_widths(self) -> torch.Tensor:
        """
        计算基函数宽度
        
        Returns:
            基函数宽度 [num_basis]
        """
        # 基于相邻中心的距离
        if self.num_basis > 1:
            diff = torch.diff(self.centers)
            widths = torch.zeros_like(self.centers)
            widths[:-1] = diff
            widths[-1] = diff[-1]
            widths = 1.0 / (widths ** 2)
        else:
            widths = torch.ones(self.num_basis)
        
        return widths
    
    def _compute_phase(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算相位变量
        
        Args:
            t: 时间步 [seq_length]
            
        Returns:
            相位变量 [seq_length]
        """
        x = torch.exp(-self.alpha_x * t / self.tau)
        return x
    
    def _compute_basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算基函数值
        
        Args:
            x: 相位变量 [seq_length]
            
        Returns:
            基函数值 [seq_length, num_basis]
        """
        # 高斯基函数
        x_expanded = x.unsqueeze(-1)  # [seq_length, 1]
        centers_expanded = self.centers.unsqueeze(0)  # [1, num_basis]
        widths_expanded = self.widths.unsqueeze(0)    # [1, num_basis]
        
        psi = torch.exp(-widths_expanded * (x_expanded - centers_expanded) ** 2)
        
        # 归一化
        psi_sum = torch.sum(psi, dim=-1, keepdim=True) + 1e-8
        psi_normalized = psi / psi_sum
        
        return psi_normalized
    
    def _integrate_dmp(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
        """
        积分DMP方程生成轨迹
        
        Args:
            start_pose: 起始位姿 [output_dim]
            end_pose: 终止位姿 [output_dim]
            weights: 基函数权重 [output_dim, num_basis]
            
        Returns:
            生成的轨迹 [seq_length, output_dim]
        """
        device = start_pose.device
        
        # 时间序列
        t = torch.linspace(0, self.tau, self.max_seq_length, device=device)
        
        # 相位变量
        x = self._compute_phase(t)
        
        # 基函数
        psi = self._compute_basis_functions(x)  # [seq_length, num_basis]
        
        # 强迫项
        f = torch.matmul(psi, weights.T) * x.unsqueeze(-1)  # [seq_length, output_dim]
        
        # 初始化状态变量
        y = torch.zeros(self.max_seq_length, self.output_dim, device=device)
        dy = torch.zeros(self.max_seq_length, self.output_dim, device=device)
        
        # 设置初始条件
        y[0] = start_pose
        dy[0] = torch.zeros_like(start_pose)
        
        # 数值积分
        for i in range(1, self.max_seq_length):
            # DMP方程: tau * ddy = alpha_y * (beta_y * (g - y) - dy) + f
            ddy = (self.alpha_y * (self.beta_y * (end_pose - y[i-1]) - dy[i-1]) + f[i-1]) / self.tau
            
            # 欧拉积分
            dy[i] = dy[i-1] + ddy * self.dt
            y[i] = y[i-1] + dy[i] * self.dt
        
        return y
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 生成DMP轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        trajectories = []
        
        for i in range(batch_size):
            # 为每个批次样本生成轨迹
            trajectory = self._integrate_dmp(
                start_pose[i], 
                end_pose[i], 
                self.weights
            )
            trajectories.append(trajectory)
        
        return torch.stack(trajectories, dim=0)
    
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
        start_tensor = torch.from_numpy(start_pose).float()
        end_tensor = torch.from_numpy(end_pose).float()
        
        # 临时设置序列长度
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self._integrate_dmp(start_tensor, end_tensor, self.weights)
            
        # 恢复原始序列长度
        self.max_seq_length = original_seq_length
        
        return trajectory.numpy()
    
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
        # 轨迹重建损失
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 动力学一致性损失
        dynamics_weight = self.config.get('dynamics_weight', 0.1)
        if dynamics_weight > 0:
            # 计算预测轨迹的加速度
            pred_vel = torch.diff(predictions, dim=1)
            pred_acc = torch.diff(pred_vel, dim=1)
            
            # 计算目标轨迹的加速度
            target_vel = torch.diff(targets, dim=1)
            target_acc = torch.diff(target_vel, dim=1)
            
            dynamics_loss = nn.MSELoss()(pred_acc, target_acc)
            total_loss = mse_loss + dynamics_weight * dynamics_loss
        else:
            total_loss = mse_loss
        
        # 权重正则化
        weight_reg = self.config.get('weight_regularization', 0.01)
        if weight_reg > 0:
            reg_loss = torch.mean(self.weights ** 2)
            total_loss = total_loss + weight_reg * reg_loss
            
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'num_basis': self.num_basis,
            'alpha_y': self.alpha_y,
            'beta_y': self.beta_y,
            'alpha_x': self.alpha_x,
            'tau': self.tau,
            'dt': self.dt,
            'model_category': 'Classical Methods'
        })
        return info


class AdaptiveDMPModel(DMPTrajectoryModel):
    """
    自适应DMP模型
    根据任务复杂度自动调整参数
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 自适应参数网络
        self.adaptive_params = config.get('adaptive_params', True)
        if self.adaptive_params:
            self.param_net = nn.Sequential(
                nn.Linear(self.input_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # alpha_y, beta_y, tau
                nn.Softplus()
            )
    
    def _get_adaptive_params(self, start_pose: torch.Tensor, 
                           end_pose: torch.Tensor) -> Dict[str, float]:
        """
        获取自适应参数
        
        Args:
            start_pose: 起始位姿
            end_pose: 终止位姿
            
        Returns:
            自适应参数字典
        """
        if not self.adaptive_params:
            return {
                'alpha_y': self.alpha_y,
                'beta_y': self.beta_y,
                'tau': self.tau
            }
        
        # 连接起始和终止位姿
        combined_input = torch.cat([start_pose, end_pose], dim=-1)
        params = self.param_net(combined_input)
        
        return {
            'alpha_y': params[0].item() * 10 + 15,  # 范围 [15, 25]
            'beta_y': params[1].item() * 5 + 4,     # 范围 [4, 9]
            'tau': params[2].item() * 2 + 0.5       # 范围 [0.5, 2.5]
        }


class MultiModalDMPModel(DMPTrajectoryModel):
    """
    多模态DMP模型
    支持生成多种可能的轨迹
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_modes = config.get('num_modes', 3)
        
        # 每个模态的权重
        self.mode_weights = nn.Parameter(
            torch.randn(self.num_modes, self.output_dim, self.num_basis) * 0.1
        )
        
        # 模态选择网络
        self.mode_selector = nn.Sequential(
            nn.Linear(self.input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_modes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        多模态前向传播
        """
        batch_size = start_pose.shape[0]
        
        # 计算模态权重
        combined_input = torch.cat([start_pose, end_pose], dim=-1)
        mode_probs = self.mode_selector(combined_input)  # [batch_size, num_modes]
        
        trajectories = []
        
        for i in range(batch_size):
            # 加权组合多个模态的权重
            weighted_weights = torch.sum(
                mode_probs[i].unsqueeze(-1).unsqueeze(-1) * self.mode_weights,
                dim=0
            )
            
            # 生成轨迹
            trajectory = self._integrate_dmp(
                start_pose[i], 
                end_pose[i], 
                weighted_weights
            )
            trajectories.append(trajectory)
        
        return torch.stack(trajectories, dim=0)