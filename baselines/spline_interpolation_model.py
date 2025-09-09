"""
Spline Interpolation Model for Trajectory Generation
样条插值轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from scipy.interpolate import CubicSpline, BSpline, splrep, splev
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import ClassicalTrajectoryModel


class SplineInterpolationModel(ClassicalTrajectoryModel):
    """
    样条插值轨迹生成模型
    使用三次样条插值生成平滑轨迹
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.spline_type = config.get('spline_type', 'cubic')  # 'cubic', 'bspline', 'natural'
        self.smoothing_factor = config.get('smoothing_factor', 0.0)
        self.boundary_conditions = config.get('boundary_conditions', 'natural')
        self.via_points_ratio = config.get('via_points_ratio', 0.3)  # 中间点的比例
        
        # 可学习的参数：中间控制点
        self.num_control_points = config.get('num_control_points', 3)
        if self.num_control_points > 0:
            self.control_points = nn.Parameter(
                torch.randn(self.num_control_points, self.output_dim) * 0.1
            )
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 生成样条插值轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        trajectories = []
        
        for i in range(batch_size):
            # 为每个批次样本生成轨迹
            start = start_pose[i].detach().cpu().numpy()
            end = end_pose[i].detach().cpu().numpy()
            
            # 生成控制点
            control_points = self._generate_control_points(start, end, device)
            
            # 使用样条插值生成轨迹
            trajectory = self._interpolate_spline(start, end, control_points)
            
            trajectories.append(torch.from_numpy(trajectory).float().to(device))
        
        return torch.stack(trajectories, dim=0)
    
    def _generate_control_points(self, start: np.ndarray, end: np.ndarray, 
                               device: str) -> np.ndarray:
        """
        生成控制点
        
        Args:
            start: 起始位姿
            end: 终止位姿
            device: 设备
            
        Returns:
            控制点数组 [num_control_points, output_dim]
        """
        if self.num_control_points == 0:
            return np.array([])
        
        # 获取可学习的控制点偏移
        control_offset = self.control_points.detach().cpu().numpy()
        
        # 在起点和终点之间线性分布控制点
        t_values = np.linspace(0, 1, self.num_control_points + 2)[1:-1]
        
        control_points = []
        for i, t in enumerate(t_values):
            # 基础线性插值位置
            base_point = start + t * (end - start)
            
            # 添加可学习的偏移
            control_point = base_point + control_offset[i]
            control_points.append(control_point)
        
        return np.array(control_points)
    
    def _interpolate_spline(self, start: np.ndarray, end: np.ndarray,
                          control_points: np.ndarray) -> np.ndarray:
        """
        执行样条插值
        
        Args:
            start: 起始位姿
            end: 终止位姿
            control_points: 控制点
            
        Returns:
            插值轨迹 [seq_length, output_dim]
        """
        # 构建关键点序列
        if len(control_points) > 0:
            key_points = np.vstack([start.reshape(1, -1), control_points, end.reshape(1, -1)])
        else:
            key_points = np.vstack([start.reshape(1, -1), end.reshape(1, -1)])
        
        num_key_points = len(key_points)
        
        # 参数化时间
        t_key = np.linspace(0, 1, num_key_points)
        t_interp = np.linspace(0, 1, self.max_seq_length)
        
        # 对每个维度分别进行样条插值
        trajectory = np.zeros((self.max_seq_length, self.output_dim))
        
        for dim in range(self.output_dim):
            y_values = key_points[:, dim]
            
            if self.spline_type == 'cubic':
                # 三次样条插值
                if num_key_points >= 4:
                    cs = CubicSpline(t_key, y_values, bc_type=self.boundary_conditions)
                    trajectory[:, dim] = cs(t_interp)
                else:
                    # 如果点数不足，使用线性插值
                    trajectory[:, dim] = np.interp(t_interp, t_key, y_values)
                    
            elif self.spline_type == 'bspline':
                # B样条插值
                if num_key_points >= 4:
                    degree = min(3, num_key_points - 1)
                    tck = splrep(t_key, y_values, s=self.smoothing_factor, k=degree)
                    trajectory[:, dim] = splev(t_interp, tck)
                else:
                    trajectory[:, dim] = np.interp(t_interp, t_key, y_values)
                    
            else:  # 'natural' or default
                # 自然样条插值
                if num_key_points >= 3:
                    cs = CubicSpline(t_key, y_values, bc_type='natural')
                    trajectory[:, dim] = cs(t_interp)
                else:
                    trajectory[:, dim] = np.interp(t_interp, t_key, y_values)
        
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
        # 重建损失
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 平滑度损失
        smoothness_weight = self.config.get('smoothness_weight', 0.2)
        if smoothness_weight > 0:
            # 计算曲率（二阶导数）
            pred_curvature = torch.diff(predictions, n=2, dim=1)
            target_curvature = torch.diff(targets, n=2, dim=1)
            
            curvature_loss = nn.MSELoss()(pred_curvature, target_curvature)
            total_loss = mse_loss + smoothness_weight * curvature_loss
        else:
            total_loss = mse_loss
        
        # 控制点正则化
        control_reg_weight = self.config.get('control_regularization', 0.01)
        if control_reg_weight > 0 and hasattr(self, 'control_points'):
            control_reg = torch.mean(self.control_points ** 2)
            total_loss = total_loss + control_reg_weight * control_reg
            
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'spline_type': self.spline_type,
            'num_control_points': self.num_control_points,
            'smoothing_factor': self.smoothing_factor,
            'boundary_conditions': self.boundary_conditions,
            'model_category': 'Classical Methods'
        })
        return info


class AdaptiveSplineModel(SplineInterpolationModel):
    """
    自适应样条模型
    根据轨迹复杂度自动调整控制点数量
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adaptive_control = config.get('adaptive_control', True)
        self.min_control_points = config.get('min_control_points', 1)
        self.max_control_points = config.get('max_control_points', 5)
        
        # 复杂度评估网络
        if self.adaptive_control:
            self.complexity_net = nn.Sequential(
                nn.Linear(self.input_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def _estimate_complexity(self, start_pose: torch.Tensor, 
                           end_pose: torch.Tensor) -> torch.Tensor:
        """
        估计轨迹复杂度
        
        Args:
            start_pose: 起始位姿
            end_pose: 终止位姿
            
        Returns:
            复杂度分数 [batch_size, 1]
        """
        if not self.adaptive_control:
            return torch.ones(start_pose.shape[0], 1, device=start_pose.device) * 0.5
        
        # 连接起始和终止位姿
        combined_input = torch.cat([start_pose, end_pose], dim=-1)
        complexity = self.complexity_net(combined_input)
        
        return complexity
    
    def _adaptive_control_points(self, complexity: float) -> int:
        """
        根据复杂度确定控制点数量
        
        Args:
            complexity: 复杂度分数
            
        Returns:
            控制点数量
        """
        # 线性映射复杂度到控制点数量
        num_points = self.min_control_points + complexity * (
            self.max_control_points - self.min_control_points
        )
        
        return int(round(num_points))