"""
Linear Interpolation Model for Trajectory Generation
线性插值轨迹生成模型
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


class LinearInterpolationModel(ClassicalTrajectoryModel):
    """
    线性插值轨迹生成模型
    最简单的轨迹生成方法，在起点和终点之间进行线性插值
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interpolation_type = config.get('interpolation_type', 'linear')
        self.add_noise = config.get('add_noise', False)
        self.noise_std = config.get('noise_std', 0.01)
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 生成线性插值轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 创建时间步长
        t = torch.linspace(0, 1, self.max_seq_length, device=device)
        t = t.unsqueeze(0).unsqueeze(-1)  # [1, seq_length, 1]
        
        # 线性插值
        start_expanded = start_pose.unsqueeze(1)  # [batch_size, 1, input_dim]
        end_expanded = end_pose.unsqueeze(1)      # [batch_size, 1, input_dim]
        
        trajectory = start_expanded + t * (end_expanded - start_expanded)
        
        # 添加噪声（如果启用）
        if self.add_noise and self.training:
            noise = torch.randn_like(trajectory) * self.noise_std
            trajectory = trajectory + noise
            
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
        计算损失函数 - 对于线性插值，主要是重建损失
        
        Args:
            predictions: 模型预测 [batch_size, seq_length, output_dim]
            targets: 目标轨迹 [batch_size, seq_length, output_dim]
            
        Returns:
            损失值
        """
        # 均方误差损失
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 平滑度损失（可选）
        smoothness_weight = self.config.get('smoothness_weight', 0.1)
        if smoothness_weight > 0:
            # 计算二阶差分（加速度）
            pred_acc = torch.diff(predictions, n=2, dim=1)
            target_acc = torch.diff(targets, n=2, dim=1)
            smoothness_loss = nn.MSELoss()(pred_acc, target_acc)
            
            total_loss = mse_loss + smoothness_weight * smoothness_loss
        else:
            total_loss = mse_loss
            
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'interpolation_type': self.interpolation_type,
            'add_noise': self.add_noise,
            'noise_std': self.noise_std,
            'model_category': 'Classical Methods'
        })
        return info


class WeightedLinearInterpolationModel(LinearInterpolationModel):
    """
    加权线性插值模型
    支持非均匀时间步长的插值
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.time_weights = config.get('time_weights', None)
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        加权线性插值前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 使用自定义时间权重或默认线性权重
        if self.time_weights is not None:
            t = torch.tensor(self.time_weights, device=device, dtype=torch.float32)
        else:
            # 使用非线性时间分布（例如，开始和结束时更密集）
            t = torch.linspace(0, 1, self.max_seq_length, device=device)
            # 应用sigmoid变换使中间部分更稀疏
            t = torch.sigmoid(6 * (t - 0.5)) * 0.8 + 0.1
            t = (t - t.min()) / (t.max() - t.min())  # 重新归一化到[0,1]
            
        t = t.unsqueeze(0).unsqueeze(-1)  # [1, seq_length, 1]
        
        # 加权插值
        start_expanded = start_pose.unsqueeze(1)
        end_expanded = end_pose.unsqueeze(1)
        
        trajectory = start_expanded + t * (end_expanded - start_expanded)
        
        # 添加噪声（如果启用）
        if self.add_noise and self.training:
            noise = torch.randn_like(trajectory) * self.noise_std
            trajectory = trajectory + noise
            
        return trajectory