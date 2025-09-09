"""
Conditional Normalizing Flow for Trajectory Generation
条件标准化流轨迹生成模型

基于标准化流的条件生成模型，能够学习复杂的条件概率分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import math

from .base_model import BaseTrajectoryModel


class CouplingLayer(nn.Module):
    """耦合层 - Real NVP风格的可逆变换"""
    
    def __init__(self, dim: int, hidden_dim: int, condition_dim: int, mask: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.mask = mask
        
        # 变换网络
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2 + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh()  # 限制尺度变化
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(dim // 2 + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向或反向变换
        
        Args:
            x: 输入张量 [batch_size, dim]
            condition: 条件张量 [batch_size, condition_dim]
            reverse: 是否反向变换
            
        Returns:
            y: 变换后的张量
            log_det: 对数行列式
        """
        if not reverse:
            return self._forward(x, condition)
        else:
            return self._inverse(x, condition)
    
    def _forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向变换"""
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)
        
        # 拼接masked部分和条件
        net_input = torch.cat([x_masked, condition], dim=1)
        
        # 计算尺度和平移
        s = self.scale_net(net_input)
        t = self.translate_net(net_input)
        
        # 应用变换
        y_unmasked = x_unmasked * torch.exp(s) + t
        y = x_masked + y_unmasked * (1 - self.mask)
        
        # 计算对数行列式
        log_det = torch.sum(s * (1 - self.mask), dim=1)
        
        return y, log_det
    
    def _inverse(self, y: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """反向变换"""
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)
        
        # 拼接masked部分和条件
        net_input = torch.cat([y_masked, condition], dim=1)
        
        # 计算尺度和平移
        s = self.scale_net(net_input)
        t = self.translate_net(net_input)
        
        # 应用反向变换
        x_unmasked = (y_unmasked - t) * torch.exp(-s)
        x = y_masked + x_unmasked * (1 - self.mask)
        
        # 计算对数行列式
        log_det = -torch.sum(s * (1 - self.mask), dim=1)
        
        return x, log_det


class ConditionalFlowModel(BaseTrajectoryModel):
    """条件标准化流模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型配置
        self.num_flows = config['architecture']['num_flows']
        self.hidden_dim = config['architecture']['hidden_dim']
        self.num_layers = config['architecture']['num_layers']
        self.condition_dim = config['architecture']['condition_dim']  # start_pose + end_pose = 14
        self.flow_type = config['architecture'].get('flow_type', 'coupling')
        
        # 数据维度
        self.trajectory_length = config.get('trajectory_length', 50)
        self.pose_dim = 7  # 3D position + quaternion
        self.trajectory_dim = self.trajectory_length * self.pose_dim
        
        # 构建网络
        self._build_networks()
        
        # 基础分布（标准正态分布）
        self.register_buffer('base_mean', torch.zeros(self.trajectory_dim))
        self.register_buffer('base_std', torch.ones(self.trajectory_dim))
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """构建流网络"""
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.condition_dim)
        )
        
        # 构建耦合层
        self.flows = nn.ModuleList()
        
        for i in range(self.num_flows):
            # 交替mask模式
            if i % 2 == 0:
                mask = torch.zeros(self.trajectory_dim)
                mask[:self.trajectory_dim // 2] = 1
            else:
                mask = torch.zeros(self.trajectory_dim)
                mask[self.trajectory_dim // 2:] = 1
            
            coupling_layer = CouplingLayer(
                dim=self.trajectory_dim,
                hidden_dim=self.hidden_dim,
                condition_dim=self.condition_dim,
                mask=mask
            )
            
            self.flows.append(coupling_layer)
            
            # 注册mask为buffer
            self.register_buffer(f'mask_{i}', mask)
    
    def forward(self, trajectory: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播 - 从数据空间到潜在空间
        
        Args:
            trajectory: [batch_size, trajectory_length, pose_dim]
            condition: [batch_size, condition_dim]
            
        Returns:
            包含z和log_prob的字典
        """
        batch_size = trajectory.shape[0]
        
        # 展平轨迹
        x = trajectory.view(batch_size, -1)
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 通过流变换
        z = x
        log_det_sum = torch.zeros(batch_size).to(x.device)
        
        for flow in self.flows:
            z, log_det = flow(z, condition_encoded, reverse=False)
            log_det_sum += log_det
        
        # 计算对数概率
        log_prob_base = self._base_log_prob(z)
        log_prob = log_prob_base + log_det_sum
        
        return {
            'z': z,
            'log_prob': log_prob,
            'log_det': log_det_sum
        }
    
    def inverse(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """反向传播 - 从潜在空间到数据空间
        
        Args:
            z: [batch_size, trajectory_dim]
            condition: [batch_size, condition_dim]
            
        Returns:
            trajectory: [batch_size, trajectory_length, pose_dim]
        """
        batch_size = z.shape[0]
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 反向通过流变换
        x = z
        for flow in reversed(self.flows):
            x, _ = flow(x, condition_encoded, reverse=True)
        
        # 重塑为轨迹格式
        trajectory = x.view(batch_size, self.trajectory_length, self.pose_dim)
        
        return trajectory
    
    def _base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """计算基础分布的对数概率"""
        log_prob = -0.5 * torch.sum((z - self.base_mean) ** 2 / (self.base_std ** 2), dim=1)
        log_prob -= 0.5 * self.trajectory_dim * math.log(2 * math.pi)
        log_prob -= torch.sum(torch.log(self.base_std))
        return log_prob
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失函数"""
        trajectory = batch['trajectory']
        condition = batch['condition']
        
        # 前向传播
        outputs = self.forward(trajectory, condition)
        
        # 负对数似然损失
        nll_loss = -torch.mean(outputs['log_prob'])
        
        return {
            'total_loss': nll_loss,
            'nll_loss': nll_loss,
            'mean_log_prob': torch.mean(outputs['log_prob'])
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        self.train()
        
        # 计算损失
        losses = self.compute_loss(batch)
        
        # 返回标量损失值
        return {
            'loss': losses['total_loss'].item(),
            'nll_loss': losses['nll_loss'].item(),
            'mean_log_prob': losses['mean_log_prob'].item()
        }
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray, 
                          num_samples: int = 1, **kwargs) -> np.ndarray:
        """生成轨迹
        
        Args:
            start_pose: 起始位姿 [7] (x, y, z, qx, qy, qz, qw)
            end_pose: 终止位姿 [7]
            num_samples: 生成样本数量
            
        Returns:
            trajectories: [num_samples, trajectory_length, 7]
        """
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 准备条件
            condition = np.concatenate([start_pose, end_pose])  # [14]
            condition = torch.FloatTensor(condition).unsqueeze(0).repeat(num_samples, 1)
            condition = condition.to(device)
            
            # 从基础分布采样
            z = torch.randn(num_samples, self.trajectory_dim).to(device)
            
            # 通过流生成轨迹
            trajectories = self.inverse(z, condition)
            
            # 强制执行边界约束
            trajectories = self._enforce_boundary_constraints(
                trajectories, start_pose, end_pose
            )
            
            return trajectories.cpu().numpy()
    
    def _enforce_boundary_constraints(self, trajectories: torch.Tensor, 
                                    start_pose: np.ndarray, end_pose: np.ndarray) -> torch.Tensor:
        """强制执行边界约束"""
        # 设置起点和终点
        trajectories[:, 0] = torch.FloatTensor(start_pose).to(trajectories.device)
        trajectories[:, -1] = torch.FloatTensor(end_pose).to(trajectories.device)
        
        # 四元数归一化
        for t in range(self.trajectory_length):
            quat = trajectories[:, t, 3:7]
            quat_norm = torch.norm(quat, dim=1, keepdim=True)
            trajectories[:, t, 3:7] = quat / (quat_norm + 1e-8)
        
        return trajectories
    
    def compute_log_likelihood(self, trajectory: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """计算给定轨迹的对数似然"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(trajectory, condition)
            return outputs['log_prob']
    
    def interpolate_conditions(self, start_pose1: np.ndarray, end_pose1: np.ndarray,
                             start_pose2: np.ndarray, end_pose2: np.ndarray,
                             num_interpolations: int = 10) -> np.ndarray:
        """在条件空间中插值生成轨迹"""
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 准备两个条件
            condition1 = np.concatenate([start_pose1, end_pose1])
            condition2 = np.concatenate([start_pose2, end_pose2])
            
            # 在条件空间中插值
            alphas = np.linspace(0, 1, num_interpolations)
            trajectories = []
            
            for alpha in alphas:
                condition_interp = (1 - alpha) * condition1 + alpha * condition2
                condition_tensor = torch.FloatTensor(condition_interp).unsqueeze(0).to(device)
                
                # 从基础分布采样
                z = torch.randn(1, self.trajectory_dim).to(device)
                
                # 生成轨迹
                traj = self.inverse(z, condition_tensor)
                
                # 强制边界约束
                start_interp = (1 - alpha) * start_pose1 + alpha * start_pose2
                end_interp = (1 - alpha) * end_pose1 + alpha * end_pose2
                traj = self._enforce_boundary_constraints(traj, start_interp, end_interp)
                
                trajectories.append(traj.squeeze(0))
            
            return torch.stack(trajectories).cpu().numpy()
    
    def sample_diverse_trajectories(self, start_pose: np.ndarray, end_pose: np.ndarray,
                                  num_samples: int = 10, temperature: float = 1.0) -> np.ndarray:
        """生成多样化的轨迹样本"""
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 准备条件
            condition = np.concatenate([start_pose, end_pose])
            condition = torch.FloatTensor(condition).unsqueeze(0).repeat(num_samples, 1)
            condition = condition.to(device)
            
            # 使用不同温度采样
            z = torch.randn(num_samples, self.trajectory_dim).to(device) * temperature
            
            # 生成轨迹
            trajectories = self.inverse(z, condition)
            
            # 强制边界约束
            trajectories = self._enforce_boundary_constraints(
                trajectories, start_pose, end_pose
            )
            
            return trajectories.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ConditionalFlow',
            'model_type': 'Probabilistic Generative Models',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_flows': self.num_flows,
            'hidden_dim': self.hidden_dim,
            'condition_dim': self.condition_dim,
            'flow_type': self.flow_type,
            'trajectory_length': self.trajectory_length,
            'pose_dim': self.pose_dim,
            'trajectory_dim': self.trajectory_dim,
            'supports_conditional_generation': True,
            'supports_likelihood_computation': True,
            'supports_condition_interpolation': True
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"模型已从 {filepath} 加载")
        return checkpoint.get('model_info', {})


# 工厂函数
def create_conditional_flow_model(config: Dict[str, Any]) -> ConditionalFlowModel:
    """创建条件流模型的工厂函数"""
    return ConditionalFlowModel(config)


# 模型注册
if __name__ == "__main__":
    # 测试代码
    config = {
        'architecture': {
            'num_flows': 8,
            'hidden_dim': 256,
            'num_layers': 4,
            'condition_dim': 14,
            'flow_type': 'coupling'
        },
        'trajectory_length': 50
    }
    
    model = ConditionalFlowModel(config)
    print("条件流模型创建成功!")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试生成
    start_pose = np.array([0, 0, 0.5, 0, 0, 0, 1])
    end_pose = np.array([1, 1, 1.0, 0, 0, 0, 1])
    
    trajectories = model.generate_trajectory(start_pose, end_pose, num_samples=3)
    print(f"生成轨迹形状: {trajectories.shape}")
    print("条件流模型测试完成!")