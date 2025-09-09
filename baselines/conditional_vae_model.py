"""
Conditional Variational Autoencoder for Trajectory Generation
条件变分自编码器轨迹生成模型

支持起点/终点条件的变分自编码器，特别适合机械臂轨迹生成任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from .base_model import BaseTrajectoryModel


class ConditionalVAEModel(BaseTrajectoryModel):
    """条件变分自编码器模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型配置
        self.encoder_dims = config['architecture']['encoder_dims']
        self.decoder_dims = config['architecture']['decoder_dims']
        self.latent_dim = config['architecture']['latent_dim']
        self.condition_dim = config['architecture']['condition_dim']  # start_pose + end_pose = 14
        self.beta = config['architecture']['beta']
        
        # 数据维度
        self.trajectory_length = config.get('trajectory_length', 50)
        self.pose_dim = 7  # 3D position + quaternion
        self.trajectory_dim = self.trajectory_length * self.pose_dim
        
        # 构建网络
        self._build_networks()
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """构建编码器和解码器网络"""
        
        # 编码器：trajectory + condition -> latent
        encoder_layers = []
        input_dim = self.trajectory_dim + self.condition_dim
        
        for i, hidden_dim in enumerate(self.encoder_dims):
            encoder_layers.extend([
                nn.Linear(input_dim if i == 0 else self.encoder_dims[i-1], hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和方差网络
        self.mu_layer = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        self.logvar_layer = nn.Linear(self.encoder_dims[-1], self.latent_dim)
        
        # 解码器：latent + condition -> trajectory
        decoder_layers = []
        input_dim = self.latent_dim + self.condition_dim
        
        for i, hidden_dim in enumerate(self.decoder_dims):
            decoder_layers.extend([
                nn.Linear(input_dim if i == 0 else self.decoder_dims[i-1], hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        decoder_layers.append(nn.Linear(self.decoder_dims[-1], self.trajectory_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 条件编码器（可选）
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.condition_dim)
        )
        
    def encode(self, trajectory: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码轨迹和条件到潜在空间
        
        Args:
            trajectory: [batch_size, trajectory_length, pose_dim]
            condition: [batch_size, condition_dim]
            
        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        batch_size = trajectory.shape[0]
        
        # 展平轨迹
        trajectory_flat = trajectory.view(batch_size, -1)
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 拼接轨迹和条件
        x = torch.cat([trajectory_flat, condition_encoded], dim=1)
        
        # 编码
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码到轨迹
        
        Args:
            z: [batch_size, latent_dim]
            condition: [batch_size, condition_dim]
            
        Returns:
            trajectory: [batch_size, trajectory_length, pose_dim]
        """
        batch_size = z.shape[0]
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 拼接潜在变量和条件
        x = torch.cat([z, condition_encoded], dim=1)
        
        # 解码
        trajectory_flat = self.decoder(x)
        
        # 重塑为轨迹格式
        trajectory = trajectory_flat.view(batch_size, self.trajectory_length, self.pose_dim)
        
        return trajectory
    
    def forward(self, trajectory: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            trajectory: [batch_size, trajectory_length, pose_dim]
            condition: [batch_size, condition_dim]
            
        Returns:
            包含重构轨迹、mu、logvar的字典
        """
        # 编码
        mu, logvar = self.encode(trajectory, condition)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_trajectory = self.decode(z, condition)
        
        return {
            'recon_trajectory': recon_trajectory,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失函数"""
        trajectory = batch['trajectory']
        condition = batch['condition']
        
        # 前向传播
        outputs = self.forward(trajectory, condition)
        
        # 重构损失
        recon_loss = self.mse_loss(outputs['recon_trajectory'], trajectory)
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        kl_loss = kl_loss / trajectory.shape[0]  # 平均到batch
        
        # 总损失
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        self.train()
        
        # 计算损失
        losses = self.compute_loss(batch)
        
        # 返回标量损失值
        return {
            'loss': losses['total_loss'].item(),
            'recon_loss': losses['recon_loss'].item(),
            'kl_loss': losses['kl_loss'].item()
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
            # 准备条件
            condition = np.concatenate([start_pose, end_pose])  # [14]
            condition = torch.FloatTensor(condition).unsqueeze(0).repeat(num_samples, 1)
            condition = condition.to(next(self.parameters()).device)
            
            # 从先验分布采样
            z = torch.randn(num_samples, self.latent_dim)
            z = z.to(next(self.parameters()).device)
            
            # 解码生成轨迹
            trajectories = self.decode(z, condition)
            
            # 确保起点和终点约束
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
        
        # 平滑插值调整（可选）
        alpha = torch.linspace(0, 1, self.trajectory_length).to(trajectories.device)
        
        for i in range(trajectories.shape[0]):
            # 对位置进行线性插值调整
            start_pos = trajectories[i, 0, :3]
            end_pos = trajectories[i, -1, :3]
            
            # 计算插值权重
            for t in range(1, self.trajectory_length - 1):
                weight = 0.1  # 插值强度
                interpolated_pos = start_pos * (1 - alpha[t]) + end_pos * alpha[t]
                trajectories[i, t, :3] = (1 - weight) * trajectories[i, t, :3] + weight * interpolated_pos
        
        return trajectories
    
    def interpolate_in_latent_space(self, start_pose: np.ndarray, end_pose: np.ndarray,
                                  num_interpolations: int = 10) -> np.ndarray:
        """在潜在空间中插值生成轨迹"""
        self.eval()
        
        with torch.no_grad():
            # 准备条件
            condition = np.concatenate([start_pose, end_pose])
            condition = torch.FloatTensor(condition).unsqueeze(0)
            condition = condition.to(next(self.parameters()).device)
            
            # 生成两个随机潜在变量
            z1 = torch.randn(1, self.latent_dim).to(next(self.parameters()).device)
            z2 = torch.randn(1, self.latent_dim).to(next(self.parameters()).device)
            
            # 在潜在空间中插值
            alphas = torch.linspace(0, 1, num_interpolations).to(next(self.parameters()).device)
            trajectories = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                traj = self.decode(z_interp, condition)
                traj = self._enforce_boundary_constraints(traj, start_pose, end_pose)
                trajectories.append(traj.squeeze(0))
            
            return torch.stack(trajectories).cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ConditionalVAE',
            'model_type': 'Fundamental Architectures',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'latent_dim': self.latent_dim,
            'condition_dim': self.condition_dim,
            'encoder_dims': self.encoder_dims,
            'decoder_dims': self.decoder_dims,
            'beta': self.beta,
            'trajectory_length': self.trajectory_length,
            'pose_dim': self.pose_dim,
            'supports_conditional_generation': True,
            'supports_latent_interpolation': True
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
def create_conditional_vae_model(config: Dict[str, Any]) -> ConditionalVAEModel:
    """创建条件VAE模型的工厂函数"""
    return ConditionalVAEModel(config)


# 模型注册
if __name__ == "__main__":
    # 测试代码
    config = {
        'architecture': {
            'encoder_dims': [512, 256, 128],
            'decoder_dims': [128, 256, 512],
            'latent_dim': 64,
            'condition_dim': 14,
            'beta': 1.0
        },
        'trajectory_length': 50
    }
    
    model = ConditionalVAEModel(config)
    print("条件VAE模型创建成功!")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试生成
    start_pose = np.array([0, 0, 0.5, 0, 0, 0, 1])
    end_pose = np.array([1, 1, 1.0, 0, 0, 0, 1])
    
    trajectories = model.generate_trajectory(start_pose, end_pose, num_samples=3)
    print(f"生成轨迹形状: {trajectories.shape}")
    print("条件VAE模型测试完成!")