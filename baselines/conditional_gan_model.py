"""
Conditional Generative Adversarial Network for Trajectory Generation
条件生成对抗网络轨迹生成模型

基于条件的对抗生成网络，特别适合生成多样化的机械臂轨迹
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from .base_model import BaseTrajectoryModel


class Generator(nn.Module):
    """生成器网络"""
    
    def __init__(self, latent_dim: int, condition_dim: int, trajectory_dim: int, hidden_dims: list):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.trajectory_dim = trajectory_dim
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64)
        )
        
        # 生成器主网络
        layers = []
        input_dim = latent_dim + 64  # latent + encoded condition
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, trajectory_dim))
        layers.append(nn.Tanh())  # 输出范围 [-1, 1]
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            noise: [batch_size, latent_dim]
            condition: [batch_size, condition_dim]
            
        Returns:
            trajectory: [batch_size, trajectory_dim]
        """
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 拼接噪声和条件
        x = torch.cat([noise, condition_encoded], dim=1)
        
        # 生成轨迹
        trajectory = self.main(x)
        
        return trajectory


class Discriminator(nn.Module):
    """判别器网络"""
    
    def __init__(self, trajectory_dim: int, condition_dim: int, hidden_dims: list):
        super().__init__()
        
        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64)
        )
        
        # 判别器主网络
        layers = []
        input_dim = trajectory_dim + 64  # trajectory + encoded condition
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, trajectory: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            trajectory: [batch_size, trajectory_dim]
            condition: [batch_size, condition_dim]
            
        Returns:
            validity: [batch_size, 1]
        """
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 拼接轨迹和条件
        x = torch.cat([trajectory, condition_encoded], dim=1)
        
        # 判别
        validity = self.main(x)
        
        return validity


class ConditionalGANModel(BaseTrajectoryModel):
    """条件生成对抗网络模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型配置
        self.generator_dims = config['architecture']['generator_dims']
        self.discriminator_dims = config['architecture']['discriminator_dims']
        self.latent_dim = config['architecture']['latent_dim']
        self.condition_dim = config['architecture']['condition_dim']  # start_pose + end_pose = 14
        self.gan_loss = config['architecture'].get('gan_loss', 'wgan_gp')
        
        # 数据维度
        self.trajectory_length = config.get('trajectory_length', 50)
        self.pose_dim = 7  # 3D position + quaternion
        self.trajectory_dim = self.trajectory_length * self.pose_dim
        
        # 构建网络
        self._build_networks()
        
        # 损失函数
        if self.gan_loss == 'wgan_gp':
            self.adversarial_loss = self._wgan_gp_loss
        elif self.gan_loss == 'lsgan':
            self.adversarial_loss = nn.MSELoss()
        else:
            self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # 训练参数
        self.lambda_gp = 10.0  # gradient penalty weight
        self.n_critic = 5  # critic iterations per generator iteration
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """构建生成器和判别器网络"""
        
        # 生成器
        self.generator = Generator(
            latent_dim=self.latent_dim,
            condition_dim=self.condition_dim,
            trajectory_dim=self.trajectory_dim,
            hidden_dims=self.generator_dims
        )
        
        # 判别器
        self.discriminator = Discriminator(
            trajectory_dim=self.trajectory_dim,
            condition_dim=self.condition_dim,
            hidden_dims=self.discriminator_dims
        )
        
    def _wgan_gp_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """WGAN-GP损失函数"""
        # Discriminator loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        # Generator loss
        g_loss = -torch.mean(fake_validity)
        
        return d_loss, g_loss
    
    def _compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor, 
                                condition: torch.Tensor) -> torch.Tensor:
        """计算梯度惩罚项"""
        batch_size = real_samples.shape[0]
        device = real_samples.device
        
        # 随机插值
        alpha = torch.rand(batch_size, 1).to(device)
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # 计算判别器输出
        d_interpolates = self.discriminator(interpolates, condition)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 梯度惩罚
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """生成器前向传播"""
        return self.generator(noise, condition)
    
    def compute_generator_loss(self, condition: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """计算生成器损失"""
        device = next(self.parameters()).device
        
        # 生成假样本
        noise = torch.randn(batch_size, self.latent_dim).to(device)
        fake_trajectories = self.generator(noise, condition)
        
        # 判别器对假样本的判断
        fake_validity = self.discriminator(fake_trajectories, condition)
        
        if self.gan_loss == 'wgan_gp':
            g_loss = -torch.mean(fake_validity)
        elif self.gan_loss == 'lsgan':
            g_loss = self.adversarial_loss(fake_validity, torch.ones_like(fake_validity))
        else:
            g_loss = self.adversarial_loss(fake_validity, torch.ones_like(fake_validity))
        
        return {
            'generator_loss': g_loss,
            'fake_trajectories': fake_trajectories
        }
    
    def compute_discriminator_loss(self, real_trajectories: torch.Tensor, 
                                 condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算判别器损失"""
        batch_size = real_trajectories.shape[0]
        device = real_trajectories.device
        
        # 真实样本
        real_validity = self.discriminator(real_trajectories, condition)
        
        # 生成假样本
        noise = torch.randn(batch_size, self.latent_dim).to(device)
        fake_trajectories = self.generator(noise, condition).detach()
        fake_validity = self.discriminator(fake_trajectories, condition)
        
        if self.gan_loss == 'wgan_gp':
            # WGAN-GP损失
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            
            # 梯度惩罚
            gradient_penalty = self._compute_gradient_penalty(
                real_trajectories, fake_trajectories, condition
            )
            d_loss += self.lambda_gp * gradient_penalty
            
            return {
                'discriminator_loss': d_loss,
                'gradient_penalty': gradient_penalty,
                'real_validity': real_validity.mean(),
                'fake_validity': fake_validity.mean()
            }
        else:
            # 标准GAN损失
            real_loss = self.adversarial_loss(real_validity, torch.ones_like(real_validity))
            fake_loss = self.adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (real_loss + fake_loss) / 2
            
            return {
                'discriminator_loss': d_loss,
                'real_loss': real_loss,
                'fake_loss': fake_loss,
                'real_validity': real_validity.mean(),
                'fake_validity': fake_validity.mean()
            }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        self.train()
        
        trajectory = batch['trajectory']
        condition = batch['condition']
        batch_size = trajectory.shape[0]
        
        # 展平轨迹
        trajectory_flat = trajectory.view(batch_size, -1)
        
        # 训练判别器
        d_losses = self.compute_discriminator_loss(trajectory_flat, condition)
        
        # 训练生成器（每n_critic次判别器训练后训练一次）
        g_losses = self.compute_generator_loss(condition, batch_size)
        
        # 返回损失值
        result = {
            'discriminator_loss': d_losses['discriminator_loss'].item(),
            'generator_loss': g_losses['generator_loss'].item(),
            'real_validity': d_losses['real_validity'].item(),
            'fake_validity': d_losses['fake_validity'].item()
        }
        
        if 'gradient_penalty' in d_losses:
            result['gradient_penalty'] = d_losses['gradient_penalty'].item()
        
        return result
    
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
            
            # 生成噪声
            noise = torch.randn(num_samples, self.latent_dim).to(device)
            
            # 生成轨迹
            trajectories_flat = self.generator(noise, condition)
            
            # 重塑为轨迹格式
            trajectories = trajectories_flat.view(num_samples, self.trajectory_length, self.pose_dim)
            
            # 反归一化（从[-1,1]到实际范围）
            trajectories = self._denormalize_trajectories(trajectories)
            
            # 强制执行边界约束
            trajectories = self._enforce_boundary_constraints(
                trajectories, start_pose, end_pose
            )
            
            return trajectories.cpu().numpy()
    
    def _denormalize_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """反归一化轨迹（从[-1,1]到实际范围）"""
        # 位置反归一化到工作空间范围
        trajectories[:, :, :3] = trajectories[:, :, :3] * 1.5  # 假设工作空间范围[-1.5, 1.5]
        
        # 四元数归一化
        quat = trajectories[:, :, 3:7]
        quat_norm = torch.norm(quat, dim=2, keepdim=True)
        trajectories[:, :, 3:7] = quat / (quat_norm + 1e-8)
        
        return trajectories
    
    def _enforce_boundary_constraints(self, trajectories: torch.Tensor, 
                                    start_pose: np.ndarray, end_pose: np.ndarray) -> torch.Tensor:
        """强制执行边界约束"""
        # 设置起点和终点
        trajectories[:, 0] = torch.FloatTensor(start_pose).to(trajectories.device)
        trajectories[:, -1] = torch.FloatTensor(end_pose).to(trajectories.device)
        
        # 平滑过渡调整
        alpha = torch.linspace(0, 1, self.trajectory_length).to(trajectories.device)
        
        for i in range(trajectories.shape[0]):
            # 对位置进行平滑调整
            start_pos = trajectories[i, 0, :3]
            end_pos = trajectories[i, -1, :3]
            
            for t in range(1, self.trajectory_length - 1):
                weight = 0.05 * (1 - abs(2 * alpha[t] - 1))  # 中间权重更大
                interpolated_pos = start_pos * (1 - alpha[t]) + end_pos * alpha[t]
                trajectories[i, t, :3] = (1 - weight) * trajectories[i, t, :3] + weight * interpolated_pos
        
        return trajectories
    
    def interpolate_in_noise_space(self, start_pose: np.ndarray, end_pose: np.ndarray,
                                 num_interpolations: int = 10) -> np.ndarray:
        """在噪声空间中插值生成轨迹"""
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 准备条件
            condition = np.concatenate([start_pose, end_pose])
            condition = torch.FloatTensor(condition).unsqueeze(0).to(device)
            
            # 生成两个随机噪声
            noise1 = torch.randn(1, self.latent_dim).to(device)
            noise2 = torch.randn(1, self.latent_dim).to(device)
            
            # 在噪声空间中插值
            alphas = torch.linspace(0, 1, num_interpolations).to(device)
            trajectories = []
            
            for alpha in alphas:
                noise_interp = (1 - alpha) * noise1 + alpha * noise2
                traj_flat = self.generator(noise_interp, condition)
                traj = traj_flat.view(1, self.trajectory_length, self.pose_dim)
                traj = self._denormalize_trajectories(traj)
                traj = self._enforce_boundary_constraints(traj, start_pose, end_pose)
                trajectories.append(traj.squeeze(0))
            
            return torch.stack(trajectories).cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        generator_params = sum(p.numel() for p in self.generator.parameters())
        discriminator_params = sum(p.numel() for p in self.discriminator.parameters())
        
        return {
            'model_name': 'ConditionalGAN',
            'model_type': 'Probabilistic Generative Models',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'generator_parameters': generator_params,
            'discriminator_parameters': discriminator_params,
            'latent_dim': self.latent_dim,
            'condition_dim': self.condition_dim,
            'generator_dims': self.generator_dims,
            'discriminator_dims': self.discriminator_dims,
            'gan_loss': self.gan_loss,
            'trajectory_length': self.trajectory_length,
            'pose_dim': self.pose_dim,
            'supports_conditional_generation': True,
            'supports_noise_interpolation': True
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.logger.info(f"模型已从 {filepath} 加载")
        return checkpoint.get('model_info', {})


# 工厂函数
def create_conditional_gan_model(config: Dict[str, Any]) -> ConditionalGANModel:
    """创建条件GAN模型的工厂函数"""
    return ConditionalGANModel(config)


# 模型注册
if __name__ == "__main__":
    # 测试代码
    config = {
        'architecture': {
            'generator_dims': [128, 256, 512],
            'discriminator_dims': [512, 256, 128],
            'latent_dim': 64,
            'condition_dim': 14,
            'gan_loss': 'wgan_gp'
        },
        'trajectory_length': 50
    }
    
    model = ConditionalGANModel(config)
    print("条件GAN模型创建成功!")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试生成
    start_pose = np.array([0, 0, 0.5, 0, 0, 0, 1])
    end_pose = np.array([1, 1, 1.0, 0, 0, 0, 1])
    
    trajectories = model.generate_trajectory(start_pose, end_pose, num_samples=3)
    print(f"生成轨迹形状: {trajectories.shape}")
    print("条件GAN模型测试完成!")