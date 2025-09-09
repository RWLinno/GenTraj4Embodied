"""
Kinematic-aware Diffusion Model for Trajectory Generation
运动学约束扩散轨迹生成模型

考虑运动学约束的扩散模型，特别适合机械臂轨迹生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import math

from .base_model import BaseTrajectoryModel


class KinematicConstraints:
    """运动学约束类"""
    
    def __init__(self, config: Dict[str, Any]):
        # 关节限制
        self.joint_limits = config.get('joint_limits', True)
        self.joint_min = torch.tensor([-2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0])  # 示例限制
        self.joint_max = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        
        # 速度限制
        self.velocity_limits = config.get('velocity_limits', True)
        self.max_velocity = 2.0  # m/s
        
        # 加速度限制
        self.acceleration_limits = config.get('acceleration_limits', True)
        self.max_acceleration = 5.0  # m/s^2
        
        # 碰撞避障
        self.collision_avoidance = config.get('collision_avoidance', True)
        self.min_distance_to_obstacles = 0.1  # m
        
    def apply_joint_limits(self, trajectory: torch.Tensor) -> torch.Tensor:
        """应用关节限制"""
        if not self.joint_limits:
            return trajectory
        
        device = trajectory.device
        joint_min = self.joint_min.to(device)
        joint_max = self.joint_max.to(device)
        
        # 对位置进行限制
        trajectory[:, :, :3] = torch.clamp(trajectory[:, :, :3], joint_min[:3], joint_max[:3])
        
        # 对四元数进行归一化
        quat = trajectory[:, :, 3:7]
        quat_norm = torch.norm(quat, dim=2, keepdim=True)
        trajectory[:, :, 3:7] = quat / (quat_norm + 1e-8)
        
        return trajectory
    
    def compute_velocity_constraint_loss(self, trajectory: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """计算速度约束损失"""
        if not self.velocity_limits:
            return torch.tensor(0.0, device=trajectory.device)
        
        # 计算速度
        velocity = (trajectory[:, 1:] - trajectory[:, :-1]) / dt
        velocity_magnitude = torch.norm(velocity[:, :, :3], dim=2)
        
        # 计算超出限制的损失
        velocity_violation = F.relu(velocity_magnitude - self.max_velocity)
        return torch.mean(velocity_violation ** 2)
    
    def compute_acceleration_constraint_loss(self, trajectory: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """计算加速度约束损失"""
        if not self.acceleration_limits:
            return torch.tensor(0.0, device=trajectory.device)
        
        # 计算加速度
        velocity = (trajectory[:, 1:] - trajectory[:, :-1]) / dt
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
        acceleration_magnitude = torch.norm(acceleration[:, :, :3], dim=2)
        
        # 计算超出限制的损失
        acceleration_violation = F.relu(acceleration_magnitude - self.max_acceleration)
        return torch.mean(acceleration_violation ** 2)


class UNet1D(nn.Module):
    """1D U-Net用于轨迹扩散"""
    
    def __init__(self, input_dim: int, time_embed_dim: int, condition_dim: int, 
                 model_channels: int = 128, num_res_blocks: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        self.condition_dim = condition_dim
        self.model_channels = model_channels
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim * 4),
        )
        
        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        
        # 输入投影
        self.input_proj = nn.Conv1d(input_dim, model_channels, 1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList([
            self._make_res_block(model_channels, model_channels, time_embed_dim * 4),
            self._make_res_block(model_channels, model_channels * 2, time_embed_dim * 4),
            self._make_res_block(model_channels * 2, model_channels * 4, time_embed_dim * 4),
        ])
        
        # 中间块
        self.middle_block = self._make_res_block(model_channels * 4, model_channels * 4, time_embed_dim * 4)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList([
            self._make_res_block(model_channels * 8, model_channels * 2, time_embed_dim * 4),
            self._make_res_block(model_channels * 4, model_channels, time_embed_dim * 4),
            self._make_res_block(model_channels * 2, model_channels, time_embed_dim * 4),
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, input_dim, 1)
        )
        
    def _make_res_block(self, in_channels: int, out_channels: int, time_embed_dim: int):
        """创建残差块"""
        return ResBlock1D(in_channels, out_channels, time_embed_dim)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim, seq_len]
            timesteps: [batch_size]
            condition: [batch_size, condition_dim]
        """
        # 时间嵌入
        t_emb = self.timestep_embedding(timesteps, self.time_embed_dim)
        t_emb = self.time_embed(t_emb)
        
        # 条件嵌入
        c_emb = self.condition_embed(condition)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 下采样
        skip_connections = []
        for block in self.down_blocks:
            h = block(h, t_emb)
            skip_connections.append(h)
            h = F.avg_pool1d(h, 2)
        
        # 中间块
        h = self.middle_block(h, t_emb)
        
        # 上采样
        for i, block in enumerate(self.up_blocks):
            h = F.interpolate(h, scale_factor=2, mode='linear', align_corners=False)
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
        
        # 输出投影
        output = self.output_proj(h)
        
        return output
    
    def timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """时间步嵌入"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ResBlock1D(nn.Module):
    """1D残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        t_proj = self.time_proj(t_emb)[:, :, None]
        h = h + t_proj
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)


class KinematicDiffusionModel(BaseTrajectoryModel):
    """运动学约束扩散模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型配置
        self.unet_dim = config['architecture']['unet_dim']
        self.num_layers = config['architecture']['num_layers']
        self.time_embed_dim = config['architecture']['time_embed_dim']
        self.num_timesteps = config['architecture']['num_timesteps']
        self.kinematic_constraints = config['architecture'].get('kinematic_constraints', True)
        self.joint_limits = config['architecture'].get('joint_limits', True)
        self.collision_avoidance = config['architecture'].get('collision_avoidance', True)
        
        # 数据维度
        self.trajectory_length = config.get('trajectory_length', 50)
        self.pose_dim = 7  # 3D position + quaternion
        self.condition_dim = 14  # start_pose + end_pose
        
        # 运动学约束
        self.constraints = KinematicConstraints(config['architecture'])
        
        # 构建网络
        self._build_networks()
        
        # 扩散调度器
        self._setup_diffusion_schedule()
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """构建网络"""
        
        # U-Net网络
        self.unet = UNet1D(
            input_dim=self.pose_dim,
            time_embed_dim=self.time_embed_dim,
            condition_dim=self.condition_dim,
            model_channels=self.unet_dim,
            num_res_blocks=self.num_layers
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.condition_dim)
        )
        
    def _setup_diffusion_schedule(self):
        """设置扩散调度"""
        # 线性beta调度
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 计算其他有用的量
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # 计算后验方差
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """从噪声预测原始数据"""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t][:, None, None]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t][:, None, None]
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def forward(self, trajectory: torch.Tensor, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len, _ = trajectory.shape
        device = trajectory.device
        
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # 添加噪声
        noise = torch.randn_like(trajectory)
        x_noisy = self.q_sample(trajectory, t, noise)
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 转换为U-Net输入格式 [batch_size, pose_dim, seq_len]
        x_noisy_transposed = x_noisy.transpose(1, 2)
        
        # 预测噪声
        predicted_noise = self.unet(x_noisy_transposed, t, condition_encoded)
        
        # 转换回原格式
        predicted_noise = predicted_noise.transpose(1, 2)
        
        return {
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'x_noisy': x_noisy,
            'timesteps': t
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失函数"""
        trajectory = batch['trajectory']
        condition = batch['condition']
        
        # 前向传播
        outputs = self.forward(trajectory, condition)
        
        # 扩散损失
        diffusion_loss = F.mse_loss(outputs['predicted_noise'], outputs['target_noise'])
        
        # 运动学约束损失
        kinematic_loss = torch.tensor(0.0, device=trajectory.device)
        
        if self.kinematic_constraints:
            # 预测原始轨迹
            predicted_x0 = self.predict_start_from_noise(
                outputs['x_noisy'], outputs['timesteps'], outputs['predicted_noise']
            )
            
            # 应用运动学约束
            predicted_x0 = self.constraints.apply_joint_limits(predicted_x0)
            
            # 计算约束损失
            velocity_loss = self.constraints.compute_velocity_constraint_loss(predicted_x0)
            acceleration_loss = self.constraints.compute_acceleration_constraint_loss(predicted_x0)
            
            kinematic_loss = velocity_loss + acceleration_loss
        
        # 总损失
        total_loss = diffusion_loss + 0.1 * kinematic_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'kinematic_loss': kinematic_loss
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        self.train()
        
        # 计算损失
        losses = self.compute_loss(batch)
        
        # 返回标量损失值
        return {
            'loss': losses['total_loss'].item(),
            'diffusion_loss': losses['diffusion_loss'].item(),
            'kinematic_loss': losses['kinematic_loss'].item()
        }
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """单步去噪"""
        batch_size = x.shape[0]
        device = x.device
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 预测噪声
        x_transposed = x.transpose(1, 2)
        predicted_noise = self.unet(x_transposed, t, condition_encoded)
        predicted_noise = predicted_noise.transpose(1, 2)
        
        # 预测原始数据
        predicted_x0 = self.predict_start_from_noise(x, t, predicted_noise)
        
        # 应用运动学约束
        if self.kinematic_constraints:
            predicted_x0 = self.constraints.apply_joint_limits(predicted_x0)
        
        # 计算前一步的均值
        alpha_t = self.alphas[t][:, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t][:, None, None]
        beta_t = self.betas[t][:, None, None]
        
        # 计算后验均值
        pred_mean = (
            (alpha_cumprod_t_prev.sqrt() * beta_t) / (1.0 - alpha_cumprod_t) * predicted_x0 +
            (alpha_t.sqrt() * (1.0 - alpha_cumprod_t_prev)) / (1.0 - alpha_cumprod_t) * x
        )
        
        if t[0] == 0:
            return pred_mean
        else:
            # 添加噪声
            posterior_variance_t = self.posterior_variance[t][:, None, None]
            noise = torch.randn_like(x)
            return pred_mean + posterior_variance_t.sqrt() * noise
    
    @torch.no_grad()
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray, 
                          num_samples: int = 1, **kwargs) -> np.ndarray:
        """生成轨迹"""
        self.eval()
        
        device = next(self.parameters()).device
        
        # 准备条件
        condition = np.concatenate([start_pose, end_pose])  # [14]
        condition = torch.FloatTensor(condition).unsqueeze(0).repeat(num_samples, 1)
        condition = condition.to(device)
        
        # 从噪声开始
        shape = (num_samples, self.trajectory_length, self.pose_dim)
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, condition)
        
        # 强制边界约束
        x = self._enforce_boundary_constraints(x, start_pose, end_pose)
        
        return x.cpu().numpy()
    
    def _enforce_boundary_constraints(self, trajectories: torch.Tensor, 
                                    start_pose: np.ndarray, end_pose: np.ndarray) -> torch.Tensor:
        """强制执行边界约束"""
        # 设置起点和终点
        trajectories[:, 0] = torch.FloatTensor(start_pose).to(trajectories.device)
        trajectories[:, -1] = torch.FloatTensor(end_pose).to(trajectories.device)
        
        # 应用运动学约束
        if self.kinematic_constraints:
            trajectories = self.constraints.apply_joint_limits(trajectories)
        
        return trajectories
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'KinematicDiffusion',
            'model_type': 'Probabilistic Generative Models',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'unet_dim': self.unet_dim,
            'num_layers': self.num_layers,
            'time_embed_dim': self.time_embed_dim,
            'num_timesteps': self.num_timesteps,
            'kinematic_constraints': self.kinematic_constraints,
            'joint_limits': self.joint_limits,
            'collision_avoidance': self.collision_avoidance,
            'trajectory_length': self.trajectory_length,
            'pose_dim': self.pose_dim,
            'supports_conditional_generation': True,
            'supports_kinematic_constraints': True
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
def create_kinematic_diffusion_model(config: Dict[str, Any]) -> KinematicDiffusionModel:
    """创建运动学扩散模型的工厂函数"""
    return KinematicDiffusionModel(config)


# 模型注册
if __name__ == "__main__":
    # 测试代码
    config = {
        'architecture': {
            'unet_dim': 256,
            'num_layers': 4,
            'time_embed_dim': 128,
            'num_timesteps': 1000,
            'kinematic_constraints': True,
            'joint_limits': True,
            'collision_avoidance': True
        },
        'trajectory_length': 50
    }
    
    model = KinematicDiffusionModel(config)
    print("运动学扩散模型创建成功!")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试生成
    start_pose = np.array([0, 0, 0.5, 0, 0, 0, 1])
    end_pose = np.array([1, 1, 1.0, 0, 0, 0, 1])
    
    trajectories = model.generate_trajectory(start_pose, end_pose, num_samples=2)
    print(f"生成轨迹形状: {trajectories.shape}")
    print("运动学扩散模型测试完成!")