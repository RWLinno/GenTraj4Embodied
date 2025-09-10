"""
Latent Diffusion Model for Trajectory Generation
潜在扩散模型轨迹生成
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from .base_model import DiffusionVariantModel


class LatentDiffusionTrajectoryModel(DiffusionVariantModel):
    """
    潜在扩散轨迹生成模型
    在低维潜在空间中进行扩散过程，提高计算效率
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.latent_dim = config.get('latent_dim', 64)
        self.use_vae = config.get('use_vae', True)
        self.vae_kl_weight = config.get('vae_kl_weight', 1e-6)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        
        # 变分自编码器 (VAE)
        if self.use_vae:
            self.vae = TrajectoryVAE(
                input_dim=self.output_dim,
                seq_length=self.max_seq_length,
                latent_dim=self.latent_dim,
                encoder_layers=config.get('vae_encoder_layers', [256, 128]),
                decoder_layers=config.get('vae_decoder_layers', [128, 256])
            )
        else:
            # 简单的自编码器
            self.encoder = TrajectoryEncoder(
                input_dim=self.output_dim,
                seq_length=self.max_seq_length,
                latent_dim=self.latent_dim
            )
            self.decoder = TrajectoryDecoder(
                latent_dim=self.latent_dim,
                output_dim=self.output_dim,
                seq_length=self.max_seq_length
            )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # 潜在空间扩散网络
        self.latent_diffusion_net = LatentUNet(
            input_dim=self.latent_dim,
            condition_dim=256,
            model_channels=config.get('model_channels', 64),
            num_res_blocks=config.get('num_res_blocks', 2),
            attention_resolutions=config.get('attention_resolutions', [8, 16]),
            dropout=self.dropout
        )
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule
        )
        
        # 感知损失网络（可选）
        if self.perceptual_weight > 0:
            self.perceptual_net = PerceptualNetwork(
                input_dim=self.output_dim,
                seq_length=self.max_seq_length
            )
        
    def encode_to_latent(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        将轨迹编码到潜在空间
        
        Args:
            trajectory: 输入轨迹 [batch_size, seq_length, output_dim]
            
        Returns:
            latent: 潜在表示 [batch_size, latent_dim]
            kl_loss: KL散度损失（如果使用VAE）
        """
        if self.use_vae:
            latent, mu, logvar = self.vae.encode(trajectory)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            return latent, kl_loss
        else:
            latent = self.encoder(trajectory)
            return latent, None
    
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码到轨迹
        
        Args:
            latent: 潜在表示 [batch_size, latent_dim]
            
        Returns:
            trajectory: 重构的轨迹 [batch_size, seq_length, output_dim]
        """
        if self.use_vae:
            return self.vae.decode(latent)
        else:
            return self.decoder(latent)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                num_inference_steps: int = 50) -> torch.Tensor:
        """
        前向传播 - 潜在扩散采样
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            num_inference_steps: 推理步数
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 在潜在空间中初始化噪声
        latent_noise = torch.randn(batch_size, self.latent_dim, device=device)
        
        # 设置推理调度器
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # 潜在扩散采样循环
        latent = latent_noise
        for t in self.noise_scheduler.timesteps:
            # 准备时间步
            timestep = t.expand(batch_size).to(device)
            
            # 预测噪声
            noise_pred = self.latent_diffusion_net(latent, timestep, condition_embedding)
            
            # 去噪步骤
            latent = self.noise_scheduler.step(noise_pred, t, latent)
        
        # 从潜在空间解码到轨迹空间
        trajectory = self.decode_from_latent(latent)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _enforce_boundary_conditions(self, trajectory: torch.Tensor,
                                   start_pose: torch.Tensor, 
                                   end_pose: torch.Tensor) -> torch.Tensor:
        """
        强制执行边界条件
        """
        trajectory[:, 0, :] = start_pose
        trajectory[:, -1, :] = end_pose
        return trajectory
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            损失字典
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        trajectory = batch['trajectory']
        
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        # 编码到潜在空间
        latent_target, kl_loss = self.encode_to_latent(trajectory)
        
        # 随机采样噪声和时间步
        noise = torch.randn_like(latent_target)
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # 在潜在空间添加噪声
        noisy_latent = self.noise_scheduler.add_noise(latent_target, noise, timesteps)
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 预测噪声
        noise_pred = self.latent_diffusion_net(noisy_latent, timesteps, condition_embedding)
        
        # 扩散损失
        diffusion_loss = nn.MSELoss()(noise_pred, noise)
        
        # 总损失
        total_loss = diffusion_loss
        loss_dict = {'diffusion_loss': diffusion_loss}
        
        # KL散度损失（如果使用VAE）
        if kl_loss is not None:
            kl_loss = kl_loss * self.vae_kl_weight
            total_loss += kl_loss
            loss_dict['kl_loss'] = kl_loss
        
        # 重构损失（定期计算以监控VAE质量）
        if torch.rand(1).item() < 0.1:  # 10%的概率计算重构损失
            reconstructed = self.decode_from_latent(latent_target)
            recon_loss = nn.MSELoss()(reconstructed, trajectory)
            loss_dict['recon_loss'] = recon_loss
            
            # 感知损失
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_net(reconstructed, trajectory)
                perceptual_loss = perceptual_loss * self.perceptual_weight
                total_loss += perceptual_loss
                loss_dict['perceptual_loss'] = perceptual_loss
        
        loss_dict['loss'] = total_loss
        return loss_dict
    
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
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self.forward(
                start_tensor, end_tensor,
                num_inference_steps=kwargs.get('num_inference_steps', 50)
            )
            
        self.max_seq_length = original_seq_length
        
        return trajectory.squeeze(0).numpy()
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    start_poses: torch.Tensor, end_poses: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算损失函数
        """
        batch = {
            'start_pose': start_poses,
            'end_pose': end_poses,
            'trajectory': targets
        }
        
        loss_dict = self.training_step(batch)
        return loss_dict['loss']
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'latent_dim': self.latent_dim,
            'use_vae': self.use_vae,
            'vae_kl_weight': self.vae_kl_weight,
            'perceptual_weight': self.perceptual_weight,
            'model_category': 'Diffusion-based Methods'
        })
        return info


class TrajectoryVAE(nn.Module):
    """
    轨迹变分自编码器
    """
    
    def __init__(self, input_dim: int, seq_length: int, latent_dim: int,
                 encoder_layers: list = [256, 128], decoder_layers: list = [128, 256]):
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # 编码器
        encoder_input_dim = input_dim * seq_length
        encoder_layers_full = [encoder_input_dim] + encoder_layers
        
        encoder_modules = []
        for i in range(len(encoder_layers_full) - 1):
            encoder_modules.extend([
                nn.Linear(encoder_layers_full[i], encoder_layers_full[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # 均值和方差层
        self.mu_layer = nn.Linear(encoder_layers[-1], latent_dim)
        self.logvar_layer = nn.Linear(encoder_layers[-1], latent_dim)
        
        # 解码器
        decoder_layers_full = [latent_dim] + decoder_layers + [encoder_input_dim]
        
        decoder_modules = []
        for i in range(len(decoder_layers_full) - 1):
            decoder_modules.append(nn.Linear(decoder_layers_full[i], decoder_layers_full[i + 1]))
            if i < len(decoder_layers_full) - 2:
                decoder_modules.extend([nn.ReLU(), nn.Dropout(0.1)])
        
        self.decoder = nn.Sequential(*decoder_modules)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码
        
        Args:
            x: 输入轨迹 [batch_size, seq_length, input_dim]
            
        Returns:
            z: 采样的潜在变量 [batch_size, latent_dim]
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
        """
        # 展平
        x_flat = x.view(x.size(0), -1)
        
        # 编码
        h = self.encoder(x_flat)
        
        # 计算均值和方差
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码
        
        Args:
            z: 潜在变量 [batch_size, latent_dim]
            
        Returns:
            重构的轨迹 [batch_size, seq_length, input_dim]
        """
        # 解码
        x_flat = self.decoder(z)
        
        # 重新整形
        x = x_flat.view(-1, self.seq_length, self.input_dim)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        """
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class TrajectoryEncoder(nn.Module):
    """
    简单的轨迹编码器
    """
    
    def __init__(self, input_dim: int, seq_length: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        # 卷积编码器
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fc = nn.Linear(256, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入轨迹 [batch_size, seq_length, input_dim]
        Returns:
            潜在表示 [batch_size, latent_dim]
        """
        # 转换为卷积格式
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_length]
        
        # 卷积编码
        h = self.conv_layers(x)
        
        # 全连接层
        z = self.fc(h)
        
        return z


class TrajectoryDecoder(nn.Module):
    """
    简单的轨迹解码器
    """
    
    def __init__(self, latent_dim: int, output_dim: int, seq_length: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        
        # 全连接层
        self.fc = nn.Linear(latent_dim, 256 * (seq_length // 4))
        
        # 反卷积解码器
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, output_dim, 3, padding=1)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 潜在表示 [batch_size, latent_dim]
        Returns:
            重构的轨迹 [batch_size, seq_length, output_dim]
        """
        # 全连接层
        h = self.fc(z)
        
        # 重新整形为卷积格式
        h = h.view(-1, 256, self.seq_length // 4)
        
        # 反卷积解码
        x = self.deconv_layers(h)
        
        # 调整到正确的序列长度
        if x.size(2) != self.seq_length:
            x = nn.functional.interpolate(x, size=self.seq_length, mode='linear', align_corners=False)
        
        # 转换回原始格式
        x = x.transpose(1, 2)  # [batch_size, seq_length, output_dim]
        
        return x


class LatentUNet(nn.Module):
    """
    潜在空间的U-Net网络
    """
    
    def __init__(self, input_dim: int, condition_dim: int, model_channels: int = 64,
                 num_res_blocks: int = 2, attention_resolutions: list = [8, 16],
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.model_channels = model_channels
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, model_channels)
        
        # 残差块
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                LatentResBlock(model_channels, model_channels, time_embed_dim, dropout)
            )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, input_dim)
        )
        
    def _get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        获取时间步嵌入
        """
        assert len(timesteps.shape) == 1
        
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
        return emb
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        潜在U-Net前向传播
        
        Args:
            x: 潜在表示 [batch_size, input_dim]
            timesteps: 时间步 [batch_size]
            condition: 条件嵌入 [batch_size, condition_dim]
            
        Returns:
            预测的噪声 [batch_size, input_dim]
        """
        # 时间嵌入
        time_emb = self._get_timestep_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(time_emb)
        
        # 条件嵌入
        cond_emb = self.condition_embed(condition)
        
        # 组合嵌入
        emb = time_emb + cond_emb
        
        # 输入投影
        h = self.input_projection(x)
        
        # 残差块
        for res_block in self.res_blocks:
            h = res_block(h, emb)
        
        # 输出投影
        output = self.output_projection(h)
        
        return output


class LatentResBlock(nn.Module):
    """
    潜在空间的残差块
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dropout: float = 0.0):
        super().__init__()
        
        # 第一个线性层
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.linear1 = nn.Linear(in_channels, out_channels)
        
        # 嵌入投影
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels)
        )
        
        # 第二个线性层
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_channels, out_channels)
        
        # 跳跃连接
        if in_channels != out_channels:
            self.skip_connection = nn.Linear(in_channels, out_channels)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        skip = self.skip_connection(x)
        
        h = self.norm1(x)
        h = nn.functional.silu(h)
        h = self.linear1(h)
        
        # 添加嵌入
        emb_out = self.emb_layers(emb)
        h = h + emb_out
        
        h = self.norm2(h)
        h = nn.functional.silu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        return h + skip


class PerceptualNetwork(nn.Module):
    """
    感知损失网络
    用于计算轨迹的感知相似性
    """
    
    def __init__(self, input_dim: int, seq_length: int):
        super().__init__()
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失
        
        Args:
            pred: 预测轨迹 [batch_size, seq_length, input_dim]
            target: 目标轨迹 [batch_size, seq_length, input_dim]
            
        Returns:
            感知损失
        """
        # 转换为卷积格式
        pred = pred.transpose(1, 2)
        target = target.transpose(1, 2)
        
        # 提取特征
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # 计算L2距离
        perceptual_loss = nn.MSELoss()(pred_features, target_features)
        
        return perceptual_loss


class DDPMScheduler:
    """
    DDPM噪声调度器
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "linear"):
        self.num_train_timesteps = num_train_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 用于推理
        self.timesteps = None
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        余弦beta调度
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        添加噪声
        """
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        
        # 广播到正确的形状
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def set_timesteps(self, num_inference_steps: int):
        """
        设置推理时间步
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        self.timesteps = timesteps.flip(0)  # 从高噪声到低噪声
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """
        去噪步骤
        """
        t = timestep
        
        # 获取调度参数
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 计算预测的原始样本
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # 计算前一个样本的系数
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        # 计算前一个样本
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample


class HierarchicalLatentDiffusion(LatentDiffusionTrajectoryModel):
    """
    分层潜在扩散模型
    在多个分辨率级别进行扩散
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_levels = config.get('num_levels', 3)
        self.level_dims = config.get('level_dims', [16, 32, 64])
        
        # 多级别编码器和解码器
        self.level_encoders = nn.ModuleList()
        self.level_decoders = nn.ModuleList()
        self.level_diffusion_nets = nn.ModuleList()
        
        for i, dim in enumerate(self.level_dims):
            # 编码器
            encoder = TrajectoryEncoder(
                input_dim=self.output_dim if i == 0 else self.level_dims[i-1],
                seq_length=self.max_seq_length // (2 ** i),
                latent_dim=dim
            )
            self.level_encoders.append(encoder)
            
            # 解码器
            decoder = TrajectoryDecoder(
                latent_dim=dim,
                output_dim=self.level_dims[i-1] if i > 0 else self.output_dim,
                seq_length=self.max_seq_length // (2 ** (i-1)) if i > 0 else self.max_seq_length
            )
            self.level_decoders.append(decoder)
            
            # 扩散网络
            diffusion_net = LatentUNet(
                input_dim=dim,
                condition_dim=256,
                model_channels=32 * (2 ** i),
                num_res_blocks=2,
                dropout=self.dropout
            )
            self.level_diffusion_nets.append(diffusion_net)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                num_inference_steps: int = 50) -> torch.Tensor:
        """
        分层扩散采样
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 从粗到细的采样
        current_sample = None
        
        for level in range(self.num_levels):
            # 初始化当前级别的噪声
            latent_noise = torch.randn(batch_size, self.level_dims[level], device=device)
            
            # 如果不是第一级，使用上一级的结果作为条件
            if current_sample is not None:
                # 上采样并作为额外条件
                upsampled_condition = self.level_decoders[level-1](current_sample)
                # 这里可以将upsampled_condition融合到condition_embedding中
            
            # 扩散采样
            self.noise_scheduler.set_timesteps(num_inference_steps // self.num_levels)
            
            latent = latent_noise
            for t in self.noise_scheduler.timesteps:
                timestep = t.expand(batch_size).to(device)
                noise_pred = self.level_diffusion_nets[level](latent, timestep, condition_embedding)
                latent = self.noise_scheduler.step(noise_pred, t, latent)
            
            current_sample = latent
        
        # 最终解码
        trajectory = self.level_decoders[-1](current_sample)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory