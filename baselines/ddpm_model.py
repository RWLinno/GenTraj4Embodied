"""
DDPM (Denoising Diffusion Probabilistic Models) for Trajectory Generation
去噪扩散概率模型轨迹生成
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


class DDPMTrajectoryModel(DiffusionVariantModel):
    """
    DDPM轨迹生成模型
    基于去噪扩散概率模型的轨迹生成方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prediction_type = config.get('prediction_type', 'epsilon')  # 'epsilon', 'x0', 'v'
        self.variance_type = config.get('variance_type', 'fixed_small')  # 'fixed_small', 'fixed_large', 'learned'
        self.clip_sample = config.get('clip_sample', True)
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type
        )
        
        # U-Net架构的去噪网络
        self.denoising_network = UNetTrajectoryModel(
            input_dim=self.output_dim,
            condition_dim=self.input_dim * 2,  # start + end pose
            model_channels=config.get('model_channels', 128),
            num_res_blocks=config.get('num_res_blocks', 2),
            attention_resolutions=config.get('attention_resolutions', [8, 16]),
            channel_mult=config.get('channel_mult', [1, 2, 4]),
            dropout=self.dropout,
            use_checkpoint=config.get('use_checkpoint', False),
            use_scale_shift_norm=config.get('use_scale_shift_norm', True)
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        前向传播 - DDPM采样过程
        
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
        
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        
        # 确保推理步数大于0
        if num_inference_steps <= 0:
            num_inference_steps = 50  # 使用默认值
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 初始化噪声
        shape = (batch_size, self.max_seq_length, self.output_dim)
        trajectory = torch.randn(shape, device=device)
        
        # 设置推理时间步
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # 去噪过程
        for t in self.noise_scheduler.timesteps:
            # 预测噪声
            timestep_embedding = self._get_timestep_embedding(t, device)
            
            # 条件化去噪
            noise_pred = self.denoising_network(
                trajectory, 
                timestep_embedding, 
                condition_embedding
            )
            
            # 调度器步骤
            trajectory = self.noise_scheduler.step(
                noise_pred, t, trajectory
            ).prev_sample
            
            # 强制边界条件
            trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _get_timestep_embedding(self, timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        获取时间步嵌入
        
        Args:
            timestep: 时间步
            device: 设备
            
        Returns:
            时间步嵌入 [1, embedding_dim]
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # 正弦位置编码用于时间步
        half_dim = 128
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timestep[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings
    
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
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(trajectory)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 预测噪声
        timestep_embeddings = torch.stack([
            self._get_timestep_embedding(t, device).squeeze(0) 
            for t in timesteps
        ])
        
        noise_pred = self.denoising_network(
            noisy_trajectory, 
            timestep_embeddings, 
            condition_embedding
        )
        
        # 计算损失
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'x0':
            target = trajectory
        else:  # 'v' (velocity parameterization)
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        
        loss = nn.MSELoss()(noise_pred, target)
        
        return {'loss': loss}
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, num_inference_steps: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            num_inference_steps: 推理步数
            
        Returns:
            生成的轨迹 [num_points, output_dim]
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self.forward(start_tensor, end_tensor, num_inference_steps=num_inference_steps)
            
        self.max_seq_length = original_seq_length
        
        return trajectory.squeeze(0).numpy()
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    start_poses: torch.Tensor, end_poses: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算损失函数
        """
        # 使用training_step计算DDPM损失
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
            'prediction_type': self.prediction_type,
            'variance_type': self.variance_type,
            'clip_sample': self.clip_sample,
            'model_category': 'Diffusion-based Methods'
        })
        return info


class DDPMScheduler:
    """
    DDPM噪声调度器
    """
    
    def __init__(self, num_train_timesteps: int = 1000, 
                 beta_schedule: str = 'linear',
                 prediction_type: str = 'epsilon'):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        
        # 计算beta值
        if beta_schedule == 'linear':
            self.betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(0.0001**0.5, 0.02**0.5, num_train_timesteps) ** 2
        
        # 预计算常用值
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 用于采样的值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于去噪的值
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 后验方差
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        余弦beta调度
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        添加噪声
        
        Args:
            original_samples: 原始样本 [batch_size, ...]
            noise: 噪声 [batch_size, ...]
            timesteps: 时间步 [batch_size]
            
        Returns:
            加噪样本 [batch_size, ...]
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # 扩展维度以匹配样本形状
        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples
    
    def step(self, model_output: torch.Tensor, timestep: int, 
             sample: torch.Tensor) -> 'DDPMSchedulerOutput':
        """
        去噪步骤
        
        Args:
            model_output: 模型输出
            timestep: 当前时间步
            sample: 当前样本
            
        Returns:
            调度器输出
        """
        t = timestep
        
        if self.prediction_type == 'epsilon':
            # 预测噪声
            pred_original_sample = (sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output) / self.sqrt_alphas_cumprod[t]
        elif self.prediction_type == 'x0':
            # 直接预测原始样本
            pred_original_sample = model_output
        else:  # 'v'
            # 速度参数化
            pred_original_sample = self.sqrt_alphas_cumprod[t] * sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        
        # 计算前一时间步的样本
        if t > 0:
            # 添加噪声
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[t]
            
            # 计算均值
            pred_sample_direction = (sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output) / self.sqrt_alphas_cumprod[t]
            prev_sample = self.sqrt_alphas_cumprod[t-1] * pred_original_sample + torch.sqrt(variance) * noise
        else:
            prev_sample = pred_original_sample
        
        return DDPMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    
    def set_timesteps(self, num_inference_steps: int):
        """
        设置推理时间步
        """
        # 确保推理步数大于0
        if num_inference_steps <= 0:
            num_inference_steps = 50  # 使用默认值
            
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        # 转换为numpy进行反向操作，然后转换回torch
        timesteps_numpy = timesteps.numpy()
        timesteps_reversed = timesteps_numpy[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps_reversed)
    
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, 
                    timesteps: torch.Tensor) -> torch.Tensor:
        """
        计算速度参数化的目标
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        while sqrt_alpha_prod.dim() < sample.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        
        return velocity


class DDPMSchedulerOutput:
    """
    DDPM调度器输出
    """
    
    def __init__(self, prev_sample: torch.Tensor, pred_original_sample: torch.Tensor):
        self.prev_sample = prev_sample
        self.pred_original_sample = pred_original_sample


class UNetTrajectoryModel(nn.Module):
    """
    用于轨迹生成的U-Net模型
    """
    
    def __init__(self, input_dim: int, condition_dim: int, model_channels: int = 128,
                 num_res_blocks: int = 2, attention_resolutions: List[int] = [8, 16],
                 channel_mult: List[int] = [1, 2, 4], dropout: float = 0.0,
                 use_checkpoint: bool = False, use_scale_shift_norm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.use_checkpoint = use_checkpoint
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(256, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4)
        )
        
        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4)
        )
        
        # 输入投影
        self.input_projection = nn.Conv1d(input_dim, model_channels, 3, padding=1)
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock1D(
                        ch, out_ch, model_channels * 4,
                        dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
                    )
                )
                ch = out_ch
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(Downsample1D(ch))
        
        # 中间块
        self.middle_block = ResBlock1D(
            ch, ch, model_channels * 4,
            dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
        )
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock1D(
                        ch + (model_channels * mult if i == 0 else 0), 
                        out_ch, model_channels * 4,
                        dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
                    )
                )
                ch = out_ch
            
            if level != 0:
                self.up_blocks.append(Upsample1D(ch))
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, input_dim, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timestep_embed: torch.Tensor,
                condition_embed: torch.Tensor) -> torch.Tensor:
        """
        U-Net前向传播
        
        Args:
            x: 输入轨迹 [batch_size, seq_length, input_dim]
            timestep_embed: 时间步嵌入 [batch_size, 256]
            condition_embed: 条件嵌入 [batch_size, 256]
            
        Returns:
            去噪输出 [batch_size, seq_length, input_dim]
        """
        # 转换为1D卷积格式
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_length]
        
        # 时间步和条件嵌入
        time_emb = self.time_embed(timestep_embed)
        cond_emb = self.condition_embed(condition_embed)
        emb = time_emb + cond_emb
        
        # 输入投影
        h = self.input_projection(x)
        
        # 下采样路径
        hs = [h]
        for module in self.down_blocks:
            if isinstance(module, ResBlock1D):
                h = module(h, emb)
            else:  # Downsample1D
                h = module(h)
            hs.append(h)
        
        # 中间块
        h = self.middle_block(h, emb)
        
        # 上采样路径
        for module in self.up_blocks:
            if isinstance(module, ResBlock1D):
                skip_connection = hs.pop()
                if h.shape != skip_connection.shape:
                    # 处理维度不匹配
                    if h.shape[-1] != skip_connection.shape[-1]:
                        skip_connection = nn.functional.interpolate(
                            skip_connection, size=h.shape[-1], mode='linear', align_corners=False
                        )
                h = torch.cat([h, skip_connection], dim=1)
                h = module(h, emb)
            else:  # Upsample1D
                h = module(h)
        
        # 输出投影
        output = self.output_projection(h)
        
        # 转换回原始格式
        output = output.transpose(1, 2)  # [batch_size, seq_length, input_dim]
        
        return output


class ResBlock1D(nn.Module):
    """
    1D残差块
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int,
                 dropout: float = 0.0, use_scale_shift_norm: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # 第一个卷积
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        # 嵌入投影
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * out_channels if use_scale_shift_norm else out_channels)
        )
        
        # 第二个卷积
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        # 跳跃连接
        if in_channels != out_channels:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        残差块前向传播
        """
        skip = self.skip_connection(x)
        
        # 第一个卷积
        h = self.norm1(x)
        h = nn.functional.silu(h)
        h = self.conv1(h)
        
        # 嵌入
        emb_out = self.emb_layers(emb)
        while emb_out.dim() < h.dim():
            emb_out = emb_out.unsqueeze(-1)
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = self.norm2(h + emb_out)
        
        h = nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + skip


class Downsample1D(nn.Module):
    """
    1D下采样模块
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    1D上采样模块
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        return self.conv(x)


class ConditionalDDPMModel(DDPMTrajectoryModel):
    """
    条件DDPM模型
    支持更复杂的条件信息
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_classifier_free_guidance = config.get('use_classifier_free_guidance', True)
        self.guidance_scale = config.get('guidance_scale', 7.5)
        self.unconditional_prob = config.get('unconditional_prob', 0.1)
        
        # 无条件嵌入
        if self.use_classifier_free_guidance:
            self.unconditional_embedding = nn.Parameter(
                torch.randn(256) * 0.02
            )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None) -> torch.Tensor:
        """
        条件前向传播（支持分类器无关引导）
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        
        # 确保推理步数大于0
        if num_inference_steps <= 0:
            num_inference_steps = 50  # 使用默认值
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 准备无条件嵌入（用于分类器无关引导）
        if self.use_classifier_free_guidance:
            unconditional_embedding = self.unconditional_embedding.unsqueeze(0).expand(batch_size, -1)
        
        # 初始化噪声
        shape = (batch_size, self.max_seq_length, self.output_dim)
        trajectory = torch.randn(shape, device=device)
        
        # 设置推理时间步
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # 去噪过程
        for t in self.noise_scheduler.timesteps:
            timestep_embedding = self._get_timestep_embedding(t, device)
            
            if self.use_classifier_free_guidance:
                # 条件预测
                noise_pred_cond = self.denoising_network(
                    trajectory, 
                    timestep_embedding.expand(batch_size, -1), 
                    condition_embedding
                )
                
                # 无条件预测
                noise_pred_uncond = self.denoising_network(
                    trajectory, 
                    timestep_embedding.expand(batch_size, -1), 
                    unconditional_embedding
                )
                
                # 分类器无关引导
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 标准条件预测
                noise_pred = self.denoising_network(
                    trajectory, 
                    timestep_embedding.expand(batch_size, -1), 
                    condition_embedding
                )
            
            # 调度器步骤
            trajectory = self.noise_scheduler.step(
                noise_pred, t, trajectory
            ).prev_sample
            
            # 强制边界条件
            trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        训练步骤（支持分类器无关引导训练）
        """
        start_pose = batch['start_pose']
        end_pose = batch['end_pose']
        trajectory = batch['trajectory']
        
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(trajectory)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 随机无条件训练（用于分类器无关引导）
        if self.use_classifier_free_guidance and self.training:
            # 随机选择一些样本进行无条件训练
            uncond_mask = torch.rand(batch_size, device=device) < self.unconditional_prob
            condition_embedding[uncond_mask] = self.unconditional_embedding.unsqueeze(0).expand(
                uncond_mask.sum(), -1
            )
        
        # 预测噪声
        timestep_embeddings = torch.stack([
            self._get_timestep_embedding(t, device).squeeze(0) 
            for t in timesteps
        ])
        
        noise_pred = self.denoising_network(
            noisy_trajectory, 
            timestep_embeddings, 
            condition_embedding
        )
        
        # 计算损失
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'x0':
            target = trajectory
        else:  # 'v'
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        
        loss = nn.MSELoss()(noise_pred, target)
        
        return {'loss': loss}