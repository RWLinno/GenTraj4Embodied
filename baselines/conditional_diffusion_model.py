"""
Conditional Diffusion Model for Trajectory Generation
条件扩散模型轨迹生成
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
from base_model import DiffusionVariantModel


class ConditionalDiffusionTrajectoryModel(DiffusionVariantModel):
    """
    条件扩散轨迹生成模型
    支持多种条件信息的扩散模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.condition_types = config.get('condition_types', ['pose', 'task', 'context'])
        self.use_classifier_free_guidance = config.get('use_classifier_free_guidance', True)
        self.guidance_scale = config.get('guidance_scale', 7.5)
        self.unconditional_prob = config.get('unconditional_prob', 0.1)
        
        # 条件编码器
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # 任务编码器（假设有离散的任务类型）
        self.num_task_types = config.get('num_task_types', 10)
        self.task_encoder = nn.Embedding(self.num_task_types, 128)
        
        # 上下文编码器
        self.context_dim = config.get('context_dim', 64)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # 条件融合层
        total_condition_dim = 256 + 128 + 256  # pose + task + context
        self.condition_fusion = nn.Sequential(
            nn.Linear(total_condition_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        # U-Net架构的扩散网络
        self.diffusion_net = ConditionalUNet(
            input_channels=self.output_dim,
            condition_dim=512,
            seq_length=self.max_seq_length,
            model_channels=config.get('model_channels', 128),
            num_res_blocks=config.get('num_res_blocks', 2),
            attention_resolutions=config.get('attention_resolutions', [8, 16]),
            dropout=self.dropout
        )
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule
        )
        
    def encode_conditions(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                         task_type: Optional[torch.Tensor] = None,
                         context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码所有条件信息
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            task_type: 任务类型 [batch_size] (整数索引)
            context: 上下文信息 [batch_size, context_dim]
            
        Returns:
            融合的条件嵌入 [batch_size, condition_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 位姿编码
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        pose_emb = self.pose_encoder(combined_pose)
        
        # 任务编码
        if task_type is not None:
            task_emb = self.task_encoder(task_type)
        else:
            task_emb = torch.zeros(batch_size, 128, device=device)
        
        # 上下文编码
        if context is not None:
            context_emb = self.context_encoder(context)
        else:
            context_emb = torch.zeros(batch_size, 256, device=device)
        
        # 融合所有条件
        combined_condition = torch.cat([pose_emb, task_emb, context_emb], dim=-1)
        condition_embedding = self.condition_fusion(combined_condition)
        
        return condition_embedding
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                task_type: Optional[torch.Tensor] = None,
                num_inference_steps: int = 50) -> torch.Tensor:
        """
        前向传播 - 条件扩散采样
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 上下文信息 [batch_size, context_dim]
            task_type: 任务类型 [batch_size]
            num_inference_steps: 推理步数
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        condition_embedding = self.encode_conditions(start_pose, end_pose, task_type, context)
        
        # 初始化噪声
        shape = (batch_size, self.max_seq_length, self.output_dim)
        trajectory = torch.randn(shape, device=device)
        
        # 设置推理调度器
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # 扩散采样循环
        for t in self.noise_scheduler.timesteps:
            # 准备时间步
            timestep = t.expand(batch_size).to(device)
            
            if self.use_classifier_free_guidance:
                # 分类器自由引导
                # 无条件预测
                uncond_embedding = torch.zeros_like(condition_embedding)
                noise_pred_uncond = self.diffusion_net(trajectory, timestep, uncond_embedding)
                
                # 条件预测
                noise_pred_cond = self.diffusion_net(trajectory, timestep, condition_embedding)
                
                # 引导
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 标准条件预测
                noise_pred = self.diffusion_net(trajectory, timestep, condition_embedding)
            
            # 去噪步骤
            trajectory = self.noise_scheduler.step(noise_pred, t, trajectory)
        
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
        task_type = batch.get('task_type', None)
        context = batch.get('context', None)
        
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        # 随机采样噪声和时间步
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # 添加噪声
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # 编码条件（支持分类器自由引导的训练）
        if self.use_classifier_free_guidance and torch.rand(1).item() < self.unconditional_prob:
            # 无条件训练
            condition_embedding = torch.zeros(batch_size, 512, device=device)
        else:
            # 条件训练
            condition_embedding = self.encode_conditions(start_pose, end_pose, task_type, context)
        
        # 预测噪声
        noise_pred = self.diffusion_net(noisy_trajectory, timesteps, condition_embedding)
        
        # 计算损失
        loss = nn.MSELoss()(noise_pred, noise)
        
        return {'loss': loss}
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, task_type: int = 0,
                          context: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            task_type: 任务类型
            context: 上下文信息 [context_dim]
            
        Returns:
            生成的轨迹 [num_points, output_dim]
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        task_tensor = torch.tensor([task_type]).long()
        
        context_tensor = None
        if context is not None:
            context_tensor = torch.from_numpy(context).float().unsqueeze(0)
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self.forward(
                start_tensor, end_tensor, 
                context=context_tensor, 
                task_type=task_tensor,
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
            'trajectory': targets,
            'task_type': kwargs.get('task_type', None),
            'context': kwargs.get('context', None)
        }
        
        loss_dict = self.training_step(batch)
        return loss_dict['loss']
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'condition_types': self.condition_types,
            'use_classifier_free_guidance': self.use_classifier_free_guidance,
            'guidance_scale': self.guidance_scale,
            'num_task_types': self.num_task_types,
            'context_dim': self.context_dim,
            'model_category': 'Diffusion-based Methods'
        })
        return info


class ConditionalUNet(nn.Module):
    """
    条件U-Net网络
    """
    
    def __init__(self, input_channels: int, condition_dim: int, seq_length: int,
                 model_channels: int = 128, num_res_blocks: int = 2,
                 attention_resolutions: list = [8, 16], dropout: float = 0.0):
        super().__init__()
        self.input_channels = input_channels
        self.condition_dim = condition_dim
        self.seq_length = seq_length
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
        self.input_projection = nn.Conv1d(input_channels, model_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        
        # 第一个下采样块
        self.down_blocks.append(
            DownBlock(ch, ch * 2, time_embed_dim, num_res_blocks, dropout)
        )
        ch = ch * 2
        
        # 第二个下采样块
        self.down_blocks.append(
            DownBlock(ch, ch * 2, time_embed_dim, num_res_blocks, dropout)
        )
        ch = ch * 2
        
        # 中间块
        self.middle_block = MiddleBlock(ch, time_embed_dim, dropout)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        
        # 第一个上采样块
        self.up_blocks.append(
            UpBlock(ch + ch, ch // 2, time_embed_dim, num_res_blocks, dropout)
        )
        ch = ch // 2
        
        # 第二个上采样块
        self.up_blocks.append(
            UpBlock(ch + ch, ch // 2, time_embed_dim, num_res_blocks, dropout)
        )
        ch = ch // 2
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, input_channels, 3, padding=1)
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
        U-Net前向传播
        
        Args:
            x: 输入轨迹 [batch_size, seq_length, input_channels]
            timesteps: 时间步 [batch_size]
            condition: 条件嵌入 [batch_size, condition_dim]
            
        Returns:
            预测的噪声 [batch_size, seq_length, input_channels]
        """
        # 转换为卷积格式
        x = x.transpose(1, 2)  # [batch_size, input_channels, seq_length]
        
        # 时间嵌入
        time_emb = self._get_timestep_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(time_emb)
        
        # 条件嵌入
        cond_emb = self.condition_embed(condition)
        
        # 组合嵌入
        emb = time_emb + cond_emb
        
        # 输入投影
        h = self.input_projection(x)
        
        # 下采样路径
        skip_connections = [h]
        
        for down_block in self.down_blocks:
            h = down_block(h, emb)
            skip_connections.append(h)
        
        # 中间块
        h = self.middle_block(h, emb)
        
        # 上采样路径
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, emb)
        
        # 输出投影
        output = self.output_projection(h)
        
        # 转换回原始格式
        output = output.transpose(1, 2)  # [batch_size, seq_length, input_channels]
        
        return output


class DownBlock(nn.Module):
    """
    下采样块
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int,
                 num_res_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        
        # 残差块
        self.res_blocks = nn.ModuleList()
        ch = in_channels
        
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(ch, out_channels if i == num_res_blocks - 1 else ch, 
                        emb_channels, dropout)
            )
            ch = out_channels if i == num_res_blocks - 1 else ch
        
        # 下采样
        self.downsample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x, emb)
        
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """
    上采样块
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int,
                 num_res_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        
        # 上采样
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # 残差块
        self.res_blocks = nn.ModuleList()
        ch = in_channels
        
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(ch, out_channels if i == num_res_blocks - 1 else ch,
                        emb_channels, dropout)
            )
            ch = out_channels if i == num_res_blocks - 1 else ch
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        for res_block in self.res_blocks:
            x = res_block(x, emb)
        
        return x


class MiddleBlock(nn.Module):
    """
    中间块
    """
    
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0):
        super().__init__()
        
        self.res_block1 = ResBlock(channels, channels, emb_channels, dropout)
        self.attention = AttentionBlock(channels)
        self.res_block2 = ResBlock(channels, channels, emb_channels, dropout)
        
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res_block1(x, emb)
        x = self.attention(x)
        x = self.res_block2(x, emb)
        return x


class ResBlock(nn.Module):
    """
    残差块
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dropout: float = 0.0):
        super().__init__()
        
        # 第一个卷积
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        # 嵌入投影
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels)
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
        skip = self.skip_connection(x)
        
        h = self.norm1(x)
        h = nn.functional.silu(h)
        h = self.conv1(h)
        
        # 添加嵌入
        emb_out = self.emb_layers(emb)
        while emb_out.dim() < h.dim():
            emb_out = emb_out.unsqueeze(-1)
        h = h + emb_out
        
        h = self.norm2(h)
        h = nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + skip


class AttentionBlock(nn.Module):
    """
    注意力块
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_length]
        Returns:
            [batch_size, channels, seq_length]
        """
        batch_size, channels, seq_length = x.shape
        
        # 归一化
        h = self.norm(x)
        
        # 转换为注意力格式 [seq_length, batch_size, channels]
        h = h.transpose(0, 2).transpose(1, 2)
        
        # 自注意力
        attn_output, _ = self.attention(h, h, h)
        
        # 转换回原始格式
        attn_output = attn_output.transpose(0, 1).transpose(1, 2)
        
        return x + attn_output


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


class MultiModalConditionalDiffusion(ConditionalDiffusionTrajectoryModel):
    """
    多模态条件扩散模型
    支持图像、文本、传感器等多种模态的条件信息
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 图像编码器
        self.use_image_condition = config.get('use_image_condition', False)
        if self.use_image_condition:
            self.image_encoder = ImageEncoder(
                input_channels=config.get('image_channels', 3),
                output_dim=256
            )
        
        # 文本编码器
        self.use_text_condition = config.get('use_text_condition', False)
        if self.use_text_condition:
            self.text_encoder = TextEncoder(
                vocab_size=config.get('vocab_size', 10000),
                embed_dim=config.get('text_embed_dim', 256),
                output_dim=256
            )
        
        # 传感器编码器
        self.use_sensor_condition = config.get('use_sensor_condition', False)
        if self.use_sensor_condition:
            self.sensor_dim = config.get('sensor_dim', 32)
            self.sensor_encoder = nn.Sequential(
                nn.Linear(self.sensor_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256)
            )
        
        # 更新条件融合层
        total_condition_dim = 256 + 128 + 256  # 基础条件
        if self.use_image_condition:
            total_condition_dim += 256
        if self.use_text_condition:
            total_condition_dim += 256
        if self.use_sensor_condition:
            total_condition_dim += 256
            
        self.condition_fusion = nn.Sequential(
            nn.Linear(total_condition_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
    
    def encode_multimodal_conditions(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                                   task_type: Optional[torch.Tensor] = None,
                                   context: Optional[torch.Tensor] = None,
                                   image: Optional[torch.Tensor] = None,
                                   text: Optional[torch.Tensor] = None,
                                   sensor_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码多模态条件信息
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 基础条件编码
        condition_list = []
        
        # 位姿编码
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        pose_emb = self.pose_encoder(combined_pose)
        condition_list.append(pose_emb)
        
        # 任务编码
        if task_type is not None:
            task_emb = self.task_encoder(task_type)
        else:
            task_emb = torch.zeros(batch_size, 128, device=device)
        condition_list.append(task_emb)
        
        # 上下文编码
        if context is not None:
            context_emb = self.context_encoder(context)
        else:
            context_emb = torch.zeros(batch_size, 256, device=device)
        condition_list.append(context_emb)
        
        # 图像编码
        if self.use_image_condition:
            if image is not None:
                image_emb = self.image_encoder(image)
            else:
                image_emb = torch.zeros(batch_size, 256, device=device)
            condition_list.append(image_emb)
        
        # 文本编码
        if self.use_text_condition:
            if text is not None:
                text_emb = self.text_encoder(text)
            else:
                text_emb = torch.zeros(batch_size, 256, device=device)
            condition_list.append(text_emb)
        
        # 传感器编码
        if self.use_sensor_condition:
            if sensor_data is not None:
                sensor_emb = self.sensor_encoder(sensor_data)
            else:
                sensor_emb = torch.zeros(batch_size, 256, device=device)
            condition_list.append(sensor_emb)
        
        # 融合所有条件
        combined_condition = torch.cat(condition_list, dim=-1)
        condition_embedding = self.condition_fusion(combined_condition)
        
        return condition_embedding


class ImageEncoder(nn.Module):
    """
    图像编码器
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 第二层卷积
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 第三层卷积
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 第四层卷积
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]
        Returns:
            图像嵌入 [batch_size, output_dim]
        """
        return self.encoder(x)


class TextEncoder(nn.Module):
    """
    文本编码器
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, output_dim: int = 256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(embed_dim * 2, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入文本tokens [batch_size, seq_length]
        Returns:
            文本嵌入 [batch_size, output_dim]
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_length, embed_dim]
        
        # LSTM编码
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # 使用最后的隐藏状态
        # hidden: [2, batch_size, embed_dim] (双向)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch_size, embed_dim * 2]
        
        # 投影到输出维度
        output = self.projection(hidden)
        
        return output