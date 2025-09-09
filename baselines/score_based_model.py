"""
Score-Based Generative Model for Trajectory Generation
基于分数的生成模型轨迹生成
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Any, Optional, Callable, List
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import DiffusionVariantModel


class ScoreBasedTrajectoryModel(DiffusionVariantModel):
    """
    基于分数的轨迹生成模型
    使用分数匹配和随机微分方程进行轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sde_type = config.get('sde_type', 'VP')  # 'VP' (Variance Preserving), 'VE' (Variance Exploding)
        self.sampling_method = config.get('sampling_method', 'euler_maruyama')  # 'euler_maruyama', 'heun', 'dpm_solver'
        self.num_sampling_steps = config.get('num_sampling_steps', 100)
        self.snr_weight = config.get('snr_weight', True)  # 信噪比加权
        
        # SDE参数
        if self.sde_type == 'VP':
            self.beta_min = config.get('beta_min', 0.1)
            self.beta_max = config.get('beta_max', 20.0)
        else:  # VE
            self.sigma_min = config.get('sigma_min', 0.01)
            self.sigma_max = config.get('sigma_max', 50.0)
        
        # 分数网络
        self.score_network = ScoreNetwork(
            input_dim=self.output_dim,
            condition_dim=self.input_dim * 2,
            seq_length=self.max_seq_length,
            model_channels=config.get('model_channels', 128),
            num_res_blocks=config.get('num_res_blocks', 2),
            attention_resolutions=config.get('attention_resolutions', [8, 16]),
            dropout=self.dropout,
            use_attention=config.get('use_attention', True)
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # SDE函数
        self.sde = self._create_sde()
        
    def _create_sde(self) -> 'SDE':
        """
        创建随机微分方程
        
        Returns:
            SDE对象
        """
        if self.sde_type == 'VP':
            return VPSDE(self.beta_min, self.beta_max)
        else:
            return VESDE(self.sigma_min, self.sigma_max)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                num_steps: Optional[int] = None) -> torch.Tensor:
        """
        前向传播 - 分数采样过程
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            num_steps: 采样步数
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        if num_steps is None:
            num_steps = self.num_sampling_steps
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 初始化（从先验分布采样）
        shape = (batch_size, self.max_seq_length, self.output_dim)
        x = self.sde.prior_sampling(shape, device)
        
        # 分数采样
        if self.sampling_method == 'euler_maruyama':
            trajectory = self._euler_maruyama_sampling(x, condition_embedding, num_steps)
        elif self.sampling_method == 'heun':
            trajectory = self._heun_sampling(x, condition_embedding, num_steps)
        elif self.sampling_method == 'dpm_solver':
            trajectory = self._dpm_solver_sampling(x, condition_embedding, num_steps)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _euler_maruyama_sampling(self, x: torch.Tensor, condition: torch.Tensor, 
                               num_steps: int) -> torch.Tensor:
        """
        Euler-Maruyama采样
        
        Args:
            x: 初始噪声 [batch_size, seq_length, output_dim]
            condition: 条件嵌入 [batch_size, condition_dim]
            num_steps: 采样步数
            
        Returns:
            采样的轨迹 [batch_size, seq_length, output_dim]
        """
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=x.device) * (1.0 - i / num_steps)
            
            # 计算分数
            score = self.score_network(x, t, condition)
            
            # SDE系数
            drift, diffusion = self.sde.sde_coefficients(x, t)
            
            # Euler-Maruyama更新
            drift_term = drift - 0.5 * (diffusion ** 2).unsqueeze(-1).unsqueeze(-1) * score
            diffusion_term = diffusion.unsqueeze(-1).unsqueeze(-1) * torch.randn_like(x)
            
            x = x + drift_term * dt + diffusion_term * math.sqrt(dt)
        
        return x
    
    def _heun_sampling(self, x: torch.Tensor, condition: torch.Tensor, 
                      num_steps: int) -> torch.Tensor:
        """
        Heun采样（二阶方法）
        """
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=x.device) * (1.0 - i / num_steps)
            
            # 第一步预测
            score1 = self.score_network(x, t, condition)
            drift1, diffusion1 = self.sde.sde_coefficients(x, t)
            
            drift_term1 = drift1 - 0.5 * (diffusion1 ** 2).unsqueeze(-1).unsqueeze(-1) * score1
            x_pred = x + drift_term1 * dt
            
            # 第二步预测
            t_next = torch.ones(x.shape[0], device=x.device) * (1.0 - (i + 1) / num_steps)
            score2 = self.score_network(x_pred, t_next, condition)
            drift2, diffusion2 = self.sde.sde_coefficients(x_pred, t_next)
            
            drift_term2 = drift2 - 0.5 * (diffusion2 ** 2).unsqueeze(-1).unsqueeze(-1) * score2
            
            # Heun更新
            drift_avg = (drift_term1 + drift_term2) / 2
            diffusion_term = diffusion1.unsqueeze(-1).unsqueeze(-1) * torch.randn_like(x)
            
            x = x + drift_avg * dt + diffusion_term * math.sqrt(dt)
        
        return x
    
    def _dpm_solver_sampling(self, x: torch.Tensor, condition: torch.Tensor, 
                           num_steps: int) -> torch.Tensor:
        """
        DPM-Solver采样（确定性方法）
        """
        # 简化的DPM-Solver实现
        time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device)
        
        for i in range(num_steps):
            t_curr = time_steps[i]
            t_next = time_steps[i + 1]
            
            t_batch = torch.ones(x.shape[0], device=x.device) * t_curr
            
            # 计算分数
            score = self.score_network(x, t_batch, condition)
            
            # 简化的DPM更新
            drift, diffusion = self.sde.sde_coefficients(x, t_batch)
            
            # 确定性更新（无随机项）
            dt = t_next - t_curr
            drift_term = drift - 0.5 * (diffusion ** 2).unsqueeze(-1).unsqueeze(-1) * score
            
            x = x + drift_term * dt
        
        return x
    
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
        训练步骤 - 分数匹配损失
        
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
        
        # 随机采样时间
        t = torch.rand(batch_size, device=device)
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 添加噪声
        noise = torch.randn_like(trajectory)
        mean, std = self.sde.marginal_prob(trajectory, t)
        
        noisy_trajectory = mean + std.unsqueeze(-1).unsqueeze(-1) * noise
        
        # 预测分数
        predicted_score = self.score_network(noisy_trajectory, t, condition_embedding)
        
        # 真实分数 = -noise / std
        true_score = -noise / std.unsqueeze(-1).unsqueeze(-1)
        
        # 分数匹配损失
        if self.snr_weight:
            # 信噪比加权
            snr = (std ** 2) / (1 - std ** 2 + 1e-8)
            weight = snr.unsqueeze(-1).unsqueeze(-1)
            loss = torch.mean(weight * (predicted_score - true_score) ** 2)
        else:
            loss = nn.MSELoss()(predicted_score, true_score)
        
        return {'loss': loss}
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, num_steps: int = 100, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
        
        Args:
            start_pose: 起始位姿 [input_dim]
            end_pose: 终止位姿 [input_dim]
            num_points: 轨迹点数量
            num_steps: 采样步数
            
        Returns:
            生成的轨迹 [num_points, output_dim]
        """
        self.eval()
        
        start_tensor = torch.from_numpy(start_pose).float().unsqueeze(0)
        end_tensor = torch.from_numpy(end_pose).float().unsqueeze(0)
        
        original_seq_length = self.max_seq_length
        self.max_seq_length = num_points
        
        with torch.no_grad():
            trajectory = self.forward(start_tensor, end_tensor, num_steps=num_steps)
            
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
            'sde_type': self.sde_type,
            'sampling_method': self.sampling_method,
            'num_sampling_steps': self.num_sampling_steps,
            'snr_weight': self.snr_weight,
            'model_category': 'Diffusion-based Methods'
        })
        return info


class ScoreNetwork(nn.Module):
    """
    分数网络
    用于预测数据分布的分数函数
    """
    
    def __init__(self, input_dim: int, condition_dim: int, seq_length: int,
                 model_channels: int = 128, num_res_blocks: int = 2,
                 attention_resolutions: list = [8, 16], dropout: float = 0.0,
                 use_attention: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.seq_length = seq_length
        self.use_attention = use_attention
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
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
        
        # 残差块
        self.res_blocks = nn.ModuleList()
        ch = model_channels
        
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                ScoreResBlock(ch, ch, model_channels * 4, dropout=dropout)
            )
        
        # 注意力块
        if use_attention:
            self.attention_blocks = nn.ModuleList()
            for res in attention_resolutions:
                if seq_length % res == 0:
                    self.attention_blocks.append(
                        AttentionBlock(ch, num_heads=4, dropout=dropout)
                    )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, input_dim, 3, padding=1)
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
        
        if embedding_dim % 2 == 1:  # 零填充
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
        return emb
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        分数网络前向传播
        
        Args:
            x: 输入轨迹 [batch_size, seq_length, input_dim]
            t: 时间 [batch_size]
            condition: 条件嵌入 [batch_size, condition_dim]
            
        Returns:
            预测的分数 [batch_size, seq_length, input_dim]
        """
        # 转换为卷积格式
        x_conv = x.transpose(1, 2)  # [batch_size, input_dim, seq_length]
        
        # 时间嵌入
        time_emb = self._get_timestep_embedding(t, 128)
        time_emb = self.time_embed(time_emb)
        
        # 条件嵌入
        cond_emb = self.condition_embed(condition)
        
        # 组合嵌入
        emb = time_emb + cond_emb
        
        # 输入投影
        h = self.input_projection(x_conv)
        
        # 残差块
        for res_block in self.res_blocks:
            h = res_block(h, emb)
        
        # 注意力块
        if self.use_attention:
            for attn_block in self.attention_blocks:
                h = attn_block(h)
        
        # 输出投影
        output = self.output_projection(h)
        
        # 转换回原始格式
        output = output.transpose(1, 2)  # [batch_size, seq_length, input_dim]
        
        return output


class ScoreResBlock(nn.Module):
    """
    分数网络的残差块
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
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
    
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # 注意：这里使用seq_first格式
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
        h = h.transpose(0, 2).transpose(1, 2)  # [seq_length, batch_size, channels]
        
        # 自注意力
        attn_output, _ = self.attention(h, h, h)
        
        # 转换回原始格式
        attn_output = attn_output.transpose(0, 1).transpose(1, 2)  # [batch_size, channels, seq_length]
        
        return x + attn_output


class SDE:
    """
    随机微分方程基类
    """
    
    def __init__(self):
        pass
    
    def sde_coefficients(self, x: torch.Tensor, t: torch.Tensor):
        """
        SDE系数 dx = f(x,t)dt + g(t)dW
        
        Returns:
            drift: f(x,t)
            diffusion: g(t)
        """
        raise NotImplementedError
    
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        """
        边际概率分布的均值和标准差
        
        Returns:
            mean: 均值
            std: 标准差
        """
        raise NotImplementedError
    
    def prior_sampling(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        从先验分布采样
        """
        return torch.randn(shape, device=device)


class VPSDE(SDE):
    """
    方差保持SDE
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        时间相关的beta函数
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def sde_coefficients(self, x: torch.Tensor, t: torch.Tensor):
        """
        VP-SDE系数
        """
        beta_t = self.beta(t)
        drift = -0.5 * beta_t.unsqueeze(-1).unsqueeze(-1) * x
        diffusion = torch.sqrt(beta_t)
        
        return drift, diffusion
    
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        """
        VP-SDE边际概率
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff.unsqueeze(-1).unsqueeze(-1)) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        
        return mean, std


class VESDE(SDE):
    """
    方差爆炸SDE
    """
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        时间相关的sigma函数
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def sde_coefficients(self, x: torch.Tensor, t: torch.Tensor):
        """
        VE-SDE系数
        """
        sigma_t = self.sigma(t)
        drift = torch.zeros_like(x)
        diffusion = sigma_t * torch.sqrt(torch.tensor(2.0 * math.log(self.sigma_max / self.sigma_min)))
        
        return drift, diffusion
    
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        """
        VE-SDE边际概率
        """
        std = self.sigma(t)
        mean = x
        
        return mean, std
    
    def prior_sampling(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        VE-SDE先验采样
        """
        return torch.randn(shape, device=device) * self.sigma_max


class ConditionalScoreModel(ScoreBasedTrajectoryModel):
    """
    条件分数模型
    支持更复杂的条件信息
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_classifier_guidance = config.get('use_classifier_guidance', True)
        self.guidance_scale = config.get('guidance_scale', 1.0)
        
        # 分类器（用于引导）
        if self.use_classifier_guidance:
            self.classifier = TrajectoryClassifier(
                input_dim=self.output_dim,
                condition_dim=self.input_dim * 2,
                seq_length=self.max_seq_length,
                num_classes=config.get('num_classes', 10)
            )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                class_labels: Optional[torch.Tensor] = None,
                guidance_scale: Optional[float] = None) -> torch.Tensor:
        """
        条件前向传播（支持分类器引导）
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 初始化
        shape = (batch_size, self.max_seq_length, self.output_dim)
        x = self.sde.prior_sampling(shape, device)
        
        # 采样过程
        dt = 1.0 / self.num_sampling_steps
        
        for i in range(self.num_sampling_steps):
            t = torch.ones(batch_size, device=device) * (1.0 - i / self.num_sampling_steps)
            
            # 无条件分数
            score_uncond = self.score_network(x, t, torch.zeros_like(condition_embedding))
            
            # 条件分数
            score_cond = self.score_network(x, t, condition_embedding)
            
            # 分类器引导
            if self.use_classifier_guidance and class_labels is not None:
                # 计算分类器梯度
                x_requires_grad = x.clone().requires_grad_(True)
                logits = self.classifier(x_requires_grad, t, condition_embedding)
                
                # 选择对应类别的logits
                selected_logits = logits.gather(1, class_labels.unsqueeze(1)).sum()
                
                # 计算梯度
                classifier_grad = torch.autograd.grad(selected_logits, x_requires_grad)[0]
                
                # 引导分数
                score = score_uncond + guidance_scale * (score_cond - score_uncond) + classifier_grad
            else:
                # 标准条件分数
                score = score_uncond + guidance_scale * (score_cond - score_uncond)
            
            # SDE更新
            drift, diffusion = self.sde.sde_coefficients(x, t)
            drift_term = drift - 0.5 * (diffusion ** 2).unsqueeze(-1).unsqueeze(-1) * score
            diffusion_term = diffusion.unsqueeze(-1).unsqueeze(-1) * torch.randn_like(x)
            
            x = x + drift_term * dt + diffusion_term * math.sqrt(dt)
        
        # 强制边界条件
        x = self._enforce_boundary_conditions(x, start_pose, end_pose)
        
        return x


class TrajectoryClassifier(nn.Module):
    """
    轨迹分类器（用于分类器引导）
    """
    
    def __init__(self, input_dim: int, condition_dim: int, seq_length: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # 分类头
        self.classifier_head = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        分类器前向传播
        
        Args:
            x: 输入轨迹 [batch_size, seq_length, input_dim]
            t: 时间 [batch_size]
            condition: 条件信息 [batch_size, condition_dim]
            
        Returns:
            分类logits [batch_size, num_classes]
        """
        # 特征提取
        x_conv = x.transpose(1, 2)  # [batch_size, input_dim, seq_length]
        traj_features = self.feature_extractor(x_conv)
        
        # 时间嵌入
        time_emb = self._get_timestep_embedding(t, 128)
        time_features = self.time_embed(time_emb)
        
        # 条件嵌入
        cond_features = self.condition_embed(condition)
        
        # 连接所有特征
        combined_features = torch.cat([traj_features, time_features, cond_features], dim=-1)
        
        # 分类
        logits = self.classifier_head(combined_features)
        
        return logits
    
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


class FlowBasedScoreModel(ScoreBasedTrajectoryModel):
    """
    基于流的分数模型
    结合归一化流和分数匹配
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_flow_layers = config.get('num_flow_layers', 4)
        self.coupling_type = config.get('coupling_type', 'affine')  # 'affine', 'additive'
        
        # 归一化流
        self.flow = NormalizingFlow(
            input_dim=self.output_dim * self.max_seq_length,
            num_layers=self.num_flow_layers,
            coupling_type=self.coupling_type,
            condition_dim=self.input_dim * 2
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        基于流的前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 从基分布采样
        z = torch.randn(batch_size, self.output_dim * self.max_seq_length, device=device)
        
        # 通过流变换
        x, log_det = self.flow.inverse(z, condition_embedding)
        
        # 重新整形为轨迹
        trajectory = x.view(batch_size, self.max_seq_length, self.output_dim)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory


class NormalizingFlow(nn.Module):
    """
    归一化流
    """
    
    def __init__(self, input_dim: int, num_layers: int, coupling_type: str = 'affine',
                 condition_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.coupling_type = coupling_type
        
        # 耦合层
        self.coupling_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = CouplingLayer(
                input_dim=input_dim,
                coupling_type=coupling_type,
                condition_dim=condition_dim,
                mask_type='checkerboard' if i % 2 == 0 else 'channel_wise'
            )
            self.coupling_layers.append(layer)
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """
        前向变换 (x -> z)
        """
        log_det_jacobian = 0.0
        
        for layer in self.coupling_layers:
            x, log_det = layer.forward(x, condition)
            log_det_jacobian += log_det
        
        return x, log_det_jacobian
    
    def inverse(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """
        逆变换 (z -> x)
        """
        log_det_jacobian = 0.0
        
        for layer in reversed(self.coupling_layers):
            z, log_det = layer.inverse(z, condition)
            log_det_jacobian += log_det
        
        return z, log_det_jacobian


class CouplingLayer(nn.Module):
    """
    耦合层
    """
    
    def __init__(self, input_dim: int, coupling_type: str = 'affine',
                 condition_dim: int = 0, mask_type: str = 'checkerboard'):
        super().__init__()
        self.input_dim = input_dim
        self.coupling_type = coupling_type
        self.condition_dim = condition_dim
        
        # 创建掩码
        if mask_type == 'checkerboard':
            self.register_buffer('mask', torch.arange(input_dim) % 2)
        else:  # channel_wise
            self.register_buffer('mask', torch.cat([
                torch.zeros(input_dim // 2),
                torch.ones(input_dim - input_dim // 2)
            ]))
        
        # 变换网络
        network_input_dim = input_dim // 2 + condition_dim
        
        if coupling_type == 'affine':
            # 仿射耦合
            self.transform_net = nn.Sequential(
                nn.Linear(network_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, (input_dim - input_dim // 2) * 2)  # scale and shift
            )
        else:  # additive
            # 加性耦合
            self.transform_net = nn.Sequential(
                nn.Linear(network_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim - input_dim // 2)  # shift only
            )
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """
        前向耦合变换
        """
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)
        
        # 网络输入
        if condition is not None:
            net_input = torch.cat([x_masked, condition], dim=-1)
        else:
            net_input = x_masked
        
        # 变换参数
        transform_params = self.transform_net(net_input)
        
        if self.coupling_type == 'affine':
            # 仿射变换
            scale, shift = torch.chunk(transform_params, 2, dim=-1)
            scale = torch.tanh(scale)  # 限制scale的范围
            
            x_transformed = x_unmasked * torch.exp(scale) + shift
            log_det = torch.sum(scale, dim=-1)
        else:  # additive
            # 加性变换
            shift = transform_params
            x_transformed = x_unmasked + shift
            log_det = torch.zeros(x.shape[0], device=x.device)
        
        # 组合结果
        output = x_masked + x_transformed * (1 - self.mask)
        
        return output, log_det
    
    def inverse(self, y: torch.Tensor, condition: Optional[torch.Tensor] = None):
        """
        逆耦合变换
        """
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)
        
        # 网络输入
        if condition is not None:
            net_input = torch.cat([y_masked, condition], dim=-1)
        else:
            net_input = y_masked
        
        # 变换参数
        transform_params = self.transform_net(net_input)
        
        if self.coupling_type == 'affine':
            # 逆仿射变换
            scale, shift = torch.chunk(transform_params, 2, dim=-1)
            scale = torch.tanh(scale)
            
            x_transformed = (y_unmasked - shift) * torch.exp(-scale)
            log_det = -torch.sum(scale, dim=-1)
        else:  # additive
            # 逆加性变换
            shift = transform_params
            x_transformed = y_unmasked - shift
            log_det = torch.zeros(y.shape[0], device=y.device)
        
        # 组合结果
        output = y_masked + x_transformed * (1 - self.mask)
        
        return output, log_det


class TrajectoryClassifier(nn.Module):
    """
    轨迹分类器
    """
    
    def __init__(self, input_dim: int, condition_dim: int, seq_length: int, num_classes: int):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU()
        )
        
        # 时间和条件嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        分类器前向传播
        """
        # 特征提取
        x_conv = x.transpose(1, 2)
        traj_features = self.feature_extractor(x_conv)
        
        # 时间嵌入
        time_emb = self._get_timestep_embedding(t, 128)
        time_features = self.time_embed(time_emb)
        
        # 条件嵌入
        cond_features = self.condition_embed(condition)
        
        # 连接特征
        combined_features = torch.cat([traj_features, time_features, cond_features], dim=-1)
        
        # 分类
        logits = self.classifier(combined_features)
        
        return logits
    
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