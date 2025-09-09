"""
Diffusion schedulers for trajectory generation
扩散调度器
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class DDPMSchedulerOutput:
    """DDPM调度器输出"""
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DDPMScheduler:
    """DDPM (Denoising Diffusion Probabilistic Models) 调度器"""
    
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 prediction_type: str = "epsilon",
                 clip_sample: bool = True,
                 clip_sample_range: float = 1.0):
        """
        初始化DDPM调度器
        
        Args:
            num_train_timesteps: 训练时间步数
            beta_start: beta起始值
            beta_end: beta结束值
            beta_schedule: beta调度类型 ("linear", "cosine", "scaled_linear")
            prediction_type: 预测类型 ("epsilon", "x0", "v")
            clip_sample: 是否裁剪样本
            clip_sample_range: 裁剪范围
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        
        # 计算beta序列
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # 缩放线性调度，常用于高分辨率图像
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"未知的beta调度类型: {beta_schedule}")
        
        # 计算相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 用于去噪的参数
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        # 初始化时间步
        self.timesteps = None
        self.num_inference_steps = None
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        余弦beta调度
        
        Args:
            timesteps: 时间步数
            s: 小的偏移量
            
        Returns:
            beta序列
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置推理时间步
        
        Args:
            num_inference_steps: 推理步数
            device: 设备
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        
        # 创建时间步序列
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
    
    def add_noise(self, 
                  original_samples: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        向原始样本添加噪声
        
        Args:
            original_samples: 原始样本
            noise: 噪声
            timesteps: 时间步
            
        Returns:
            加噪后的样本
        """
        # 确保参数在正确的设备上
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # 调整形状以进行广播
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> DDPMSchedulerOutput:
        """
        执行一步去噪
        
        Args:
            model_output: 模型输出
            timestep: 当前时间步
            sample: 当前样本
            generator: 随机数生成器
            
        Returns:
            去噪后的样本
        """
        t = timestep
        
        # 获取参数
        beta_prod_t = self.alphas_cumprod[t].to(sample.device)
        beta_prod_t_prev = self.alphas_cumprod_prev[t].to(sample.device)
        
        # 根据预测类型计算原始样本
        if self.prediction_type == "epsilon":
            # 预测噪声
            sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].to(sample.device)
            sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].to(sample.device)
            
            # 调整形状
            while len(sqrt_recip_alphas_cumprod_t.shape) < len(sample.shape):
                sqrt_recip_alphas_cumprod_t = sqrt_recip_alphas_cumprod_t.unsqueeze(-1)
                sqrt_recipm1_alphas_cumprod_t = sqrt_recipm1_alphas_cumprod_t.unsqueeze(-1)
            
            pred_original_sample = sqrt_recip_alphas_cumprod_t * sample - sqrt_recipm1_alphas_cumprod_t * model_output
            
        elif self.prediction_type == "x0":
            # 直接预测原始样本
            pred_original_sample = model_output
            
        elif self.prediction_type == "v":
            # v-parameterization
            sqrt_alpha_prod_t = self.sqrt_alphas_cumprod[t].to(sample.device)
            sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alphas_cumprod[t].to(sample.device)
            
            while len(sqrt_alpha_prod_t.shape) < len(sample.shape):
                sqrt_alpha_prod_t = sqrt_alpha_prod_t.unsqueeze(-1)
                sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.unsqueeze(-1)
            
            pred_original_sample = sqrt_alpha_prod_t * sample - sqrt_one_minus_alpha_prod_t * model_output
        else:
            raise ValueError(f"未知的预测类型: {self.prediction_type}")
        
        # 裁剪样本
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)
        
        # 计算前一个样本的均值
        posterior_mean_coef1_t = self.posterior_mean_coef1[t].to(sample.device)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t].to(sample.device)
        
        while len(posterior_mean_coef1_t.shape) < len(sample.shape):
            posterior_mean_coef1_t = posterior_mean_coef1_t.unsqueeze(-1)
            posterior_mean_coef2_t = posterior_mean_coef2_t.unsqueeze(-1)
        
        pred_prev_sample = posterior_mean_coef1_t * pred_original_sample + posterior_mean_coef2_t * sample
        
        # 添加噪声（除了最后一步）
        if t > 0:
            variance = self.posterior_variance[t].to(sample.device)
            while len(variance.shape) < len(sample.shape):
                variance = variance.unsqueeze(-1)
            
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
        
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
    
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        计算速度参数化
        
        Args:
            sample: 样本
            noise: 噪声
            timesteps: 时间步
            
        Returns:
            速度
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(sample.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(sample.device)
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class DDIMScheduler(DDPMScheduler):
    """DDIM (Denoising Diffusion Implicit Models) 调度器"""
    
    def __init__(self, *args, eta: float = 0.0, **kwargs):
        """
        初始化DDIM调度器
        
        Args:
            eta: DDIM参数，0为确定性采样，1为DDPM采样
        """
        super().__init__(*args, **kwargs)
        self.eta = eta
    
    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> DDPMSchedulerOutput:
        """
        执行DDIM采样步骤
        
        Args:
            model_output: 模型输出
            timestep: 当前时间步
            sample: 当前样本
            generator: 随机数生成器
            
        Returns:
            去噪后的样本
        """
        # 获取当前和前一个时间步的索引
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # 获取alpha参数
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0).to(sample.device)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 计算预测的原始样本
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "x0":
            pred_original_sample = model_output
        else:
            raise ValueError(f"DDIM不支持预测类型: {self.prediction_type}")
        
        # 裁剪样本
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)
        
        # 计算方向向量
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        
        # 计算前一个样本
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        
        # 添加随机性（如果eta > 0）
        if self.eta > 0:
            variance = self.eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_prev)
            if prev_timestep > 0:
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                pred_prev_sample = pred_prev_sample + variance ** 0.5 * noise
        
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


class DPMSolverScheduler(DDPMScheduler):
    """DPM-Solver调度器，用于快速采样"""
    
    def __init__(self, *args, solver_order: int = 2, **kwargs):
        """
        初始化DPM-Solver调度器
        
        Args:
            solver_order: 求解器阶数 (1, 2, 3)
        """
        super().__init__(*args, **kwargs)
        self.solver_order = solver_order
        self.model_outputs = []
    
    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> DDPMSchedulerOutput:
        """
        执行DPM-Solver采样步骤
        
        Args:
            model_output: 模型输出
            timestep: 当前时间步
            sample: 当前样本
            generator: 随机数生成器
            
        Returns:
            去噪后的样本
        """
        # 存储模型输出用于多步求解
        self.model_outputs.append(model_output)
        if len(self.model_outputs) > self.solver_order:
            self.model_outputs.pop(0)
        
        # 根据求解器阶数选择方法
        if len(self.model_outputs) == 1 or self.solver_order == 1:
            return self._dpm_solver_first_order_update(model_output, timestep, sample)
        elif len(self.model_outputs) == 2 and self.solver_order >= 2:
            return self._dpm_solver_second_order_update(timestep, sample)
        elif len(self.model_outputs) == 3 and self.solver_order == 3:
            return self._dpm_solver_third_order_update(timestep, sample)
        else:
            return self._dpm_solver_first_order_update(model_output, timestep, sample)
    
    def _dpm_solver_first_order_update(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> DDPMSchedulerOutput:
        """一阶DPM-Solver更新"""
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0).to(sample.device)
        
        lambda_t = torch.log(alpha_prod_t) - torch.log(1 - alpha_prod_t)
        lambda_s = torch.log(alpha_prod_t_prev) - torch.log(1 - alpha_prod_t_prev)
        
        h = lambda_s - lambda_t
        
        if self.prediction_type == "epsilon":
            x_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5 * sample - (torch.exp(h) - 1) * model_output
        else:
            raise ValueError(f"DPM-Solver不支持预测类型: {self.prediction_type}")
        
        return DDPMSchedulerOutput(prev_sample=x_t)
    
    def _dpm_solver_second_order_update(self, timestep: int, sample: torch.Tensor) -> DDPMSchedulerOutput:
        """二阶DPM-Solver更新"""
        # 简化的二阶实现
        model_output_1, model_output_2 = self.model_outputs[-2], self.model_outputs[-1]
        
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0).to(sample.device)
        
        # 使用线性组合近似
        coeff = 0.5
        combined_output = (1 + coeff) * model_output_2 - coeff * model_output_1
        
        x_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5 * sample - ((1 - alpha_prod_t_prev) ** 0.5 - (1 - alpha_prod_t) ** 0.5) * combined_output
        
        return DDPMSchedulerOutput(prev_sample=x_t)
    
    def _dpm_solver_third_order_update(self, timestep: int, sample: torch.Tensor) -> DDPMSchedulerOutput:
        """三阶DPM-Solver更新"""
        # 简化的三阶实现
        model_output_1, model_output_2, model_output_3 = self.model_outputs[-3], self.model_outputs[-2], self.model_outputs[-1]
        
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0).to(sample.device)
        
        # 使用三次插值
        coeff1, coeff2 = 23/12, -16/12, 5/12
        combined_output = coeff1 * model_output_3 + coeff2 * model_output_2 + (5/12) * model_output_1
        
        x_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5 * sample - ((1 - alpha_prod_t_prev) ** 0.5 - (1 - alpha_prod_t) ** 0.5) * combined_output
        
        return DDPMSchedulerOutput(prev_sample=x_t)