"""
FRMD (Fast Robot Motion Diffusion) Model for Trajectory Generation
快速机器人运动扩散模型 - 基于一致性蒸馏的运动基元
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


class FRMDTrajectoryModel(DiffusionVariantModel):
    """
    FRMD轨迹生成模型
    结合Movement Primitives和Consistency Models实现快速轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_primitives = config.get('num_primitives', 10)
        self.primitive_dim = config.get('primitive_dim', 32)
        self.consistency_steps = config.get('consistency_steps', 4)
        self.use_consistency_distillation = config.get('use_consistency_distillation', True)
        self.sigma_min = config.get('sigma_min', 0.002)
        self.sigma_max = config.get('sigma_max', 80.0)
        
        # Movement Primitives编码器
        self.primitive_encoder = MovementPrimitiveEncoder(
            input_dim=self.output_dim,
            primitive_dim=self.primitive_dim,
            num_primitives=self.num_primitives,
            seq_length=self.max_seq_length
        )
        
        # Movement Primitives解码器
        self.primitive_decoder = MovementPrimitiveDecoder(
            primitive_dim=self.primitive_dim,
            output_dim=self.output_dim,
            num_primitives=self.num_primitives,
            seq_length=self.max_seq_length
        )
        
        # 一致性模型
        self.consistency_model = ConsistencyModel(
            input_dim=self.primitive_dim,
            condition_dim=self.input_dim * 2,  # start + end pose
            model_channels=config.get('model_channels', 128),
            num_res_blocks=config.get('num_res_blocks', 2),
            dropout=self.dropout,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # 噪声调度（用于训练）
        self.sigma_schedule = self._create_sigma_schedule()
        
    def _create_sigma_schedule(self) -> torch.Tensor:
        """
        创建噪声水平调度
        
        Returns:
            噪声水平序列 [num_timesteps]
        """
        # 对数线性插值
        log_sigma_min = math.log(self.sigma_min)
        log_sigma_max = math.log(self.sigma_max)
        
        log_sigmas = torch.linspace(log_sigma_max, log_sigma_min, self.num_timesteps)
        sigmas = torch.exp(log_sigmas)
        
        return sigmas
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                num_steps: Optional[int] = None) -> torch.Tensor:
        """
        前向传播 - FRMD单步生成
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            num_steps: 一致性步数
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        if num_steps is None:
            num_steps = self.consistency_steps
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 初始化噪声（在基元空间中）
        noise_primitives = torch.randn(batch_size, self.num_primitives, self.primitive_dim, device=device)
        
        # 一致性模型生成
        if self.use_consistency_distillation:
            # 单步生成
            clean_primitives = self.consistency_model.generate_single_step(
                noise_primitives, condition_embedding
            )
        else:
            # 多步生成
            clean_primitives = self.consistency_model.generate_multi_step(
                noise_primitives, condition_embedding, num_steps
            )
        
        # 解码为轨迹
        trajectory = self.primitive_decoder(clean_primitives, condition_embedding)
        
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
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        # 编码轨迹为运动基元
        target_primitives = self.primitive_encoder(trajectory, condition_embedding)
        
        # 随机采样噪声水平
        sigma_indices = torch.randint(0, len(self.sigma_schedule), (batch_size,), device=device)
        sigmas = self.sigma_schedule[sigma_indices]
        
        # 添加噪声到基元
        noise = torch.randn_like(target_primitives)
        noisy_primitives = target_primitives + sigmas.view(-1, 1, 1) * noise
        
        # 一致性模型预测
        if self.use_consistency_distillation:
            # 一致性蒸馏损失
            pred_primitives = self.consistency_model(noisy_primitives, sigmas, condition_embedding)
            consistency_loss = nn.MSELoss()(pred_primitives, target_primitives)
            
            # 解码损失
            pred_trajectory = self.primitive_decoder(pred_primitives, condition_embedding)
            reconstruction_loss = nn.MSELoss()(pred_trajectory, trajectory)
            
            total_loss = consistency_loss + reconstruction_loss
        else:
            # 标准扩散损失
            pred_noise = self.consistency_model(noisy_primitives, sigmas, condition_embedding)
            diffusion_loss = nn.MSELoss()(pred_noise, noise)
            
            total_loss = diffusion_loss
        
        # 基元重建损失
        primitive_loss_weight = self.config.get('primitive_loss_weight', 0.1)
        if primitive_loss_weight > 0:
            reconstructed_trajectory = self.primitive_decoder(target_primitives, condition_embedding)
            primitive_loss = nn.MSELoss()(reconstructed_trajectory, trajectory)
            total_loss = total_loss + primitive_loss_weight * primitive_loss
        
        return {'loss': total_loss}
    
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
            trajectory = self.forward(start_tensor, end_tensor)
            
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
            'num_primitives': self.num_primitives,
            'primitive_dim': self.primitive_dim,
            'consistency_steps': self.consistency_steps,
            'use_consistency_distillation': self.use_consistency_distillation,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'model_category': 'Diffusion-based Methods'
        })
        return info


class MovementPrimitiveEncoder(nn.Module):
    """
    运动基元编码器
    将轨迹编码为低维运动基元表示
    """
    
    def __init__(self, input_dim: int, primitive_dim: int, num_primitives: int, seq_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.primitive_dim = primitive_dim
        self.num_primitives = num_primitives
        self.seq_length = seq_length
        
        # 轨迹编码器
        self.trajectory_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(num_primitives),
            nn.Conv1d(128, primitive_dim, 1)
        )
        
        # 条件融合
        self.condition_fusion = nn.Sequential(
            nn.Linear(256, primitive_dim),
            nn.ReLU(),
            nn.Linear(primitive_dim, primitive_dim)
        )
        
    def forward(self, trajectory: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        编码轨迹为运动基元
        
        Args:
            trajectory: 输入轨迹 [batch_size, seq_length, input_dim]
            condition: 条件信息 [batch_size, condition_dim]
            
        Returns:
            运动基元 [batch_size, num_primitives, primitive_dim]
        """
        # 转换为卷积格式
        traj_conv = trajectory.transpose(1, 2)  # [batch_size, input_dim, seq_length]
        
        # 编码轨迹
        primitives = self.trajectory_encoder(traj_conv)  # [batch_size, primitive_dim, num_primitives]
        primitives = primitives.transpose(1, 2)  # [batch_size, num_primitives, primitive_dim]
        
        # 融合条件信息
        condition_effect = self.condition_fusion(condition)  # [batch_size, primitive_dim]
        condition_effect = condition_effect.unsqueeze(1).expand(-1, self.num_primitives, -1)
        
        conditioned_primitives = primitives + condition_effect
        
        return conditioned_primitives


class MovementPrimitiveDecoder(nn.Module):
    """
    运动基元解码器
    将运动基元解码为完整轨迹
    """
    
    def __init__(self, primitive_dim: int, output_dim: int, num_primitives: int, seq_length: int):
        super().__init__()
        self.primitive_dim = primitive_dim
        self.output_dim = output_dim
        self.num_primitives = num_primitives
        self.seq_length = seq_length
        
        # 基元解码器
        self.primitive_decoder = nn.Sequential(
            nn.ConvTranspose1d(primitive_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, output_dim, 3, padding=1)
        )
        
        # 时间基函数
        self.time_basis = nn.Parameter(
            torch.randn(num_primitives, seq_length) * 0.1
        )
        
        # 条件调制
        self.condition_modulation = nn.Sequential(
            nn.Linear(256, primitive_dim * num_primitives),
            nn.Tanh()
        )
        
        # 输出细化
        self.output_refinement = nn.Sequential(
            nn.Conv1d(output_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, output_dim, 3, padding=1)
        )
        
    def forward(self, primitives: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        解码运动基元为轨迹
        
        Args:
            primitives: 运动基元 [batch_size, num_primitives, primitive_dim]
            condition: 条件信息 [batch_size, condition_dim]
            
        Returns:
            解码的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = primitives.shape[0]
        device = primitives.device
        
        # 条件调制
        modulation = self.condition_modulation(condition)  # [batch_size, primitive_dim * num_primitives]
        modulation = modulation.view(batch_size, self.num_primitives, self.primitive_dim)
        
        # 应用调制
        modulated_primitives = primitives + modulation
        
        # 时间基函数组合
        time_weights = torch.softmax(self.time_basis, dim=0)  # [num_primitives, seq_length]
        
        # 计算每个基元对轨迹的贡献
        trajectory_contributions = []
        
        for i in range(self.num_primitives):
            # 当前基元
            primitive = modulated_primitives[:, i, :]  # [batch_size, primitive_dim]
            
            # 扩展到序列长度
            primitive_seq = primitive.unsqueeze(1).expand(-1, self.seq_length, -1)
            primitive_seq = primitive_seq.transpose(1, 2)  # [batch_size, primitive_dim, seq_length]
            
            # 解码基元
            decoded = self.primitive_decoder(primitive_seq)  # [batch_size, output_dim, seq_length]
            decoded = decoded.transpose(1, 2)  # [batch_size, seq_length, output_dim]
            
            # 应用时间权重
            time_weight = time_weights[i].unsqueeze(0).unsqueeze(-1)  # [1, seq_length, 1]
            weighted_contribution = decoded * time_weight
            
            trajectory_contributions.append(weighted_contribution)
        
        # 组合所有基元的贡献
        combined_trajectory = torch.sum(torch.stack(trajectory_contributions, dim=0), dim=0)
        
        # 输出细化
        refined_trajectory = combined_trajectory.transpose(1, 2)  # [batch_size, output_dim, seq_length]
        refined_trajectory = self.output_refinement(refined_trajectory)
        refined_trajectory = refined_trajectory.transpose(1, 2)  # [batch_size, seq_length, output_dim]
        
        return refined_trajectory


class ConsistencyModel(nn.Module):
    """
    一致性模型
    用于快速单步生成
    """
    
    def __init__(self, input_dim: int, condition_dim: int, model_channels: int = 128,
                 num_res_blocks: int = 2, dropout: float = 0.0,
                 sigma_min: float = 0.002, sigma_max: float = 80.0):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # 噪声水平嵌入
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
        
        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
        
        # 主网络
        self.network = nn.Sequential(
            nn.Linear(input_dim + model_channels * 2, model_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(model_channels * 2, model_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(model_channels, input_dim)
        )
        
        # 跳跃连接权重
        self.skip_weight = nn.Parameter(torch.ones(1))
        
    def _c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算跳跃连接权重
        """
        return (sigma ** 2) / (sigma ** 2 + self.sigma_min ** 2)
    
    def _c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算输出权重
        """
        return sigma * self.sigma_min / torch.sqrt(sigma ** 2 + self.sigma_min ** 2)
    
    def _c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算输入权重
        """
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_min ** 2)
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        一致性模型前向传播
        
        Args:
            x: 输入基元 [batch_size, num_primitives, primitive_dim]
            sigma: 噪声水平 [batch_size]
            condition: 条件信息 [batch_size, condition_dim]
            
        Returns:
            预测的清洁基元 [batch_size, num_primitives, primitive_dim]
        """
        batch_size, num_primitives, primitive_dim = x.shape
        
        # 展平基元
        x_flat = x.view(batch_size, -1)  # [batch_size, num_primitives * primitive_dim]
        
        # 计算权重
        c_skip = self._c_skip(sigma).view(-1, 1)
        c_out = self._c_out(sigma).view(-1, 1)
        c_in = self._c_in(sigma).view(-1, 1)
        
        # 输入缩放
        x_scaled = c_in * x_flat
        
        # 嵌入
        sigma_emb = self.sigma_embed(sigma.unsqueeze(-1))
        condition_emb = self.condition_embed(condition)
        
        # 连接所有输入
        network_input = torch.cat([x_scaled, sigma_emb, condition_emb], dim=-1)
        
        # 网络预测
        network_output = self.network(network_input)
        
        # 一致性模型输出
        output = c_skip * x_flat + c_out * network_output
        
        # 重新整形
        output = output.view(batch_size, num_primitives, primitive_dim)
        
        return output
    
    def generate_single_step(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        单步生成（一致性蒸馏）
        
        Args:
            noise: 初始噪声 [batch_size, num_primitives, primitive_dim]
            condition: 条件信息 [batch_size, condition_dim]
            
        Returns:
            生成的基元 [batch_size, num_primitives, primitive_dim]
        """
        batch_size = noise.shape[0]
        device = noise.device
        
        # 使用最大噪声水平
        sigma = torch.full((batch_size,), self.sigma_max, device=device)
        
        # 单步预测
        clean_primitives = self.forward(noise, sigma, condition)
        
        return clean_primitives
    
    def generate_multi_step(self, noise: torch.Tensor, condition: torch.Tensor, 
                          num_steps: int) -> torch.Tensor:
        """
        多步生成
        
        Args:
            noise: 初始噪声 [batch_size, num_primitives, primitive_dim]
            condition: 条件信息 [batch_size, condition_dim]
            num_steps: 生成步数
            
        Returns:
            生成的基元 [batch_size, num_primitives, primitive_dim]
        """
        batch_size = noise.shape[0]
        device = noise.device
        
        # 创建时间步序列
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_steps, device=device)
        
        x = noise
        
        for sigma in sigmas:
            sigma_batch = torch.full((batch_size,), sigma, device=device)
            x = self.forward(x, sigma_batch, condition)
        
        return x


class AdaptiveFRMDModel(FRMDTrajectoryModel):
    """
    自适应FRMD模型
    根据轨迹复杂度自动调整基元数量和一致性步数
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adaptive_primitives = config.get('adaptive_primitives', True)
        self.min_primitives = config.get('min_primitives', 5)
        self.max_primitives = config.get('max_primitives', 20)
        
        # 复杂度估计网络
        if self.adaptive_primitives:
            self.complexity_estimator = nn.Sequential(
                nn.Linear(self.input_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # 基元数量 + 一致性步数
                nn.Sigmoid()
            )
    
    def _estimate_complexity(self, start_pose: torch.Tensor, end_pose: torch.Tensor) -> Tuple[int, int]:
        """
        估计轨迹复杂度并确定基元数量和步数
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            
        Returns:
            基元数量和一致性步数
        """
        if not self.adaptive_primitives:
            return self.num_primitives, self.consistency_steps
        
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        complexity_scores = self.complexity_estimator(combined_pose).mean(dim=0)
        
        # 基元数量
        num_primitives = int(
            self.min_primitives + complexity_scores[0].item() * (self.max_primitives - self.min_primitives)
        )
        
        # 一致性步数
        consistency_steps = max(1, int(complexity_scores[1].item() * 10))
        
        return num_primitives, consistency_steps


class HierarchicalFRMDModel(FRMDTrajectoryModel):
    """
    分层FRMD模型
    使用多层次的运动基元表示
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_hierarchy_levels = config.get('num_hierarchy_levels', 2)
        self.level_primitives = config.get('level_primitives', [5, 10])
        self.level_primitive_dims = config.get('level_primitive_dims', [16, 32])
        
        # 多层次的编码器和解码器
        self.level_encoders = nn.ModuleList()
        self.level_decoders = nn.ModuleList()
        self.level_consistency_models = nn.ModuleList()
        
        for i, (num_prims, prim_dim) in enumerate(zip(self.level_primitives, self.level_primitive_dims)):
            # 编码器
            encoder = MovementPrimitiveEncoder(
                input_dim=self.output_dim,
                primitive_dim=prim_dim,
                num_primitives=num_prims,
                seq_length=self.max_seq_length
            )
            self.level_encoders.append(encoder)
            
            # 解码器
            decoder = MovementPrimitiveDecoder(
                primitive_dim=prim_dim,
                output_dim=self.output_dim,
                num_primitives=num_prims,
                seq_length=self.max_seq_length
            )
            self.level_decoders.append(decoder)
            
            # 一致性模型
            consistency_model = ConsistencyModel(
                input_dim=prim_dim,
                condition_dim=self.input_dim * 2,
                model_channels=64,
                num_res_blocks=1,
                dropout=self.dropout,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max
            )
            self.level_consistency_models.append(consistency_model)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        分层前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件
        combined_condition = torch.cat([start_pose, end_pose], dim=-1)
        condition_embedding = self.condition_encoder(combined_condition)
        
        prev_trajectory = None
        
        # 逐层生成
        for level, (encoder, decoder, consistency_model, num_prims, prim_dim) in enumerate(
            zip(self.level_encoders, self.level_decoders, self.level_consistency_models,
                self.level_primitives, self.level_primitive_dims)
        ):
            # 初始化噪声基元
            noise_primitives = torch.randn(batch_size, num_prims, prim_dim, device=device)
            
            # 一致性生成
            if self.use_consistency_distillation:
                clean_primitives = consistency_model.generate_single_step(
                    noise_primitives, condition_embedding
                )
            else:
                clean_primitives = consistency_model.generate_multi_step(
                    noise_primitives, condition_embedding, self.consistency_steps
                )
            
            # 解码为轨迹
            level_trajectory = decoder(clean_primitives, condition_embedding)
            
            # 层级融合
            if prev_trajectory is not None:
                # 简单平均融合
                level_trajectory = (level_trajectory + prev_trajectory) / 2
            
            # 强制边界条件
            level_trajectory = self._enforce_boundary_conditions(
                level_trajectory, start_pose, end_pose
            )
            
            prev_trajectory = level_trajectory
        
        return prev_trajectory