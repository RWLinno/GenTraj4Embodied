"""
Conditional Transformer for Trajectory Generation
条件Transformer轨迹生成模型

支持起点/终点条件的Transformer模型，使用交叉注意力机制融合条件信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple
import logging

from .base_model import BaseTrajectoryModel


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class ConditionalMultiHeadAttention(nn.Module):
    """条件多头注意力机制"""
    
    def __init__(self, d_model: int, nhead: int, condition_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 交叉注意力（用于条件融合）
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 条件投影
        self.condition_proj = nn.Linear(condition_dim, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            condition: [batch_size, condition_dim]
            src_mask: 注意力掩码
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影条件到模型维度
        condition_proj = self.condition_proj(condition)  # [batch_size, d_model]
        condition_proj = condition_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + attn_output)
        
        # 交叉注意力（与条件）
        cross_attn_output, _ = self.cross_attention(x, condition_proj, condition_proj)
        x = self.norm2(x + cross_attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x


class ConditionalTransformerModel(BaseTrajectoryModel):
    """条件Transformer模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型配置
        self.d_model = config['architecture']['d_model']
        self.nhead = config['architecture']['nhead']
        self.num_layers = config['architecture']['num_layers']
        self.dim_feedforward = config['architecture']['dim_feedforward']
        self.dropout = config['architecture']['dropout']
        self.max_seq_length = config['architecture']['max_seq_length']
        self.condition_dim = config['architecture']['condition_dim']  # start_pose + end_pose = 14
        self.cross_attention = config['architecture'].get('cross_attention', True)
        
        # 数据维度
        self.trajectory_length = config.get('trajectory_length', 50)
        self.pose_dim = 7  # 3D position + quaternion
        
        # 构建网络
        self._build_networks()
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """构建网络"""
        
        # 输入投影
        self.input_projection = nn.Linear(self.pose_dim, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.condition_dim)
        )
        
        # Transformer层
        if self.cross_attention:
            # 使用条件交叉注意力
            self.transformer_layers = nn.ModuleList([
                ConditionalMultiHeadAttention(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    condition_dim=self.condition_dim,
                    dropout=self.dropout
                ) for _ in range(self.num_layers)
            ])
        else:
            # 标准Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
            
            # 条件融合层
            self.condition_fusion = nn.Sequential(
                nn.Linear(self.d_model + self.condition_dim, self.d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.pose_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果掩码（用于自回归生成）"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, trajectory: torch.Tensor, condition: torch.Tensor, 
                use_causal_mask: bool = False) -> torch.Tensor:
        """前向传播
        
        Args:
            trajectory: [batch_size, seq_len, pose_dim]
            condition: [batch_size, condition_dim]
            use_causal_mask: 是否使用因果掩码
            
        Returns:
            output: [batch_size, seq_len, pose_dim]
        """
        batch_size, seq_len, _ = trajectory.shape
        device = trajectory.device
        
        # 输入投影
        x = self.input_projection(trajectory)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 编码条件
        condition_encoded = self.condition_encoder(condition)
        
        # 创建掩码
        src_mask = None
        if use_causal_mask:
            src_mask = self.create_causal_mask(seq_len, device)
        
        # Transformer处理
        if self.cross_attention:
            # 使用条件交叉注意力
            for layer in self.transformer_layers:
                x = layer(x, condition_encoded, src_mask)
        else:
            # 标准Transformer + 条件融合
            x = self.transformer(x, src_mask)
            
            # 条件融合
            condition_expanded = condition_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, condition_expanded], dim=-1)
            x = self.condition_fusion(x)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失函数"""
        trajectory = batch['trajectory']
        condition = batch['condition']
        
        # 前向传播
        pred_trajectory = self.forward(trajectory, condition)
        
        # 重构损失
        recon_loss = self.mse_loss(pred_trajectory, trajectory)
        
        # 平滑性损失（可选）
        velocity = trajectory[:, 1:] - trajectory[:, :-1]
        pred_velocity = pred_trajectory[:, 1:] - pred_trajectory[:, :-1]
        smooth_loss = self.mse_loss(pred_velocity, velocity)
        
        # 总损失
        total_loss = recon_loss + 0.1 * smooth_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'smooth_loss': smooth_loss
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
            'smooth_loss': losses['smooth_loss'].item()
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
            
            # 自回归生成
            trajectories = self._autoregressive_generation(condition, start_pose, end_pose)
            
            return trajectories.cpu().numpy()
    
    def _autoregressive_generation(self, condition: torch.Tensor, 
                                 start_pose: np.ndarray, end_pose: np.ndarray) -> torch.Tensor:
        """自回归生成轨迹"""
        batch_size = condition.shape[0]
        device = condition.device
        
        # 初始化轨迹
        trajectories = torch.zeros(batch_size, self.trajectory_length, self.pose_dim).to(device)
        
        # 设置起始位姿
        trajectories[:, 0] = torch.FloatTensor(start_pose).to(device)
        
        # 自回归生成
        for t in range(1, self.trajectory_length):
            # 使用已生成的部分作为输入
            input_traj = trajectories[:, :t]
            
            # 前向传播
            pred_traj = self.forward(input_traj, condition, use_causal_mask=True)
            
            # 取最后一个时间步的预测
            next_pose = pred_traj[:, -1]
            
            # 添加一些随机性
            if t < self.trajectory_length - 1:  # 不在最后一步添加噪声
                noise = torch.randn_like(next_pose) * 0.01
                next_pose = next_pose + noise
            
            trajectories[:, t] = next_pose
        
        # 强制设置终点
        trajectories[:, -1] = torch.FloatTensor(end_pose).to(device)
        
        # 平滑处理
        trajectories = self._smooth_trajectory(trajectories)
        
        return trajectories
    
    def _smooth_trajectory(self, trajectories: torch.Tensor) -> torch.Tensor:
        """平滑轨迹"""
        # 简单的移动平均平滑
        window_size = 3
        if self.trajectory_length < window_size:
            return trajectories
        
        smoothed = trajectories.clone()
        
        for i in range(1, self.trajectory_length - 1):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.trajectory_length, i + window_size // 2 + 1)
            
            # 对位置进行平滑
            smoothed[:, i, :3] = torch.mean(trajectories[:, start_idx:end_idx, :3], dim=1)
            
            # 四元数归一化
            quat = smoothed[:, i, 3:7]
            quat_norm = torch.norm(quat, dim=1, keepdim=True)
            smoothed[:, i, 3:7] = quat / (quat_norm + 1e-8)
        
        return smoothed
    
    def generate_with_waypoints(self, start_pose: np.ndarray, end_pose: np.ndarray,
                              waypoints: List[np.ndarray], waypoint_times: List[int]) -> np.ndarray:
        """根据路径点生成轨迹"""
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 准备条件
            condition = np.concatenate([start_pose, end_pose])
            condition = torch.FloatTensor(condition).unsqueeze(0).to(device)
            
            # 初始化轨迹
            trajectory = torch.zeros(1, self.trajectory_length, self.pose_dim).to(device)
            
            # 设置关键点
            trajectory[0, 0] = torch.FloatTensor(start_pose).to(device)
            trajectory[0, -1] = torch.FloatTensor(end_pose).to(device)
            
            for waypoint, time_idx in zip(waypoints, waypoint_times):
                if 0 < time_idx < self.trajectory_length:
                    trajectory[0, time_idx] = torch.FloatTensor(waypoint).to(device)
            
            # 分段生成
            known_indices = [0] + waypoint_times + [self.trajectory_length - 1]
            known_indices = sorted(list(set(known_indices)))
            
            for i in range(len(known_indices) - 1):
                start_idx = known_indices[i]
                end_idx = known_indices[i + 1]
                
                if end_idx - start_idx > 1:
                    # 生成中间部分
                    segment_length = end_idx - start_idx + 1
                    segment_input = trajectory[:, start_idx:end_idx+1]
                    
                    # 使用模型预测中间部分
                    pred_segment = self.forward(segment_input, condition)
                    trajectory[:, start_idx+1:end_idx] = pred_segment[:, 1:-1]
            
            # 平滑处理
            trajectory = self._smooth_trajectory(trajectory)
            
            return trajectory.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ConditionalTransformer',
            'model_type': 'Sequential Modeling',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'condition_dim': self.condition_dim,
            'cross_attention': self.cross_attention,
            'trajectory_length': self.trajectory_length,
            'pose_dim': self.pose_dim,
            'supports_conditional_generation': True,
            'supports_autoregressive_generation': True,
            'supports_waypoint_generation': True
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
def create_conditional_transformer_model(config: Dict[str, Any]) -> ConditionalTransformerModel:
    """创建条件Transformer模型的工厂函数"""
    return ConditionalTransformerModel(config)


# 模型注册
if __name__ == "__main__":
    # 测试代码
    config = {
        'architecture': {
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'max_seq_length': 100,
            'condition_dim': 14,
            'cross_attention': True
        },
        'trajectory_length': 50
    }
    
    model = ConditionalTransformerModel(config)
    print("条件Transformer模型创建成功!")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试生成
    start_pose = np.array([0, 0, 0.5, 0, 0, 0, 1])
    end_pose = np.array([1, 1, 1.0, 0, 0, 0, 1])
    
    trajectories = model.generate_trajectory(start_pose, end_pose, num_samples=2)
    print(f"生成轨迹形状: {trajectories.shape}")
    print("条件Transformer模型测试完成!")