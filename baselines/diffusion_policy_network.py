"""
Diffusion network architectures for trajectory generation
扩散网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DiffusionUNet(nn.Module):
    """扩散U-Net网络"""
    
    def __init__(self, 
                 input_dim: int = 7,
                 condition_dim: int = 14,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 time_embed_dim: int = 128,
                 dropout: float = 0.1):
        """
        初始化扩散U-Net
        
        Args:
            input_dim: 输入维度（动作维度）
            condition_dim: 条件维度
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            time_embed_dim: 时间嵌入维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # 条件嵌入
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(
                ResidualBlock(
                    hidden_dim, 
                    hidden_dim, 
                    time_embed_dim,
                    dropout=dropout
                )
            )
        
        # 中间层
        self.middle_block = ResidualBlock(
            hidden_dim, 
            hidden_dim, 
            time_embed_dim,
            dropout=dropout
        )
        
        # 解码器层
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                ResidualBlock(
                    hidden_dim * 2,  # 跳跃连接
                    hidden_dim, 
                    time_embed_dim,
                    dropout=dropout
                )
            )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor, 
                timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入轨迹 [batch_size, seq_len, input_dim]
            condition: 条件 [batch_size, condition_dim]
            timesteps: 时间步 [batch_size]
            
        Returns:
            输出轨迹 [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)  # [batch_size, time_embed_dim]
        
        # 条件嵌入
        cond_emb = self.condition_embedding(condition)  # [batch_size, hidden_dim]
        
        # 输入投影
        x = x.reshape(batch_size * seq_len, -1)
        x = self.input_projection(x)  # [batch_size * seq_len, hidden_dim]
        x = x.reshape(batch_size, seq_len, -1)
        
        # 添加条件信息
        x = x + cond_emb.unsqueeze(1)  # 广播条件到所有时间步
        
        # 编码器
        encoder_outputs = []
        for layer in self.encoder_layers:
            x = layer(x, time_emb)
            encoder_outputs.append(x)
        
        # 中间层
        x = self.middle_block(x, time_emb)
        
        # 解码器（带跳跃连接）
        for i, layer in enumerate(self.decoder_layers):
            # 跳跃连接
            skip_connection = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip_connection], dim=-1)
            x = layer(x, time_emb)
        
        # 输出投影
        x = x.reshape(batch_size * seq_len, -1)
        x = self.output_projection(x)
        x = x.reshape(batch_size, seq_len, -1)
        
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 time_embed_dim: int,
                 dropout: float = 0.1):
        """
        初始化残差块
        
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            time_embed_dim: 时间嵌入维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 时间投影
        self.time_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_embed_dim, out_dim)
        )
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
        # 残差连接
        if in_dim != out_dim:
            self.residual_projection = nn.Linear(in_dim, out_dim)
        else:
            self.residual_projection = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, in_dim]
            time_emb: 时间嵌入 [batch_size, time_embed_dim]
            
        Returns:
            输出张量 [batch_size, seq_len, out_dim]
        """
        residual = self.residual_projection(x)
        
        # 第一个卷积
        h = self.conv1(x)
        
        # 添加时间嵌入
        time_proj = self.time_projection(time_emb)
        h = h + time_proj.unsqueeze(1)  # 广播到序列维度
        
        # 第二个卷积
        h = self.conv2(h)
        
        return h + residual


class TimeEmbedding(nn.Module):
    """时间嵌入模块"""
    
    def __init__(self, embed_dim: int):
        """
        初始化时间嵌入
        
        Args:
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 正弦位置编码后的维度就是embed_dim，所以linear1的输入应该是embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            timesteps: 时间步 [batch_size]
            
        Returns:
            时间嵌入 [batch_size, embed_dim]
        """
        # 正弦位置编码
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 处理奇数维度的情况
        if emb.shape[-1] != self.embed_dim:
            # 如果维度不匹配，截断或填充
            if emb.shape[-1] > self.embed_dim:
                emb = emb[:, :self.embed_dim]
            else:
                padding = torch.zeros(emb.shape[0], self.embed_dim - emb.shape[-1], device=emb.device)
                emb = torch.cat([emb, padding], dim=-1)
        
        # 线性变换
        emb = self.linear1(emb)
        emb = F.relu(emb)
        emb = self.linear2(emb)
        
        return emb
