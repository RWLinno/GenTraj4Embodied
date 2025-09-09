"""
RNN Model for Trajectory Generation
RNN轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import LinearArchitectureModel


class RNNTrajectoryModel(LinearArchitectureModel):
    """
    基础RNN轨迹生成模型
    使用标准RNN进行序列到序列的轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rnn_type = config.get('rnn_type', 'RNN')  # 'RNN', 'Elman', 'Jordan'
        self.activation = config.get('activation', 'tanh')  # 'tanh', 'relu'
        self.use_bias = config.get('use_bias', True)
        self.bidirectional = config.get('bidirectional', False)
        
        # 输入编码器
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.Tanh() if self.activation == 'tanh' else nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # RNN层
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                nonlinearity=self.activation,
                bias=self.use_bias,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'Elman':
            # Elman网络实现
            self.rnn = ElmanRNN(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
        else:  # Jordan
            # Jordan网络实现
            self.rnn = JordanRNN(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                output_size=self.output_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        
        # 输出投影
        rnn_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.output_projection = nn.Sequential(
            nn.Linear(rnn_output_dim, self.hidden_dim),
            nn.Tanh() if self.activation == 'tanh' else nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # 时间位置编码
        self.position_encoding = nn.Parameter(
            torch.randn(self.max_seq_length, self.hidden_dim) * 0.1
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码输入
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        encoded_input = self.input_encoder(combined_pose)  # [batch_size, hidden_dim]
        
        # 创建序列输入
        input_sequence = encoded_input.unsqueeze(1).expand(-1, self.max_seq_length, -1)
        
        # 添加位置编码
        input_sequence = input_sequence + self.position_encoding.unsqueeze(0)
        
        # RNN前向传播
        if isinstance(self.rnn, (ElmanRNN, JordanRNN)):
            rnn_output = self.rnn(input_sequence)
        else:
            rnn_output, _ = self.rnn(input_sequence)
        
        # 输出投影
        trajectory = self.output_projection(rnn_output)
        
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
    
    def generate_trajectory(self, start_pose: np.ndarray, end_pose: np.ndarray,
                          num_points: int = 50, **kwargs) -> np.ndarray:
        """
        生成轨迹的推理接口
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
                    **kwargs) -> torch.Tensor:
        """
        计算损失函数
        """
        # 基础重建损失
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # 梯度惩罚（防止梯度爆炸）
        gradient_penalty_weight = self.config.get('gradient_penalty_weight', 0.01)
        if gradient_penalty_weight > 0:
            # 计算轨迹梯度
            pred_grad = torch.diff(predictions, dim=1)
            grad_penalty = torch.mean(torch.clamp(torch.norm(pred_grad, dim=-1) - 1.0, min=0) ** 2)
            
            total_loss = mse_loss + gradient_penalty_weight * grad_penalty
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'rnn_type': self.rnn_type,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'bidirectional': self.bidirectional,
            'model_category': 'Linear Architecture'
        })
        return info


class ElmanRNN(nn.Module):
    """
    Elman循环神经网络实现
    特点：隐藏状态反馈到输入层
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Elman网络层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            layer = ElmanLayer(layer_input_size, hidden_size)
            self.layers.append(layer)
            
            if bidirectional:
                backward_layer = ElmanLayer(layer_input_size, hidden_size)
                self.layers.append(backward_layer)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_seq: 输入序列 [batch_size, seq_length, input_size]
            
        Returns:
            输出序列 [batch_size, seq_length, hidden_size * (2 if bidirectional else 1)]
        """
        batch_size, seq_length, _ = input_seq.shape
        
        outputs = []
        
        for layer_idx in range(0, len(self.layers), 2 if self.bidirectional else 1):
            forward_layer = self.layers[layer_idx]
            
            # 前向处理
            forward_output = forward_layer(input_seq)
            
            if self.bidirectional:
                # 反向处理
                backward_layer = self.layers[layer_idx + 1]
                reversed_input = torch.flip(input_seq, dims=[1])
                backward_output = backward_layer(reversed_input)
                backward_output = torch.flip(backward_output, dims=[1])
                
                # 连接前向和反向输出
                layer_output = torch.cat([forward_output, backward_output], dim=-1)
            else:
                layer_output = forward_output
            
            if self.dropout is not None:
                layer_output = self.dropout(layer_output)
            
            input_seq = layer_output  # 为下一层准备输入
            outputs.append(layer_output)
        
        return outputs[-1]


class ElmanLayer(nn.Module):
    """
    单层Elman网络
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入到隐藏层的权重
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        # 隐藏层到隐藏层的权重（循环连接）
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 激活函数
        self.activation = nn.Tanh()
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        batch_size, seq_length, _ = input_seq.shape
        device = input_seq.device
        
        # 初始化隐藏状态
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        
        outputs = []
        
        for t in range(seq_length):
            # Elman网络更新规则
            hidden = self.activation(
                self.W_ih(input_seq[:, t, :]) + self.W_hh(hidden)
            )
            outputs.append(hidden)
        
        return torch.stack(outputs, dim=1)


class JordanRNN(nn.Module):
    """
    Jordan循环神经网络实现
    特点：输出反馈到输入层
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Jordan网络层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size + output_size if i == 0 else hidden_size + output_size
            layer = JordanLayer(layer_input_size, hidden_size, output_size)
            self.layers.append(layer)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        batch_size, seq_length, _ = input_seq.shape
        device = input_seq.device
        
        # 初始化输出反馈
        prev_output = torch.zeros(batch_size, self.output_size, device=device)
        
        outputs = []
        
        for t in range(seq_length):
            # 连接输入和前一时刻的输出
            layer_input = torch.cat([input_seq[:, t, :], prev_output], dim=-1)
            
            # 通过所有层
            for layer in self.layers:
                layer_output = layer(layer_input.unsqueeze(1))
                layer_input = layer_output.squeeze(1)
                
                if self.dropout is not None:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(layer_input)
            prev_output = layer_input  # 更新输出反馈
        
        return torch.stack(outputs, dim=1)


class JordanLayer(nn.Module):
    """
    单层Jordan网络
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 输入到隐藏层
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        # 隐藏层到输出
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
        # 激活函数
        self.hidden_activation = nn.Tanh()
        self.output_activation = nn.Identity()
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 输入到隐藏层
        hidden = self.hidden_activation(self.input_to_hidden(input_tensor))
        
        # 隐藏层到输出
        output = self.output_activation(self.hidden_to_output(hidden))
        
        return output


class RecurrentTrajectoryModel(LinearArchitectureModel):
    """
    递归轨迹生成模型
    使用自回归方式逐步生成轨迹点
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.autoregressive = config.get('autoregressive', True)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 递归单元
        self.recurrent_cell = nn.LSTMCell(
            input_size=self.output_dim + self.hidden_dim,
            hidden_size=self.hidden_dim
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                target_trajectory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        递归前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码条件信息
        combined_pose = torch.cat([start_pose, end_pose], dim=-1)
        condition = self.encoder(combined_pose)
        
        # 初始化隐藏状态
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        cell = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # 初始化输出
        current_output = start_pose
        outputs = [current_output]
        
        # 递归生成
        for t in range(1, self.max_seq_length):
            # 准备输入
            if self.training and target_trajectory is not None and torch.rand(1).item() < 0.5:
                # Teacher forcing
                recurrent_input = torch.cat([target_trajectory[:, t-1, :], condition], dim=-1)
            else:
                # 使用前一时刻的输出
                recurrent_input = torch.cat([current_output, condition], dim=-1)
            
            # 递归单元更新
            hidden, cell = self.recurrent_cell(recurrent_input, (hidden, cell))
            
            # 生成输出
            current_output = self.output_projection(hidden)
            outputs.append(current_output)
        
        # 强制最后一个点为终点
        outputs[-1] = end_pose
        
        return torch.stack(outputs, dim=1)