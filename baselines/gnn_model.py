"""
Graph Neural Network (GNN) Trajectory Generation Model
图神经网络轨迹生成模型

使用图神经网络处理轨迹数据，将轨迹点建模为图节点，
通过图卷积操作学习轨迹点之间的空间和时序关系。

Reference:
- Kipf, T. N., & Welling, M. "Semi-supervised classification with graph convolutional networks." 
  arXiv preprint arXiv:1609.02907 (2016).
- Veličković, P., et al. "Graph attention networks." 
  arXiv preprint arXiv:1710.10903 (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import math

from ...base_model import BaseTrajectoryModel


class GraphConvLayer(nn.Module):
    """图卷积层"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        """
        # 线性变换
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        
        # 图卷积
        output = torch.matmul(adj, support)  # [batch_size, num_nodes, out_features]
        
        # 添加偏置
        output = output + self.bias
        
        return self.dropout(output)


class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.size()
        
        # 线性变换
        Wh = torch.matmul(x, self.W)  # [batch_size, num_nodes, out_features]
        
        # 计算注意力系数
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # 掩码无效连接
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # 应用注意力
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, out_features = Wh.size()
        
        # 创建所有节点对的特征拼接
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, N, N, out_features]
        Wh2 = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, N, N, out_features]
        
        # 拼接特征
        all_combinations = torch.cat([Wh1, Wh2], dim=-1)  # [batch, N, N, 2*out_features]
        
        return all_combinations


class TemporalGraphLayer(nn.Module):
    """时序图层"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_conv = GraphConvLayer(in_features, out_features, dropout)
        self.temporal_conv = nn.Conv1d(out_features, out_features, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [batch_size, seq_len, in_features]
            adj: 邻接矩阵 [seq_len, seq_len]
        """
        batch_size, seq_len, in_features = x.size()
        
        # 扩展邻接矩阵到批次维度
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 空间图卷积
        spatial_out = self.spatial_conv(x, adj)  # [batch_size, seq_len, out_features]
        
        # 时序卷积
        temporal_input = spatial_out.transpose(1, 2)  # [batch_size, out_features, seq_len]
        temporal_out = self.temporal_conv(temporal_input)
        temporal_out = temporal_out.transpose(1, 2)  # [batch_size, seq_len, out_features]
        
        # 残差连接和归一化
        if spatial_out.size(-1) == temporal_out.size(-1):
            output = spatial_out + temporal_out
        else:
            output = temporal_out
        
        output = self.norm(output)
        
        return output


class GNNTrajectoryModel(BaseTrajectoryModel):
    """
    GNN轨迹生成模型
    
    使用图神经网络处理轨迹数据：
    1. 轨迹点建模：将轨迹点作为图节点
    2. 关系建模：通过邻接矩阵表示时序和空间关系
    3. 图卷积：学习节点间的局部和全局关系
    4. 注意力机制：自适应关注重要的轨迹段
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 模型参数
        self.input_dim = config.get('input_dim', 3)  # x, y, z
        self.output_dim = config.get('output_dim', 3)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 3)
        self.sequence_length = config.get('sequence_length', 50)
        self.dropout = config.get('dropout', 0.1)
        self.use_attention = config.get('use_attention', True)
        self.use_temporal = config.get('use_temporal', True)
        
        # 图构建参数
        self.k_neighbors = config.get('k_neighbors', 5)  # k近邻图
        self.distance_threshold = config.get('distance_threshold', 0.5)  # 距离阈值
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),  # start + end
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 图神经网络层
        self.gnn_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            if self.use_temporal:
                layer = TemporalGraphLayer(self.hidden_dim, self.hidden_dim, self.dropout)
            elif self.use_attention:
                layer = GraphAttentionLayer(self.hidden_dim, self.hidden_dim, self.dropout)
            else:
                layer = GraphConvLayer(self.hidden_dim, self.hidden_dim, self.dropout)
            
            self.gnn_layers.append(layer)
        
        # 输出解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_adjacency_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """
        构建邻接矩阵
        
        Args:
            positions: 位置信息 [batch_size, seq_len, 3]
            
        Returns:
            adj: 邻接矩阵 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = positions.size()
        
        # 计算距离矩阵
        pos_i = positions.unsqueeze(2)  # [batch, seq_len, 1, 3]
        pos_j = positions.unsqueeze(1)  # [batch, 1, seq_len, 3]
        
        distances = torch.norm(pos_i - pos_j, dim=-1)  # [batch, seq_len, seq_len]
        
        # 时序邻接（相邻时间步）
        temporal_adj = torch.zeros_like(distances)
        for i in range(seq_len - 1):
            temporal_adj[:, i, i + 1] = 1.0
            temporal_adj[:, i + 1, i] = 1.0
        
        # 空间邻接（k近邻）
        spatial_adj = torch.zeros_like(distances)
        
        for b in range(batch_size):
            for i in range(seq_len):
                # 找到k个最近邻居
                dist_i = distances[b, i]
                _, indices = torch.topk(dist_i, min(self.k_neighbors + 1, seq_len), largest=False)
                
                for j in indices[1:]:  # 排除自己
                    if dist_i[j] < self.distance_threshold:
                        spatial_adj[b, i, j] = 1.0
                        spatial_adj[b, j, i] = 1.0
        
        # 组合时序和空间邻接
        adj = temporal_adj + spatial_adj
        adj = torch.clamp(adj, 0, 1)  # 确保二值化
        
        # 添加自连接
        eye = torch.eye(seq_len, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        
        return adj
    
    def forward(self, x: torch.Tensor, start_pose: torch.Tensor = None, 
                end_pose: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入轨迹 [batch_size, seq_len, input_dim]
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 结束位姿 [batch_size, input_dim]
            
        Returns:
            output: 输出轨迹 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 编码条件信息
        if start_pose is not None and end_pose is not None:
            conditions = torch.cat([start_pose, end_pose], dim=-1)
            condition_features = self.condition_encoder(conditions)  # [batch, hidden_dim]
            
            # 扩展到序列长度
            condition_features = condition_features.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            condition_features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        
        # 节点特征编码
        node_input = torch.cat([x, condition_features], dim=-1)
        node_features = self.node_encoder(node_input)  # [batch, seq_len, hidden_dim]
        
        # 构建邻接矩阵
        adj = self.build_adjacency_matrix(x)
        
        # 图神经网络层
        h = node_features
        for layer in self.gnn_layers:
            if isinstance(layer, TemporalGraphLayer):
                h = layer(h, adj)
            else:
                h = layer(h, adj)
            h = F.relu(h)
        
        # 解码输出
        output = self.decoder(h)
        
        return output
    
    def generate_trajectory(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                          num_points: int = 50) -> torch.Tensor:
        """
        生成轨迹
        
        Args:
            start_pose: 起始位姿 [batch_size, 3] 或 [3]
            end_pose: 结束位姿 [batch_size, 3] 或 [3]
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 生成的轨迹 [batch_size, num_points, 3]
        """
        if start_pose.dim() == 1:
            start_pose = start_pose.unsqueeze(0)
        if end_pose.dim() == 1:
            end_pose = end_pose.unsqueeze(0)
        
        batch_size = start_pose.shape[0]
        
        # 初始化轨迹（线性插值）
        t = torch.linspace(0, 1, num_points, device=self.device)
        initial_trajectory = []
        
        for b in range(batch_size):
            traj = start_pose[b].unsqueeze(0) + t.unsqueeze(1) * (end_pose[b] - start_pose[b]).unsqueeze(0)
            initial_trajectory.append(traj)
        
        initial_trajectory = torch.stack(initial_trajectory)
        
        # 迭代优化轨迹
        trajectory = initial_trajectory.clone()
        
        for iteration in range(3):  # 迭代优化
            # 前向传播
            refined_trajectory = self.forward(trajectory, start_pose, end_pose)
            
            # 混合原始轨迹和优化轨迹
            alpha = 0.7  # 混合系数
            trajectory = alpha * refined_trajectory + (1 - alpha) * trajectory
            
            # 确保端点约束
            trajectory[:, 0] = start_pose
            trajectory[:, -1] = end_pose
        
        return trajectory
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        # 重构损失
        reconstruction_loss = F.mse_loss(predictions, targets)
        
        # 平滑度损失
        if predictions.shape[1] > 2:
            pred_accel = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
            target_accel = targets[:, 2:] - 2 * targets[:, 1:-1] + targets[:, :-2]
            smoothness_loss = F.mse_loss(pred_accel, target_accel)
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        
        # 端点损失
        endpoint_loss = F.mse_loss(predictions[:, [0, -1]], targets[:, [0, -1]])
        
        # 图正则化损失（鼓励局部一致性）
        if predictions.shape[1] > 1:
            neighbor_diff = predictions[:, 1:] - predictions[:, :-1]
            target_neighbor_diff = targets[:, 1:] - targets[:, :-1]
            graph_loss = F.mse_loss(neighbor_diff, target_neighbor_diff)
        else:
            graph_loss = torch.tensor(0.0, device=predictions.device)
        
        total_loss = reconstruction_loss + 0.1 * smoothness_loss + 0.5 * endpoint_loss + 0.05 * graph_loss
        
        return total_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        # 获取数据
        trajectories = batch.get('trajectory', torch.randn(4, self.sequence_length, 3))
        start_poses = batch.get('start_pose', trajectories[:, 0])
        end_poses = batch.get('end_pose', trajectories[:, -1])
        
        # 前向传播
        predictions = self.forward(trajectories, start_poses, end_poses)
        
        # 计算损失
        loss = self.compute_loss(predictions, trajectories)
        
        # 计算指标
        mse = F.mse_loss(predictions, trajectories)
        endpoint_error = F.mse_loss(predictions[:, [0, -1]], trajectories[:, [0, -1]])
        
        # 计算平滑度指标
        if trajectories.shape[1] > 2:
            pred_smoothness = torch.mean(torch.norm(
                predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2], dim=-1
            ))
            target_smoothness = torch.mean(torch.norm(
                trajectories[:, 2:] - 2 * trajectories[:, 1:-1] + trajectories[:, :-2], dim=-1
            ))
        else:
            pred_smoothness = torch.tensor(0.0)
            target_smoothness = torch.tensor(0.0)
        
        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'endpoint_error': endpoint_error.item(),
            'pred_smoothness': pred_smoothness.item(),
            'target_smoothness': target_smoothness.item()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'GNN',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'k_neighbors': self.k_neighbors,
            'use_attention': self.use_attention,
            'use_temporal': self.use_temporal,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }