"""
Sequence-to-Sequence Model for Trajectory Generation
序列到序列轨迹生成模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from base_model import LinearArchitectureModel


class Seq2SeqTrajectoryModel(LinearArchitectureModel):
    """
    序列到序列轨迹生成模型
    使用编码器-解码器架构进行轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.encoder_type = config.get('encoder_type', 'LSTM')  # 'LSTM', 'GRU', 'RNN'
        self.decoder_type = config.get('decoder_type', 'LSTM')
        self.use_attention = config.get('use_attention', True)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        self.bidirectional_encoder = config.get('bidirectional_encoder', True)
        
        # 编码器
        self.encoder = Encoder(
            input_dim=self.hidden_dim,  # Use hidden_dim instead of input_dim * 2
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            encoder_type=self.encoder_type,
            bidirectional=self.bidirectional_encoder
        )
        
        # 解码器
        encoder_output_dim = self.hidden_dim * (2 if self.bidirectional_encoder else 1)
        self.decoder = Decoder(
            input_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            encoder_output_dim=encoder_output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            decoder_type=self.decoder_type,
            use_attention=self.use_attention
        )
        
        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 输出嵌入
        self.output_embedding = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                target_trajectory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            start_pose: 起始位姿 [batch_size, input_dim]
            end_pose: 终止位姿 [batch_size, input_dim]
            context: 可选上下文信息
            target_trajectory: 目标轨迹（用于teacher forcing）[batch_size, seq_length, output_dim]
            
        Returns:
            生成的轨迹 [batch_size, seq_length, output_dim]
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        # 编码输入
        combined_input = torch.cat([start_pose, end_pose], dim=-1)
        embedded_input = self.input_embedding(combined_input)
        
        # 编码器
        encoder_outputs, encoder_hidden = self.encoder(embedded_input)
        
        # 解码器
        decoder_input = start_pose  # 使用起始位姿作为解码器的初始输入
        decoder_hidden = self._prepare_decoder_hidden(encoder_hidden)
        
        outputs = []
        
        for t in range(self.max_seq_length):
            # 解码一步
            decoder_output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            outputs.append(decoder_output)
            
            # 准备下一步的输入
            if t < self.max_seq_length - 1:
                if (self.training and target_trajectory is not None and 
                    torch.rand(1).item() < self.teacher_forcing_ratio):
                    # Teacher forcing
                    decoder_input = target_trajectory[:, t, :]
                else:
                    # 使用模型输出
                    decoder_input = decoder_output
        
        trajectory = torch.stack(outputs, dim=1)
        
        # 强制边界条件
        trajectory = self._enforce_boundary_conditions(trajectory, start_pose, end_pose)
        
        return trajectory
    
    def _prepare_decoder_hidden(self, encoder_hidden):
        """
        准备解码器的初始隐藏状态
        
        Args:
            encoder_hidden: 编码器的隐藏状态
            
        Returns:
            解码器的初始隐藏状态
        """
        if isinstance(encoder_hidden, tuple):  # LSTM
            h, c = encoder_hidden
            if self.bidirectional_encoder:
                # 双向编码器：连接前向和后向的隐藏状态
                h = torch.cat([h[-2], h[-1]], dim=-1)
                c = torch.cat([c[-2], c[-1]], dim=-1)
                
                # 投影到解码器的隐藏维度
                h = nn.Linear(h.size(-1), self.hidden_dim, device=h.device)(h)
                c = nn.Linear(c.size(-1), self.hidden_dim, device=c.device)(c)
            else:
                h = h[-1]  # 取最后一层
                c = c[-1]
            
            return (h.unsqueeze(0).expand(self.num_layers, -1, -1),
                    c.unsqueeze(0).expand(self.num_layers, -1, -1))
        else:  # GRU or RNN
            h = encoder_hidden
            if self.bidirectional_encoder:
                h = torch.cat([h[-2], h[-1]], dim=-1)
                h = nn.Linear(h.size(-1), self.hidden_dim, device=h.device)(h)
            else:
                h = h[-1]
            
            return h.unsqueeze(0).expand(self.num_layers, -1, -1)
    
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
        
        # 序列一致性损失
        sequence_weight = self.config.get('sequence_weight', 0.1)
        if sequence_weight > 0:
            # 计算相邻时间步的差异
            pred_diff = torch.diff(predictions, dim=1)
            target_diff = torch.diff(targets, dim=1)
            sequence_loss = nn.MSELoss()(pred_diff, target_diff)
            
            total_loss = mse_loss + sequence_weight * sequence_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        info = super().get_model_info()
        info.update({
            'encoder_type': self.encoder_type,
            'decoder_type': self.decoder_type,
            'use_attention': self.use_attention,
            'teacher_forcing_ratio': self.teacher_forcing_ratio,
            'bidirectional_encoder': self.bidirectional_encoder,
            'model_category': 'Linear Architecture'
        })
        return info


class Encoder(nn.Module):
    """
    编码器模块
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.0, encoder_type: str = 'LSTM',
                 bidirectional: bool = True):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # RNN层
        if encoder_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif encoder_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:  # RNN
            self.rnn = nn.RNN(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
    
    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码器前向传播
        
        Args:
            input_tensor: 输入张量 [batch_size, input_dim]
            
        Returns:
            outputs: 所有时间步的输出 [batch_size, seq_length, hidden_dim * directions]
            hidden: 最终隐藏状态
        """
        batch_size = input_tensor.shape[0]
        
        # 投影输入
        projected_input = self.input_projection(input_tensor)  # [batch_size, hidden_dim]
        
        # 扩展为序列（这里我们只有一个时间步的输入）
        sequence_input = projected_input.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # RNN前向传播
        outputs, hidden = self.rnn(sequence_input)
        
        return outputs, hidden


class Decoder(nn.Module):
    """
    解码器模块（带注意力机制）
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 encoder_output_dim: int, num_layers: int = 1,
                 dropout: float = 0.0, decoder_type: str = 'LSTM',
                 use_attention: bool = True):
        super().__init__()
        self.decoder_type = decoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 注意力机制
        if use_attention:
            self.attention = Attention(hidden_dim, encoder_output_dim)
            rnn_input_dim = hidden_dim + encoder_output_dim
        else:
            rnn_input_dim = hidden_dim
        
        # RNN层
        if decoder_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif decoder_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # RNN
            self.rnn = nn.RNN(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        解码器前向传播
        
        Args:
            input_tensor: 当前时间步输入 [batch_size, input_dim]
            hidden: 隐藏状态
            encoder_outputs: 编码器输出 [batch_size, encoder_seq_length, encoder_output_dim]
            
        Returns:
            output: 当前时间步输出 [batch_size, output_dim]
            hidden: 更新后的隐藏状态
            attention_weights: 注意力权重（如果使用注意力）
        """
        # 输入嵌入
        embedded_input = self.input_embedding(input_tensor)  # [batch_size, hidden_dim]
        embedded_input = embedded_input.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        attention_weights = None
        
        # 注意力机制
        if self.use_attention:
            # 获取当前隐藏状态作为查询
            if isinstance(hidden, tuple):  # LSTM
                query = hidden[0][-1].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            else:  # GRU or RNN
                query = hidden[-1].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            context, attention_weights = self.attention(query, encoder_outputs)
            
            # 连接输入和上下文
            rnn_input = torch.cat([embedded_input, context], dim=-1)
        else:
            rnn_input = embedded_input
        
        # RNN前向传播
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        
        # 输出投影
        output = self.output_projection(rnn_output.squeeze(1))  # [batch_size, output_dim]
        
        return output, new_hidden, attention_weights


class Attention(nn.Module):
    """
    注意力机制模块
    """
    
    def __init__(self, query_dim: int, key_dim: int):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, key_dim)
        self.energy_layer = nn.Linear(key_dim, 1, bias=False)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意力计算
        
        Args:
            query: 查询 [batch_size, 1, query_dim]
            keys: 键值 [batch_size, seq_length, key_dim]
            
        Returns:
            context: 上下文向量 [batch_size, 1, key_dim]
            attention_weights: 注意力权重 [batch_size, seq_length]
        """
        # 投影查询到键的维度
        projected_query = self.query_projection(query)  # [batch_size, 1, key_dim]
        
        # 计算能量分数
        # 广播相加
        energy_input = projected_query + keys  # [batch_size, seq_length, key_dim]
        energy_scores = self.energy_layer(torch.tanh(energy_input)).squeeze(-1)  # [batch_size, seq_length]
        
        # 计算注意力权重
        attention_weights = torch.softmax(energy_scores, dim=-1)  # [batch_size, seq_length]
        
        # 计算上下文向量
        context = torch.sum(keys * attention_weights.unsqueeze(-1), dim=1, keepdim=True)  # [batch_size, 1, key_dim]
        
        return context, attention_weights


class BeamSearchSeq2SeqModel(Seq2SeqTrajectoryModel):
    """
    带束搜索的Seq2Seq模型
    支持生成多个候选轨迹
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.beam_size = config.get('beam_size', 3)
        self.length_penalty = config.get('length_penalty', 1.0)
    
    def beam_search_generate(self, start_pose: torch.Tensor, end_pose: torch.Tensor,
                           beam_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用束搜索生成多个候选轨迹
        
        Args:
            start_pose: 起始位姿 [1, input_dim]
            end_pose: 终止位姿 [1, input_dim]
            beam_size: 束大小
            
        Returns:
            trajectories: 候选轨迹 [beam_size, seq_length, output_dim]
            scores: 轨迹分数 [beam_size]
        """
        if beam_size is None:
            beam_size = self.beam_size
        
        device = start_pose.device
        
        # 编码
        combined_input = torch.cat([start_pose, end_pose], dim=-1)
        embedded_input = self.input_embedding(combined_input)
        encoder_outputs, encoder_hidden = self.encoder(embedded_input)
        
        # 初始化束
        beams = [(torch.zeros(1, 0, self.output_dim, device=device), 0.0, encoder_hidden)]
        
        for t in range(self.max_seq_length):
            candidates = []
            
            for trajectory, score, hidden in beams:
                if t == 0:
                    decoder_input = start_pose
                else:
                    decoder_input = trajectory[:, -1, :]
                
                # 解码一步
                decoder_output, new_hidden, _ = self.decoder(
                    decoder_input, hidden, encoder_outputs
                )
                
                # 计算概率分布（这里简化为使用输出值）
                log_probs = -torch.norm(decoder_output, dim=-1)  # 简化的分数计算
                
                # 扩展轨迹
                new_trajectory = torch.cat([trajectory, decoder_output.unsqueeze(1)], dim=1)
                new_score = score + log_probs.item()
                
                candidates.append((new_trajectory, new_score, new_hidden))
            
            # 选择top-k候选
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
        
        # 提取结果
        trajectories = torch.cat([beam[0] for beam in beams], dim=0)
        scores = torch.tensor([beam[1] for beam in beams], device=device)
        
        return trajectories, scores


class HierarchicalSeq2SeqModel(Seq2SeqTrajectoryModel):
    """
    分层Seq2Seq模型
    使用多层次的编码器-解码器进行轨迹生成
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_levels = config.get('num_levels', 2)
        self.level_seq_lengths = config.get('level_seq_lengths', [10, 50])
        
        # 多层次的编码器-解码器
        self.level_encoders = nn.ModuleList()
        self.level_decoders = nn.ModuleList()
        
        for i, seq_len in enumerate(self.level_seq_lengths):
            # 编码器
            encoder_input_dim = self.input_dim * 2 if i == 0 else self.input_dim * 2 + self.output_dim
            encoder = Encoder(
                input_dim=encoder_input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                encoder_type=self.encoder_type,
                bidirectional=self.bidirectional_encoder
            )
            self.level_encoders.append(encoder)
            
            # 解码器
            encoder_output_dim = self.hidden_dim * (2 if self.bidirectional_encoder else 1)
            decoder = Decoder(
                input_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                encoder_output_dim=encoder_output_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                decoder_type=self.decoder_type,
                use_attention=self.use_attention
            )
            self.level_decoders.append(decoder)
    
    def forward(self, start_pose: torch.Tensor, end_pose: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        分层前向传播
        """
        batch_size = start_pose.shape[0]
        device = start_pose.device
        
        prev_trajectory = None
        
        for level, (seq_len, encoder, decoder) in enumerate(
            zip(self.level_seq_lengths, self.level_encoders, self.level_decoders)
        ):
            # 准备编码器输入
            if level == 0:
                encoder_input = torch.cat([start_pose, end_pose], dim=-1)
            else:
                # 使用前一层的轨迹信息
                prev_summary = torch.mean(prev_trajectory, dim=1)  # 简单平均作为摘要
                encoder_input = torch.cat([start_pose, end_pose, prev_summary], dim=-1)
            
            # 编码
            embedded_input = self.input_embedding(encoder_input)
            encoder_outputs, encoder_hidden = encoder(embedded_input)
            
            # 解码
            decoder_input = start_pose
            decoder_hidden = self._prepare_decoder_hidden(encoder_hidden)
            
            outputs = []
            original_seq_length = self.max_seq_length
            self.max_seq_length = seq_len
            
            for t in range(seq_len):
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                outputs.append(decoder_output)
                decoder_input = decoder_output
            
            self.max_seq_length = original_seq_length
            level_trajectory = torch.stack(outputs, dim=1)
            
            # 强制边界条件
            level_trajectory = self._enforce_boundary_conditions(level_trajectory, start_pose, end_pose)
            
            prev_trajectory = level_trajectory
        
        return prev_trajectory