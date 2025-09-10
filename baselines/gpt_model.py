"""
GPT: Generative Pre-trained Transformer
Applied to trajectory generation with autoregressive modeling
Based on "Improving Language Understanding by Generative Pre-Training"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseTrajectoryModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking for GPT"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(attention)
        
        return output, attention_weights
    
    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention = torch.matmul(attention_weights, value)
        
        return attention, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GPTBlock(nn.Module):
    """Single GPT transformer block"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection (pre-norm)
        norm_x = self.norm1(x)
        attn_output, attention_weights = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection (pre-norm)
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output
        
        return x, attention_weights


class GPTModel(BaseTrajectoryModel):
    """
    GPT-based trajectory generation model
    
    Uses autoregressive generation to predict trajectory points sequentially,
    where each point can only attend to previous points in the sequence.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Model parameters
        self.input_dim = getattr(config, 'input_dim', 7)
        self.d_model = getattr(config, 'd_model', 512)
        self.num_heads = getattr(config, 'num_heads', 8)
        self.num_layers = getattr(config, 'num_layers', 6)
        self.d_ff = getattr(config, 'd_ff', 2048)
        self.dropout = getattr(config, 'dropout', 0.1)
        self.max_length = getattr(config, 'sequence_length', 64)
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_length)
        
        # GPT blocks
        self.blocks = nn.ModuleList([
            GPTBlock(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.input_dim)
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(self.d_model)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Create causal mask
        self.register_buffer('causal_mask', self._create_causal_mask(self.max_length))
        
    def _create_causal_mask(self, seq_len):
        """Create causal (lower triangular) mask for autoregressive attention"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, x, use_causal_mask=True):
        """
        Forward pass
        Args:
            x: [batch_size, sequence_length, input_dim]
            use_causal_mask: Whether to use causal masking
        """
        batch_size, seq_len, _ = x.shape
        
        # Input embedding and positional encoding
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout_layer(x)
        
        # Prepare causal mask
        mask = None
        if use_causal_mask:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
        
        # Pass through GPT blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def generate_trajectory(self, start_pose, end_pose, num_points=None):
        """
        Generate trajectory using autoregressive sampling
        
        GPT generates trajectories sequentially, predicting each point
        based on all previous points in the sequence.
        """
        if num_points is None:
            num_points = self.max_length
        
        device = next(self.parameters()).device
        
        # Initialize trajectory with start pose
        trajectory = torch.zeros(1, num_points, self.input_dim, device=device)
        
        if start_pose is not None:
            trajectory[0, 0] = torch.tensor(start_pose, device=device, dtype=trajectory.dtype)
        
        # Autoregressive generation
        with torch.no_grad():
            for i in range(1, num_points):
                # Forward pass on current sequence
                output = self.forward(trajectory[:, :i+1], use_causal_mask=True)
                
                # Use the prediction for the current position
                trajectory[0, i] = output[0, i-1]  # Predict next position
        
        # Apply end pose constraint if provided
        if end_pose is not None:
            trajectory[0, -1] = torch.tensor(end_pose, device=device, dtype=trajectory.dtype)
            
            # Optionally blend the trajectory to smoothly reach the end pose
            # Linear interpolation for the last few points
            blend_length = min(5, num_points // 4)
            for i in range(blend_length):
                alpha = (i + 1) / blend_length
                idx = num_points - blend_length + i
                trajectory[0, idx] = (1 - alpha) * trajectory[0, idx] + alpha * trajectory[0, -1]
        
        return trajectory.squeeze(0)
    
    def compute_loss(self, batch):
        """Compute autoregressive training loss"""
        trajectories = batch['trajectories']
        batch_size, seq_len, _ = trajectories.shape
        
        # Input: all points except the last
        input_seq = trajectories[:, :-1]
        
        # Target: all points except the first (shifted by one)
        target_seq = trajectories[:, 1:]
        
        # Forward pass
        predictions = self.forward(input_seq, use_causal_mask=True)
        
        # Compute loss (predict next point given previous points)
        loss = self.criterion(predictions, target_seq)
        
        return loss
    
    def train_step(self, batch):
        """Single training step"""
        loss = self.compute_loss(batch)
        return loss