"""
BERT: Bidirectional Encoder Representations from Transformers
Applied to trajectory generation with bidirectional context modeling
Based on "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseTrajectoryModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
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
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
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


class BERTLayer(nn.Module):
    """Single BERT transformer layer"""
    
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
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attention_weights


class BERTModel(BaseTrajectoryModel):
    """
    BERT-based trajectory generation model
    
    Uses bidirectional attention to model trajectory sequences,
    allowing each point to attend to all other points in the sequence.
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
        
        # BERT layers
        self.layers = nn.ModuleList([
            BERTLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.input_dim)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def create_padding_mask(self, x, lengths=None):
        """Create padding mask for variable length sequences"""
        if lengths is None:
            return None
        
        batch_size, max_len = x.size(0), x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(
            batch_size, max_len) < lengths.unsqueeze(1)
        
        return mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    
    def forward(self, x, mask=None):
        """
        Forward pass
        Args:
            x: [batch_size, sequence_length, input_dim]
            mask: Optional attention mask
        """
        # Input embedding and positional encoding
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout_layer(x)
        
        # Pass through BERT layers
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def generate_trajectory(self, start_pose, end_pose, num_points=None):
        """
        Generate trajectory using BERT model
        
        For generation, we use a masked language model approach:
        1. Create a sequence with start and end poses
        2. Mask intermediate points
        3. Use BERT to predict the masked points
        """
        if num_points is None:
            num_points = self.max_length
        
        device = next(self.parameters()).device
        
        # Create initial sequence with start and end poses
        trajectory = torch.zeros(1, num_points, self.input_dim, device=device)
        
        if start_pose is not None:
            trajectory[0, 0] = torch.tensor(start_pose, device=device, dtype=trajectory.dtype)
        if end_pose is not None:
            trajectory[0, -1] = torch.tensor(end_pose, device=device, dtype=trajectory.dtype)
        
        # Create mask for intermediate points (to be predicted)
        mask_indices = torch.arange(1, num_points - 1, device=device)
        
        # Iterative refinement (similar to masked language modeling)
        with torch.no_grad():
            for iteration in range(5):  # Multiple iterations for refinement
                # Mask some intermediate points
                if iteration < 4:
                    # Randomly mask some points for iterative refinement
                    num_mask = len(mask_indices) // 2
                    masked_indices = mask_indices[torch.randperm(len(mask_indices))[:num_mask]]
                    trajectory[0, masked_indices] = 0
                
                # Forward pass
                output = self.forward(trajectory)
                
                # Update masked positions
                if iteration < 4:
                    trajectory[0, masked_indices] = output[0, masked_indices]
                else:
                    # Final iteration: update all intermediate points
                    trajectory[0, 1:-1] = output[0, 1:-1]
        
        return trajectory.squeeze(0)
    
    def compute_loss(self, batch):
        """Compute training loss using masked trajectory modeling"""
        trajectories = batch['trajectories']
        batch_size, seq_len, _ = trajectories.shape
        
        # Create random masks for training (similar to BERT pretraining)
        mask_prob = 0.15
        mask = torch.rand(batch_size, seq_len) < mask_prob
        
        # Don't mask start and end points
        mask[:, 0] = False
        mask[:, -1] = False
        
        # Create input with masked positions
        input_trajectories = trajectories.clone()
        input_trajectories[mask] = 0  # Zero out masked positions
        
        # Forward pass
        predictions = self.forward(input_trajectories)
        
        # Compute loss only on masked positions
        loss = self.criterion(predictions[mask], trajectories[mask])
        
        return loss
    
    def train_step(self, batch):
        """Single training step"""
        loss = self.compute_loss(batch)
        return loss