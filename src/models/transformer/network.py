"""Transformer-based Demand Predictor Network.

This module implements a transformer architecture for multi-horizon demand
prediction with uncertainty estimation. It takes enhanced states from the
Information Sharing Network and outputs demand predictions with confidence
intervals.

References:
    [1] Vaswani et al. (2017) - "Attention is all you need"
    [2] Salinas et al. (2020) - "DeepAR: Probabilistic forecasting with autoregressive recurrent networks"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from config.transformer_config import TransformerConfig


class PositionalEncoding(nn.Module):
    """Implement the PE function from the original transformer paper."""
    
    def __init__(self, d_model: int, max_len: int = 24):
        super().__init__()
        
        # Create positional encoding matrix
        self.d_model = d_model
        self.max_len = max_len
        self._create_encoding()
        
    def _create_encoding(self):
        """Create positional encoding matrix."""
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) *
            (-math.log(10000.0) / self.d_model)
        )
        
        pe = torch.zeros(1, self.max_len, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # Dynamically extend positional encoding if needed
            old_max_len = self.max_len
            self.max_len = seq_len
            self._create_encoding()
            print(f"Warning: Extended positional encoding from {old_max_len} to {seq_len}")
            
        return x + self.pe[:, :seq_len]


class OutputProjection(nn.Module):
    """Project transformer output to mean and standard deviation."""
    
    def __init__(self, d_model: int, output_dim: int):
        """Initialize output projection.
        
        Args:
            d_model: Model dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.mean_proj = nn.Linear(d_model, output_dim)
        self.std_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of mean and standard deviation tensors [batch_size, seq_len, output_dim]
        """
        mean = self.mean_proj(x)  # [B, seq_len, output_dim]
        std = torch.exp(self.std_proj(x)) * 1.5  # Increase std by 50% to be more conservative
        return mean, std


class DemandPredictor(nn.Module):
    """Transformer-based demand predictor with uncertainty estimation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # Output projection
        if config.uncertainty_type == 'probabilistic':
            self.output_proj = OutputProjection(config.d_model, config.output_dim)
        else:
            self.output_proj = nn.Linear(config.d_model, config.output_dim)
        
        # Store attention weights
        self.attention_weights = None
        
        # Initialize parameters
        self._init_parameters()
        
        # Move model to device
        self.to(self.device)
    
    def _init_parameters(self):
        """Initialize network parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _get_attention_weights(self):
        """Get attention weights from the last forward pass."""
        return self.attention_weights
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate masking tensor for autoregressive generation."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            src: Source sequence [batch_size, src_len, input_dim]
            tgt: Optional target sequence for teacher forcing [batch_size, tgt_len]
            tgt_mask: Optional target attention mask
            src_key_padding_mask: Optional source key padding mask
            tgt_key_padding_mask: Optional target key padding mask
            
        Returns:
            Tuple of predictions and uncertainties [batch_size, seq_len, output_dim]
        """
        # Project and encode source sequence
        src_proj = self.input_proj(src)  # [B, src_len, d_model]
        src_proj = self.pos_encoder(src_proj)
        memory = self.transformer_encoder(src_proj, None, src_key_padding_mask)
        
        # Generate sequence
        if tgt is None:
            return self._generate_sequence(memory, src_key_padding_mask)
        
        # Teacher forcing: prepare target sequence
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(-1)  # Add feature dimension
        tgt = tgt.repeat(1, 1, self.config.input_dim)  # Repeat across input dimensions
        
        # Project and encode target
        tgt_proj = self.input_proj(tgt)  # [B, tgt_len, d_model]
        tgt_proj = self.pos_encoder(tgt_proj)
        
        output = self.transformer_decoder(
            tgt_proj, memory,
            tgt_mask, None,
            tgt_key_padding_mask, src_key_padding_mask
        )
        
        # Project to output space
        if self.config.uncertainty_type == 'probabilistic':
            mean, std = self.output_proj(output)
            return mean, std  # [B, seq_len, output_dim]
        else:
            pred = self.output_proj(output)
            return pred, None  # [B, seq_len, output_dim]
    
    def _generate_sequence(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate sequence autoregressively during inference.
        
        Args:
            memory: Encoded source sequence [batch_size, src_len, d_model]
            memory_key_padding_mask: Optional key padding mask
            
        Returns:
            Tuple of predictions and uncertainties [batch_size, horizon, output_dim]
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Initialize target sequence with zeros
        tgt = torch.zeros(batch_size, 1, self.config.input_dim, device=device)  # [B, 1, input_dim]
        
        means, stds = [], []
        for i in range(self.config.forecast_horizon):
            # Create mask for autoregressive generation
            tgt_mask = self._generate_square_subsequent_mask(i + 1).to(device)
            
            # Project and encode target
            tgt_proj = self.input_proj(tgt)  # [B, seq_len, d_model]
            tgt_proj = self.pos_encoder(tgt_proj)
            
            # Generate next prediction
            output = self.transformer_decoder(
                tgt_proj, memory,
                tgt_mask, None,
                None, memory_key_padding_mask
            )
            
            # Project to output space
            if self.config.uncertainty_type == 'probabilistic':
                mean, std = self.output_proj(output)  # [B, seq_len, output_dim]
                means.append(mean[:, -1:])  # Keep only the last prediction
                stds.append(std[:, -1:])
                
                # Prepare next input: use the last prediction as input
                next_input = mean[:, -1:, 0].unsqueeze(-1).repeat(1, 1, self.config.input_dim)  # [B, 1, input_dim]
                tgt = torch.cat([tgt, next_input], dim=1)
            else:
                pred = self.output_proj(output)  # [B, seq_len, output_dim]
                means.append(pred[:, -1:])  # Keep only the last prediction
                
                # Prepare next input: use the last prediction as input
                next_input = pred[:, -1:, 0].unsqueeze(-1).repeat(1, 1, self.config.input_dim)  # [B, 1, input_dim]
                tgt = torch.cat([tgt, next_input], dim=1)
        
        # Stack predictions
        means = torch.cat(means, dim=1)  # [B, horizon, output_dim]
        if self.config.uncertainty_type == 'probabilistic':
            stds = torch.cat(stds, dim=1)  # [B, horizon, output_dim]
            return means, stds
        else:
            return means, None
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
