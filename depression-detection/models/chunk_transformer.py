import torch
import torch.nn as nn
from .base import BaseNet


class PositionalEncoding(nn.Module):
    """Positional encoding for chunk sequences (dynamic length)"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        x: [B, num_chunks, d_model]
        Dynamically computes positional encoding for variable sequence lengths
        """
        B, seq_len, d_model = x.shape
        device = x.device
        
        # Compute positional encoding on the fly (no size limitation!)
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
            -(torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add to input
        return x + pe.unsqueeze(0)  # [B, seq_len, d_model]


class MultimodalChunkFusion(nn.Module):
    """Fuse audio and visual information within a chunk"""
    def __init__(self, d_model=256, num_heads=8, dropout=0.5):
        super().__init__()
        
        # Feature projections
        self.audio_proj = nn.Linear(25, d_model)
        self.visual_proj = nn.Linear(136, d_model)
        
        # Self-attention within modality (to capture temporal structure)
        self.audio_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.visual_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Cross-modal attention
        self.cross_av = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.cross_va = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm_a_self = nn.LayerNorm(d_model)
        self.norm_v_self = nn.LayerNorm(d_model)
        self.norm_av = nn.LayerNorm(d_model)
        self.norm_va = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.ff_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ff_v = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.final_norm_a = nn.LayerNorm(d_model)
        self.final_norm_v = nn.LayerNorm(d_model)
        
        # Gating mechanism for fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, a_chunk, v_chunk):
        """
        a_chunk: [B, chunk_size, 25]
        v_chunk: [B, chunk_size, 136]
        Returns: fused representation [B, d_model]
        """
        # Project to d_model
        a_feat = self.audio_proj(a_chunk)  # [B, chunk_size, d_model]
        v_feat = self.visual_proj(v_chunk)  # [B, chunk_size, d_model]
        
        # ===== Intra-modal self-attention =====
        a_self, _ = self.audio_self_attn(a_feat, a_feat, a_feat)
        a_self = self.norm_a_self(a_self + a_feat)
        
        v_self, _ = self.visual_self_attn(v_feat, v_feat, v_feat)
        v_self = self.norm_v_self(v_self + v_feat)
        
        # ===== Cross-modal attention =====
        av_out, _ = self.cross_av(a_self, v_self, v_self)
        a_cross = self.norm_av(av_out + a_self)
        
        va_out, _ = self.cross_va(v_self, a_self, a_self)
        v_cross = self.norm_va(va_out + v_self)
        
        # ===== Feed-forward =====
        a_ff = self.ff_a(a_cross)
        a_final = self.final_norm_a(a_ff + a_cross)
        
        v_ff = self.ff_v(v_cross)
        v_final = self.final_norm_v(v_ff + v_cross)
        
        # ===== Pool within chunk =====
        a_pool = torch.mean(a_final, dim=1)  # [B, d_model]
        v_pool = torch.mean(v_final, dim=1)  # [B, d_model]
        
        # ===== Gated fusion =====
        combined = torch.cat([a_pool, v_pool], dim=-1)  # [B, 2*d_model]
        gate_weights = self.gate(combined)  # [B, d_model]
        
        # Weighted sum
        fused = gate_weights * a_pool + (1 - gate_weights) * v_pool
        
        return self.dropout(fused)


class ChunkTransformerNet(BaseNet):
    """
    Advanced Chunk-wise Cross Attention with Transformer Encoder
    
    Features:
    - Per-chunk multimodal fusion with self-attention
    - Bidirectional temporal modeling with Transformer encoder
    - Positional encoding for chunk sequences
    - Gated fusion mechanism
    - Better gradient flow
    """
    def __init__(
        self,
        d_model=256,
        chunk_size=20,
        num_heads=8,
        num_encoder_layers=3,
        dropout=0.5,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        # Chunk fusion module
        self.chunk_fusion = MultimodalChunkFusion(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Positional encoding for chunks (dynamic, handles any sequence length)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # ===== Temporal Transformer Encoder for chunks =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # ===== Aggregation =====
        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # ===== Classification head =====
        self.dropout_final = nn.Dropout(dropout)
        self.norm_final = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def feature_extractor(self, x):
        """
        x shape: [B, T, 161]
        """
        B, T, _ = x.shape
        
        # Split modalities
        xv = x[:, :, :136]   # visual
        xa = x[:, :, 136:]   # audio
        
        # Process chunks
        chunk_representations = []
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, T)
            
            v_chunk = xv[:, start:end, :]
            a_chunk = xa[:, start:end, :]
            
            # Pad if necessary
            if v_chunk.shape[1] < self.chunk_size:
                pad_len = self.chunk_size - v_chunk.shape[1]
                v_chunk = torch.nn.functional.pad(v_chunk, (0, 0, 0, pad_len))
                a_chunk = torch.nn.functional.pad(a_chunk, (0, 0, 0, pad_len))
            
            # Fuse within chunk
            chunk_feat = self.chunk_fusion(a_chunk, v_chunk)  # [B, d_model]
            chunk_representations.append(chunk_feat)
        
        # Stack chunks
        chunk_seq = torch.stack(chunk_representations, dim=1)  # [B, num_chunks, d_model]
        
        # Add positional encoding
        chunk_seq = self.positional_encoding(chunk_seq)
        
        # ===== Temporal encoding with Transformer =====
        transformer_out = self.transformer_encoder(chunk_seq)  # [B, num_chunks, d_model]
        
        # ===== Attention-based aggregation =====
        # Create a learnable query for aggregation
        global_query = transformer_out[:, 0, :].unsqueeze(1)  # Use first chunk as query
        
        aggregated, _ = self.chunk_attention(
            global_query,
            transformer_out,
            transformer_out
        )  # [B, 1, d_model]
        
        aggregated = aggregated.squeeze(1)  # [B, d_model]
        
        # Also add mean pooling for robustness
        mean_pooled = torch.mean(transformer_out, dim=1)  # [B, d_model]
        
        # Combine both
        final_feat = aggregated + mean_pooled
        final_feat = self.norm_final(final_feat)
        
        return self.dropout_final(final_feat)

    def classifier(self, x):
        return self.fc(x)