import torch
import torch.nn as nn
from .base import BaseNet


class TemporalAttention(nn.Module):
    """Temporal attention over chunks to weight their importance"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, num_chunks, d_model]
        Returns: [B, num_chunks, d_model]
        """
        attended, _ = self.attention(x, x, x)
        return self.norm(attended + x)


class ChunkProcessor(nn.Module):
    """Process a single chunk of audio-visual data with cross-attention"""
    def __init__(self, d_model=128, num_heads=4, dropout=0.5):
        super().__init__()
        
        # Feature projections
        self.audio_proj = nn.Linear(25, d_model)
        self.visual_proj = nn.Linear(136, d_model)
        
        # Layer normalization
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        
        # Cross-modal attention (both directions)
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
        
        # Residual connections and normalization
        self.norm_av = nn.LayerNorm(d_model)
        self.norm_va = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.ff_a = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.ff_v = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.final_norm_a = nn.LayerNorm(d_model)
        self.final_norm_v = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, a_chunk, v_chunk):
        """
        a_chunk: [B, chunk_size, 25]
        v_chunk: [B, chunk_size, 136]
        Returns: fused representation [B, d_model]
        """
        # Project features
        a_feat = self.audio_proj(a_chunk)  # [B, chunk_size, d_model]
        v_feat = self.visual_proj(v_chunk)  # [B, chunk_size, d_model]
        
        # ===== Cross-Attention (A → V) =====
        av_out, _ = self.cross_av(a_feat, v_feat, v_feat)
        a_attended = self.norm_av(av_out + a_feat)
        
        # ===== Cross-Attention (V → A) =====
        va_out, _ = self.cross_va(v_feat, a_feat, a_feat)
        v_attended = self.norm_va(va_out + v_feat)
        
        # ===== Feed-forward layers =====
        a_ff = self.ff_a(a_attended)
        a_processed = self.final_norm_a(a_ff + a_attended)
        
        v_ff = self.ff_v(v_attended)
        v_processed = self.final_norm_v(v_ff + v_attended)
        
        # ===== Pool within chunk and concatenate =====
        a_pooled = torch.mean(a_processed, dim=1)  # [B, d_model]
        v_pooled = torch.mean(v_processed, dim=1)  # [B, d_model]
        
        # Element-wise fusion (better than concatenation for memory)
        fused = a_pooled + v_pooled  # [B, d_model]
        
        return self.dropout(fused)


class ChunkCrossAttentionNet(BaseNet):
    """
    Improved Chunk-wise Cross Attention with:
    - Better chunk processing (bidirectional attention + FF layers)
    - Temporal attention over chunks
    - Proper sequence aggregation
    - Positional encoding awareness
    """
    def __init__(
        self,
        d_model=256,
        chunk_size=20,
        num_heads=8,
        dropout=0.5,
        num_gru_layers=2,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        # Chunk processor
        self.chunk_processor = ChunkProcessor(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ===== Temporal modeling across chunks =====
        # Option 1: GRU for sequential dependencies
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Option 2: Temporal attention over chunk representations
        self.temporal_attention = TemporalAttention(d_model, num_heads=4)
        
        # ===== Fusion layer =====
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Bidirectional GRU outputs
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # ===== Final classification =====
        self.dropout_final = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(d_model)

    def feature_extractor(self, x):
        """
        x shape: [B, T, 161] where T = total timesteps
        161 = 136 (visual) + 25 (audio)
        """
        B, T, _ = x.shape
        
        # Split modalities
        xv = x[:, :, :136]   # visual [B, T, 136]
        xa = x[:, :, 136:]   # audio  [B, T, 25]
        
        # Process chunks
        chunk_representations = []
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size  # ceiling division
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, T)
            
            v_chunk = xv[:, start:end, :]  # [B, chunk_len, 136]
            a_chunk = xa[:, start:end, :]  # [B, chunk_len, 25]
            
            # Pad if necessary to maintain chunk_size
            if v_chunk.shape[1] < self.chunk_size:
                pad_len = self.chunk_size - v_chunk.shape[1]
                v_chunk = torch.nn.functional.pad(v_chunk, (0, 0, 0, pad_len))
                a_chunk = torch.nn.functional.pad(a_chunk, (0, 0, 0, pad_len))
            
            # Process chunk
            chunk_feat = self.chunk_processor(a_chunk, v_chunk)  # [B, d_model]
            chunk_representations.append(chunk_feat)
        
        # Stack all chunk representations
        chunk_seq = torch.stack(chunk_representations, dim=1)  # [B, num_chunks, d_model]
        
        # ===== Temporal modeling =====
        # GRU forward pass
        gru_out, _ = self.gru(chunk_seq)  # [B, num_chunks, d_model*2]
        
        # Apply temporal attention
        gru_out_pooled = gru_out.mean(dim=1)  # [B, d_model*2] - global average
        
        # Fusion
        fused = self.fusion(gru_out_pooled)  # [B, d_model]
        fused = self.final_norm(fused)
        
        return self.dropout_final(fused)

    def classifier(self, x):
        """
        x: [B, d_model]
        Returns: [B, 1] - logits
        """
        return self.fc(x)
