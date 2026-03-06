"""
Flow Matching model for text-conditioned motion generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps."""

    def __init__(self, embed_dim, max_len=100, dropout=0.1, use_sinusoidal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.use_sinusoidal = use_sinusoidal
        self.dropout = nn.Dropout(p=dropout)

        if use_sinusoidal:
            pe = torch.zeros(max_len, 1, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Parameter(torch.empty(max_len, 1, embed_dim))
            nn.init.uniform_(self.pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, embed_dim]
        """
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """
    Embed continuous timestep t in [0, 1] using sinusoidal encoding.

    Uses π-scaled frequencies optimized for unit interval,
    following Rectified Flow and modern diffusion model practices.
    """

    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim or embed_dim * 4

        # Pre-compute frequencies (π-scaled for [0,1] interval)
        half_dim = embed_dim // 2
        # Frequencies: [π, 2π, 3π, ..., half_dim * π]
        self.register_buffer(
            'frequencies',
            torch.pi * torch.arange(1, half_dim + 1)
        )

        # MLP to project embedding (DiT-style)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, timesteps):
        """
        Args:
            timesteps: [batch_size] - values in [0, 1]

        Returns:
            [batch_size, embed_dim]
        """

        # Sinusoidal encoding: sin(π * t * i), cos(π * t * i)
        # t: [batch, 1], frequencies: [half_dim] -> [batch, half_dim]
        emb = timesteps[:, None] * self.frequencies[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Pad if embed_dim is odd (shouldn't happen with typical dims)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.time_embed(emb)


class FlowMatchingTransformer(nn.Module):
    """
    Flow matching model for text-conditioned motion generation.
    Predicts target latent z (x_1) instead of velocity.

    Training:
        - Sample t ~ Uniform(0, 1)
        - Interpolate: x_t = (1 - t) * x_0 + t * x_1 (noise -> data)
        - Predict z: pred_z = model(x_t, t, history, text)
        - Loss: MSE(pred_z, x_1)

    Sampling (Euler):
        - x = noise ~ N(0, I)
        - for t in [0, dt, 2dt, ..., 1-dt]:
            pred_z = model(x, t, history, text)
            v = (pred_z - x) / (1 - t)  # velocity from z prediction
            x = x + dt * v
        - Decode x with VAE

    Input:
        - x_t: interpolated latent at time t [batch_size, 1, latent_dim]
        - timesteps: continuous timestep in [0, 1] [batch_size]
        - history_motion: [batch_size, history_len, history_dim]
        - text: CLIP text embedding [batch_size, clip_dim]
        - all_mask: if True, mask all text (for classifier-free guidance)

    Output:
        - pred_z: predicted target latent x_1 [batch_size, 1, latent_dim]
    """

    def __init__(
        self,
        embed_dim,
        d_ff,
        n_head,
        num_layers,
        clip_dim,
        history_dim, 
        latent_dim,
        mask_prob,
        dropout=0.1,
        activation='gelu',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.d_ff = d_ff
        self.n_head = n_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        self.history_dim = history_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.mask_prob = mask_prob

        # Input embeddings
        # Note: max_len=100 is sufficient for timesteps in [0, 1]
        self.position_encoder = PositionalEncoding(self.embed_dim, max_len=100, dropout=0.0)

        self.timestep_embedder = TimestepEmbedder(self.embed_dim)

        self.text_embedder = nn.Linear(self.clip_dim, self.embed_dim)
        self.history_embedder = nn.Linear(self.history_dim, self.embed_dim)
        self.latent_embedder = nn.Linear(self.latent_dim, self.embed_dim)

        # Transformer encoder
        TransformerLayer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_head,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=False, 
            norm_first=True,
        )

        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(
            TransformerLayer,
            num_layers=self.num_layers,
            norm=encoder_norm,
        )

        # Output projection: predict target latent z (x_1)
        self.output_proj = nn.Linear(self.embed_dim, self.latent_dim)

    def mask_text(self, text, all_mask=False):
        """Mask text embedding for classifier-free guidance."""
        batch_size = text.shape[0]

        if all_mask:
            return torch.zeros_like(text)
        elif self.training and self.mask_prob > 0:
            mask = torch.bernoulli(
                torch.ones(batch_size, device=text.device) * self.mask_prob
            ).view(batch_size, 1)
            return text * (1.0 - mask)
        else:
            return text

    def forward(self, x_t, timesteps, history_motion, text, all_mask=False):
        """
        Forward pass of flow matching model.

        Args:
            x_t: Interpolated latent at time t [batch_size, 1, latent_dim]
            timesteps: Continuous timestep in [0, 1] [batch_size]
            history_motion: History motion [batch_size, history_len, history_dim]
            text: CLIP text embedding [batch_size, clip_dim]
            all_mask: Mask text for classifier-free guidance

        Returns:
            pred_z: Predicted target latent x_1 [batch_size, 1, latent_dim]
        """
        # Embed timestep: [batch_size, embed_dim] -> [1, batch_size, embed_dim]
        time_embed = self.timestep_embedder(timesteps).unsqueeze(0)

        # Embed history: [batch_size, history_len, history_dim]
        # -> [history_len, batch_size, embed_dim]
        history_embed = self.history_embedder(history_motion).permute(1, 0, 2)

        # Embed latent: [batch_size, 1, latent_dim] -> [1, batch_size, embed_dim]
        latent_embed = self.latent_embedder(x_t.permute(1, 0, 2))

        # Embed text: [batch_size, clip_dim] -> [1, batch_size, embed_dim]
        text_embed = self.text_embedder(self.mask_text(text, all_mask)).unsqueeze(0)

        # Concatenate all embeddings
        # [1 + 1 + history_len + 1, batch_size, embed_dim]
        src_seq = torch.cat((time_embed, text_embed, history_embed, latent_embed), dim=0)

        # Add positional encoding
        src_seq = self.position_encoder(src_seq)

        # Transform
        output = self.encoder(src_seq)

        # Extract last token (latent) and predict target z (x_1)
        pred_z = self.output_proj(output[-1])  # (batch_size, latent_dim)

        # Add sequence dimension for consistency
        return pred_z.unsqueeze(1)  # (batch_size, 1, latent_dim)
