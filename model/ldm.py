import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=20, dropout=0.1, use_sinusoidal=True,):
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
    def __init__(self, embed_dim, position_encoder):
        super().__init__()
        self.embed_dim = embed_dim
        self.position_encoder = position_encoder

        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
    
    def forward(self, timesteps):
        """
        timesteps: [batch_size]
        """
        # [1, batch_size, embed_dim]
        return self.time_embed(self.position_encoder.pe[timesteps]).permute(1, 0, 2)

class DenoiserTransformer(nn.Module):
    """
    Input:
        - x_t: latent at timestep t [batch_size, 1, latent_dim]
        - timesteps: diffusion timestep [batch_size]
        - history_motion: [batch_size, history_len, history_dim]
        - text: CLIP text embedding [batch_size, clip_dim]
        - all_mask: if True, mask all text embeddings (for classifier-free guidance)
    Output:
        - pred_latent: [batch_size, 1, latent_dim]
    """
    def __init__(
        self,
        embed_dim,
        d_ff,
        n_head,
        num_layers,
        dropout,
        activation,
        clip_dim,
        history_dim,
        latent_dim,
        mask_prob,
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
        
        # input embeddings
        # max_len needs to accommodate timesteps (e.g., 1000)
        self.position_encoder = PositionalEncoding(self.embed_dim, max_len=1000)
        
        self.timestep_embedder = TimestepEmbedder(self.embed_dim, self.position_encoder)

        self.text_embedder = nn.Linear(self.clip_dim, self.embed_dim)

        self.history_embedder = nn.Linear(self.history_dim, self.embed_dim)

        self.latent_embedder = nn.Linear(self.latent_dim, self.embed_dim)

        TransformerLayer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_head,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.encoder = nn.TransformerEncoder(
            TransformerLayer,
            num_layers=self.num_layers,
        )

        self.output_proj = nn.Linear(self.embed_dim, self.latent_dim)

    def mask_text(self, text, all_mask=False):
        batch_size, clip_dim = text.shape

        if all_mask:
            return torch.zeros_like(text)
        elif self.training and self.mask_prob>0:
            mask = torch.bernoulli(
                torch.ones(batch_size, device=text.device) * self.mask_prob).view(batch_size, 1)
            return text * (1. - mask)
        else:
            return text

    def forward(self, x_t, timesteps, history_motion, text, all_mask):
        # [1, batch_size, embed_dim]
        time_embed = self.timestep_embedder(timesteps)

        # [history_len, batch_size, embed_dim]
        history_embed = self.history_embedder(history_motion.permute(1, 0, 2))

        # [1, batch_size, embed_dim]
        latent_embed = self.latent_embedder(x_t.permute(1, 0, 2))

        # [1, batch_size, embed_dim]
        text_embed = self.text_embedder(self.mask_text(text, all_mask)).unsqueeze(0)

        src_seq = torch.cat((time_embed, text_embed, history_embed, latent_embed), dim=0)

        src_seq = self.position_encoder(src_seq)

        output = self.encoder(src_seq)

        pred_latent = self.output_proj(output[-1])  # (batch_size, latent_dim)

        # Add sequence dimension for consistency
        return pred_latent.unsqueeze(1)  # (batch_size, 1, latent_dim)