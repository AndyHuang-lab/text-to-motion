"""
VAE model for text-conditioned motion generation.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=50, dropout=0.1, use_sinusoidal=True):
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

class VAE(nn.Module):
    def __init__(
        self,
        nfeats, 
        embed_dim,
        n_head,
        d_ff,
        num_layers,
        latent_dim,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()

        self.input_dim = nfeats
        self.output_dim = nfeats
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        #########################################################
        # Encoder
        #########################################################

        # Input Embedding for encoder
        self.input_embedding = nn.Linear(nfeats, embed_dim)

        # Position Encoding for encoder (max_len needs to accommodate history + future + 2 CLS tokens)
        # Default to 100 to be safe for most configurations
        self.position_encoder = PositionalEncoding(embed_dim, max_len=100, dropout=dropout)

        # Global Motion Token -> mu and logvar
        self.cls_token = nn.Parameter(torch.randn(2, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,      
            batch_first=False,          
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )
        self.encoder_latent_proj = nn.Linear(embed_dim, latent_dim)

        #########################################################
        # Decoder
        #########################################################

        # Position Encoding for decoder (max_len needs to accommodate history + future + 1 latent token)
        self.position_decoder = PositionalEncoding(embed_dim, max_len=100, dropout=dropout)

        # Latent Projection for decoder
        self.decoder_latent_proj = nn.Linear(latent_dim, embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=True,
        )
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            norm=decoder_norm,
        )

        self.final_layer = nn.Linear(embed_dim, nfeats)

        #########################################################
        # Register Buffer
        #########################################################

        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))
    
    def encode(self, history_motion, future_motion):
        """
        history_motion:[batch_size, history_len, nfeats]
        future_motion:[batch_size, future_len, nfeats]

        Returns:
            latent: [batch_size, latent_dim]
            dist: distribution of latent
        """

        batch_size = history_motion.shape[0]

        # concat history and future motion -> [batch_size, history_len + future_len, nfeats]
        motion = torch.cat((history_motion, future_motion), dim=1)

        # embed motion -> [batch_size, seq_len, embed_dim]
        embed_motion = self.input_embedding(motion)

        # [seq_len, batch_size, embed_dim]
        embed_motion = embed_motion.permute(1, 0, 2)

        # [2, batch_size, embed_dim]  
        cls_tokens = self.cls_token.unsqueeze(1).expand(-1, batch_size, -1)

        # [2 + seq_len, batch_size, embed_dim]
        src_seq = torch.cat((cls_tokens, embed_motion), dim=0)

        # add position encoding 
        src_seq = self.position_encoder(src_seq)

        # Transformer Encoder -> [2 + seq_len, batch_size, embed_dim]
        encoder_output = self.encoder(src_seq)

        distribution_tokens = encoder_output[:cls_tokens.shape[0]]

        # [2, batch_size, latent_dim]
        distribution_parameters = self.encoder_latent_proj(distribution_tokens)
        mu = distribution_parameters[0, ...]
        logvar = distribution_parameters[1, ...]

        logvar = torch.clamp(logvar, min=-10, max=10)
        std = logvar.exp().pow(0.5)

        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        return latent, dist

    def decode(self, latent, history_motion, future_len):
        """
        latent: [batch_size, latent_dim]
        history_motion: [batch_size, history_len, nfeats]
        future_len: int

        Returns:
            future_motion: [batch_size, future_len, nfeats]
        """

        batch_size = history_motion.shape[0]
        device = next(self.parameters()).device

        # [1, batch_size, embed_dim]
        latent_embedding = self.decoder_latent_proj(latent).unsqueeze(0)

        # [batch_size, history_len, embed_dim]
        history_embedding = self.input_embedding(history_motion)

        # [history_len, batch_size, embed_dim]
        history_embedding = history_embedding.permute(1, 0, 2)

        queries = torch.zeros(future_len, batch_size, self.embed_dim, device=device)

        # [1 + history_len + future_len, batch_size, embed_dim]
        tar_seq = torch.cat((latent_embedding, history_embedding, queries), dim=0)

        tar_seq = self.position_decoder(tar_seq)
        
        # [1 + history_len + future_len, batch_size, embed_dim]
        decoder_output = self.decoder(tar_seq)

        future_output = decoder_output[-future_len:]

        # [future_len, batch_size, nfeats]
        future_output = self.final_layer(future_output)

        pred_motion = future_output.permute(1, 0, 2)

        return pred_motion

    def forward(self, z, history_motion, future_len):
        return self.decode(z, history_motion, future_len)
