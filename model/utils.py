import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer Encoder Layer
    Structure: Norm -> Attention -> Residual -> Norm -> FFN -> Residual
    """
    def __init__(
        self,
        d_model, 
        n_head,
        d_ff,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
 
        self.dropout1 = nn.Dropout(dropout) # Attention dropout
        self.dropout2 = nn.Dropout(dropout) # FFN dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):

        # Pre-norm attention
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(
            src_norm, # query
            src_norm, # key
            src_norm, # value
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(attn_output)

        # Pre-norm FFN
        src_norm = self.norm2(src)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(src_norm))))
        src = src + ff_output

        return src