import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.rec.modules import MLP
from modules.rec.attention import MultiheadAttention, DistMultiheadAttention
from modules.rqvae.normalize import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
        self,
        attn_dim,
        num_heads,
        dropout,
        mlp_hidden_dims=None,
    ):
        super().__init__()

        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attn_norm = RMSNorm(attn_dim)

        # self.attention = nn.MultiheadAttention(
        #     attn_dim,
        #     num_heads,
        #     dropout
        # )

        self.attention = MultiheadAttention(
            attn_dim,
            num_heads,
            dropout
        )

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [attn_dim]

        self.ff = nn.Sequential(
            RMSNorm(attn_dim),
            MLP(
                input_dim=attn_dim,
                hidden_dims=mlp_hidden_dims,
                out_dim=attn_dim,
                dropout=dropout,
                normalize=False
            )
        )

    def forward(
        self,
        x,
        attention_mask
    ):
        x = torch.transpose(x, 0, 1)
        _x = self.attn_norm(x)
        mha_outputs, _ = self.attention(
            _x, _x, _x, attn_mask=attention_mask
        )
        x = x + mha_outputs
        x = torch.transpose(x, 0, 1)
        x = x + self.ff(x)
        return x

class DistTransformerBlock(nn.Module):
    def __init__(
        self,
        attn_dim,
        num_heads,
        dropout,
        mlp_hidden_dims=None
    ):
        super().__init__()

        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.mean_attn_norm = RMSNorm(attn_dim)
        self.cov_attn_norm = RMSNorm(attn_dim)

        self.attention = DistMultiheadAttention(
            attn_dim,
            num_heads,
            dropout
        )

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [attn_dim]

        self.mean_ff = nn.Sequential(
            RMSNorm(attn_dim),
            MLP(
                input_dim=attn_dim,
                hidden_dims=mlp_hidden_dims,
                out_dim=attn_dim,
                dropout=dropout,
                normalize=False
            )
        )
        
        self.cov_ff = nn.Sequential(
            RMSNorm(attn_dim),
            MLP(
                input_dim=attn_dim,
                hidden_dims=mlp_hidden_dims,
                out_dim=attn_dim,
                dropout=dropout,
                normalize=False
            )
        )

    def forward(
        self,
        mean_x,
        cov_x,
        attention_mask
    ):
        mean_x = self.mean_attn_norm(mean_x)
        cov_x = self.cov_attn_norm(cov_x)

        _mean_x, _cov_x, _ = self.attention(
            mean_x, cov_x, attn_mask=attention_mask
        )
        
        mean_x = mean_x + _mean_x
        cov_x = cov_x + _cov_x
        
        mean_x = mean_x + self.mean_ff(mean_x)
        cov_x = cov_x + self.cov_ff(cov_x)
        return mean_x, cov_x

class RecTransformer(nn.Module):
    def __init__(
        self,
        n_layers,
        attn_dim,
        num_heads,
        dropout,
        mlp_hidden_dims=None,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.ModuleList([
            TransformerBlock(
                attn_dim=attn_dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_hidden_dims=mlp_hidden_dims,
            ) for _ in range(n_layers)
        ])

    def forward(
        self, 
        x,
        attention_mask
    ):
        for layer in self.layers:
            x = layer(
                x=x,
                attention_mask=attention_mask
            )
        return x

class DistRecTransformer(nn.Module):
    def __init__(
        self,
        n_layers,
        attn_dim,
        num_heads,
        dropout,
        mlp_hidden_dims=None,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.ModuleList([
            DistTransformerBlock(
                attn_dim=attn_dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_hidden_dims=mlp_hidden_dims,
            ) for _ in range(n_layers)
        ])

    def forward(
        self, 
        mean_x,
        cov_x,
        attention_mask
    ):
        for layer in self.layers:
            mean_x, cov_x = layer(
                mean_x=mean_x,
                cov_x=cov_x,
                attention_mask=attention_mask
            )
        return mean_x, cov_x
