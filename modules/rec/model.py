import os
import sys
import torch


import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from modules.rec.embedder import UserIdEmbedder, SemIdEmbedder, SemIdPosition
from modules.rec.transformer import RecTransformer, DistRecTransformer
from modules.rec.utils import wasserstein_distance_matmul
from modules.rqvae.normalize import RMSNorm

from modules.utils import eval_mode

from einops import rearrange

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

class Recommender(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_ids_dim,
        num_items,
        num_users,
        max_pos = 2048,
        ids_embeddings = -1,
        mlp_hidden_dims = None,
        token_embedding_dim = None,
    ):
        super().__init__()

        # INIT
        self.embedding_dim = embedding_dim
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.num_embeddings = num_embeddings
        self.sem_ids_dim = sem_ids_dim
        self.max_pos = max_pos
        self.ids_embeddings = ids_embeddings
        self.enable_generation = False

        # INIT EMBEDDER
        if ids_embeddings == -1:
            ids_embeddings = num_embeddings
        
        self.user_id_embedder = nn.Embedding(
            num_embeddings=num_users + 1,
            embedding_dim=attn_dim,
            padding_idx=num_users
        )

        if token_embedding_dim is None:
            token_embedding_dim = embedding_dim // sem_ids_dim

        self.token_embedding_dim = token_embedding_dim

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=ids_embeddings,
            sem_ids_dim=sem_ids_dim,
            embedding_dim=token_embedding_dim
        )

        self.token_pos_embedder = SemIdPosition(
            sem_ids_dim=sem_ids_dim,
            embedding_dim=embedding_dim // sem_ids_dim,
            max_pos=max_pos + 10,
        )

        self.pos_emb = nn.Embedding(
            num_embeddings=max_pos + 10, 
            embedding_dim=embedding_dim
        )

        self.items_embedder = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=num_items
        )

        # INIT TRANSFORMER
        self.in_proj = nn.Linear(embedding_dim + token_embedding_dim * sem_ids_dim, attn_dim, bias=True)
        self.transformer = RecTransformer(
            n_layers=n_layers,
            attn_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_hidden_dims=mlp_hidden_dims
        )

        self.out_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(attn_dim, embedding_dim, bias=False)
        self.token_dropout = nn.Dropout(dropout)
        self.token_proj = nn.Linear(embedding_dim, sem_ids_dim * num_embeddings, bias=False)
        
        # self.cls_dropout = nn.Dropout(p=dropout)
        # self.cls_head = nn.Linear(embedding_dim, num_embeddings, bias=True)
        self.initializer_range = 0.02
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _predict(self, batch):
        # PREPARE EMBEDDING

        # user embedding
        user_emb = self.user_id_embedder(batch.user_ids)

        # token embedding
        sem_ids = batch.sem_ids
        sem_ids_emb = self.sem_id_embedder(sem_ids, batch.token_type_ids, batch.seq_mask)
        
        B, N, D = sem_ids_emb.shape         # [Batch size, 使用的vae层数(4) * 最大长度, embedding_dim]
        pos_max = N // self.sem_ids_dim
        pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat(B, 1)
        pos_emb = self.pos_emb(pos)
        
        # token_pos_emb = self.token_pos_embedder(batch.token_type_ids, pos)

        # sem_ids_emb += token_pos_emb

        # ids embedding
        ids_emb = self.items_embedder(batch.ids)

        # embedding fusion
        sem_ids_emb = sem_ids_emb.reshape(B, pos_max, self.token_embedding_dim * self.sem_ids_dim)
        ids_emb = ids_emb + pos_emb

        ids_emb = self.in_proj(torch.cat([ids_emb, sem_ids_emb], dim=2))

        ids_emb = torch.cat([user_emb, ids_emb], dim=1)                             # [batch_size, pos_max + 1, embedding_dim]

        # print(input_embedding.shape)
        attention_mask = nn.Transformer.generate_square_subsequent_mask(ids_emb.shape[1]).to(ids_emb.device)

        output = self.transformer(ids_emb, attention_mask)

        output = self.out_proj(self.out_dropout(output))[:, 1:, :]

        return output

    @eval_mode
    @torch.no_grad
    def get_candidates_embeddings(self, candidates, device):
        # candidates [num_candidates, sem_ids_dim]
        B, D = candidates.shape
        token_type_ids = torch.arange(D, device=device).repeat(B, 1)
        candidates = token_type_ids * self.sem_id_embedder.num_embeddings + candidates
        return self.sem_id_embedder.emb(candidates)

    @torch.compile
    def forward(self, batch):
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        # sequence representation
        transformer_out = self._predict(batch)
        # print(transformer_out.shape)

        logits = transformer_out.matmul(self.items_embedder.weight.transpose(0, 1))
        transformer_out = self.token_proj(self.token_dropout(transformer_out))
        # print(logits.shape)
        # raise NotImplementedError()
        # print(transformer_out.shape, N, self.num_embeddings)
        return logits, transformer_out.reshape(-1, self.sem_ids_dim, self.num_embeddings)

class DistRecommender(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_ids_dim,
        num_items,
        num_users,
        max_pos = 2048,
        ids_embeddings = -1,
        mlp_hidden_dims = None,
    ):
        super().__init__()

        # INIT
        self.embedding_dim = embedding_dim
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.num_embeddings = num_embeddings
        self.sem_ids_dim = sem_ids_dim
        self.max_pos = max_pos
        self.ids_embeddings = ids_embeddings
        self.enable_generation = False

        # INIT EMBEDDER
        if ids_embeddings == -1:
            ids_embeddings = num_embeddings
        
        self.elu_activation = torch.nn.ELU()

        self.mean_user_id_embedder = nn.Embedding(
            num_embeddings=num_users + 1,
            embedding_dim=embedding_dim,
            padding_idx=num_users
        )
        self.cov_user_id_embedder = nn.Embedding(
            num_embeddings=num_users + 1,
            embedding_dim=embedding_dim,
            padding_idx=num_users
        )

        self.mean_sem_id_embedder = SemIdEmbedder(
            num_embeddings=ids_embeddings,
            sem_ids_dim=sem_ids_dim,
            embedding_dim=embedding_dim // sem_ids_dim
        )
        
        self.cov_sem_id_embedder = SemIdEmbedder(
            num_embeddings=ids_embeddings,
            sem_ids_dim=sem_ids_dim,
            embedding_dim=embedding_dim // sem_ids_dim
        )

        self.mean_token_pos_embedder = nn.Embedding(
            num_embeddings=max_pos * sem_ids_dim + 10,
            embedding_dim=embedding_dim // sem_ids_dim
        )
        self.cov_token_pos_embedder = nn.Embedding(
            num_embeddings=max_pos * sem_ids_dim + 10,
            embedding_dim=embedding_dim // sem_ids_dim
        )

        self.mean_pos_emb = nn.Embedding(
            num_embeddings=max_pos + 10, 
            embedding_dim=embedding_dim
        )
        self.cov_pos_emb = nn.Embedding(
            num_embeddings=max_pos + 10, 
            embedding_dim=embedding_dim
        )

        self.mean_items_embedder = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=num_items
        )

        self.cov_items_embedder = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=num_items
        )

        # INIT TRANSFORMER
        self.mean_in_proj = nn.Linear(2 * embedding_dim, attn_dim, bias=True)
        self.cov_in_proj = nn.Linear(2 * embedding_dim, attn_dim, bias=True)

        self.transformer = DistRecTransformer(
            n_layers=n_layers,
            attn_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_hidden_dims=mlp_hidden_dims
        )

        self.mean_out_dropout = nn.Dropout(dropout)
        self.cov_out_dropout = nn.Dropout(dropout)

        self.mean_out_proj = nn.Linear(attn_dim, embedding_dim, bias=False)
        self.cov_out_proj = nn.Linear(attn_dim, embedding_dim, bias=False)

        self.token_dropout = nn.Dropout(dropout)
        self.token_proj = nn.Linear(embedding_dim, sem_ids_dim * num_embeddings, bias=False)
        
        # self.cls_dropout = nn.Dropout(p=dropout)
        # self.cls_head = nn.Linear(embedding_dim, num_embeddings, bias=True)
        self.initializer_range = 0.02
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _predict(self, batch):
        # PREPARE EMBEDDING

        # user embedding
        mean_user_emb = self.mean_user_id_embedder(batch.user_ids)
        cov_user_emb = self.cov_user_id_embedder(batch.user_ids)

        # token embedding
        sem_ids = batch.sem_ids
        mean_sem_ids_emb = self.mean_sem_id_embedder(sem_ids, batch.token_type_ids, batch.seq_mask)
        cov_sem_ids_emb = self.cov_sem_id_embedder(sem_ids, batch.token_type_ids, batch.seq_mask)
        
        B, N, D = mean_sem_ids_emb.shape         # [Batch size, 使用的vae层数(4) * 最大长度, embedding_dim]
        pos_max = N // self.sem_ids_dim
        pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat(B, 1)
        pos_token = torch.arange(pos_max * self.sem_ids_dim, device=batch.sem_ids.device).repeat(B, 1)

        mean_pos_emb = self.mean_pos_emb(pos)
        cov_pos_emb = self.cov_pos_emb(pos)

        mean_token_pos_emb = self.mean_token_pos_embedder(pos_token)
        cov_token_pos_emb = self.cov_token_pos_embedder(pos_token)

        mean_sem_ids_emb += mean_token_pos_emb
        cov_sem_ids_emb += cov_token_pos_emb
        
        # token_pos_emb = self.token_pos_embedder(batch.token_type_ids, pos)

        # sem_ids_emb += token_pos_emb

        # ids embedding
        # print(self.mean_items_embedder.weight.shape)
        mean_ids_emb = self.mean_items_embedder(batch.ids)
        cov_ids_emb = self.cov_items_embedder(batch.ids)

        # embedding fusion
        mean_sem_ids_emb = mean_sem_ids_emb.reshape(B, pos_max, self.embedding_dim)
        cov_sem_ids_emb = cov_sem_ids_emb.reshape(B, pos_max, self.embedding_dim)

        mean_ids_emb = mean_ids_emb + mean_pos_emb
        cov_ids_emb = cov_ids_emb + cov_pos_emb

        mean_ids_emb = self.mean_in_proj(torch.cat([mean_ids_emb, mean_sem_ids_emb], dim=2))
        cov_ids_emb = self.cov_in_proj(torch.cat([cov_ids_emb, cov_sem_ids_emb], dim=2))

        mean_ids_emb = torch.cat([mean_user_emb, mean_ids_emb], dim=1)                             # [batch_size, pos_max + 1, embedding_dim]
        cov_ids_emb = torch.cat([cov_user_emb, cov_ids_emb], dim=1)

        cov_ids_emb = self.elu_activation(cov_ids_emb) + 1
        mean_ids_emb = self.elu_activation(mean_ids_emb)

        # print(input_embedding.shape)
        attention_mask = nn.Transformer.generate_square_subsequent_mask(mean_ids_emb.shape[1]).to(mean_ids_emb.device)

        mean_output, cov_output = self.transformer(
            mean_ids_emb, 
            cov_ids_emb,
            attention_mask
        )

        mean_output = self.mean_out_proj(self.mean_out_dropout(mean_output))[:, 1:, :]
        cov_output = self.cov_out_proj(self.cov_out_dropout(cov_output))[:, 1:, :]

        return mean_output, cov_output

    @eval_mode
    @torch.no_grad
    def get_candidates_embeddings(self, candidates, device):
        # candidates [num_candidates, sem_ids_dim]
        B, D = candidates.shape
        token_type_ids = torch.arange(D, device=device).repeat(B, 1)
        candidates = token_type_ids * self.sem_id_embedder.num_embeddings + candidates
        return self.sem_id_embedder.emb(candidates)

    @torch.compile
    def forward(self, batch):
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        # sequence representation
        mean_transformer_out, cov_transformer_out = self._predict(batch)
        # print(transformer_out.shape)
        item_mean_emb = self.mean_items_embedder.weight
        item_cov_emb = self.elu_activation(self.cov_items_embedder.weight) + 1

        logits = wasserstein_distance_matmul(
            mean_transformer_out,
            cov_transformer_out,
            item_mean_emb,
            item_cov_emb
        )

        # logits = transformer_out.matmul(self.items_embedder.weight.transpose(0, 1))
        transformer_out = self.token_proj(self.token_dropout(mean_transformer_out))
        # print(logits.shape)
        # raise NotImplementedError()
        # print(transformer_out.shape, N, self.num_embeddings)
        return logits, transformer_out.reshape(-1, self.sem_ids_dim, self.num_embeddings)