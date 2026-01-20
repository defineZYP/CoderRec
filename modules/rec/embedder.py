import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import NamedTuple

from data.schemas import TokenizedSeqBatch

class SemIdEmbeddingBatch(NamedTuple):
    seq: Tensor
    fut: Tensor

class SemIdEmbedder(nn.Module):
    def __init__(
        self,
        num_embeddings,                     # 数量，一般为codebook size
        sem_ids_dim,                        # 多少层rqvae 3 + 1
        embedding_dim,                     # embedding是多少维
    ):
        super().__init__()

        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim * num_embeddings

        self.emb = nn.Embedding(
            num_embeddings= num_embeddings * self.sem_ids_dim + 1,
            embedding_dim = embedding_dim,
            padding_idx = self.padding_idx
        )

    def forward(self, sem_ids, token_type_ids, seq_mask):
        sem_ids = token_type_ids * self.num_embeddings + sem_ids
        sem_ids[~seq_mask] = self.padding_idx
        return self.emb(sem_ids)

class SemIdPosition(nn.Module):
    def __init__(
        self,
        sem_ids_dim,
        embedding_dim,
        max_pos
    ):
        super().__init__()
        self.sem_ids_dim = sem_ids_dim
        self.position_emb = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.token_position_emb = nn.Embedding(num_embeddings=sem_ids_dim, embedding_dim=embedding_dim)

        self.token_position_fuser = nn.Linear(
            2 * embedding_dim,
            embedding_dim,
            bias=True
        )
        nn.init.zeros_(self.token_position_fuser.weight)
        nn.init.zeros_(self.token_position_fuser.bias)

    def forward(
        self, 
        token_type_ids,
        pos,
    ):
        token_emb = self.token_position_emb(token_type_ids)
        pos_emb = self.position_emb(pos).repeat_interleave(self.sem_ids_dim, dim=1)
        wpe = self.token_position_fuser(torch.cat([pos_emb, token_emb], axis=-1)) + token_emb
        return token_emb

class UserIdEmbedder(nn.Module):
    def __init__(
        self,
        num_buckets,
        embedding_dim
    ):
        super().__init__()
        # TODO change buckets to some other methods
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, x):
        hashed_indices = x % self.num_buckets
        return self.emb(hashed_indices)
