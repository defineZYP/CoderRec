import argparse
import os
import sys
import gin
import torch
import time
import random
import wandb
import pickle

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData

from data.schemas import TokenizedSeqBatch

from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch

from einops import rearrange, pack

from get_args import parse_args

from modules.rec.model import Recommender
from modules.rec.utils import calculate_metrics
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.rqvae.quantize import QuantizeForwardMode, QuantizeDistance
from modules.rqvae.loss import ReconstructionLoss

from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup

import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    warmup_steps=10000,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    num_users=-1,
    num_items=-1,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.STE,
    vae_codebook_distance_mode=QuantizeDistance.L2,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    vae_enable_token_count=True,
    decoder_embed_dim=64,
    decoder_token_embed_dim=None,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    dataset_split="beauty",
    train_data_subsample=True,
    lamb_aux=0.1
):
    # prepare backend of WANDB and accelerator
    setup_seed(42)
    
    if dataset != RecDataset.AMAZON and dataset != RecDataset.AMAZON_QWEN:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device
    # device = 'cpu'

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )

    # ============ INIT DATASET ==================
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split
    )
    
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_codebook_mode=vae_codebook_mode,
        rqvae_codebook_distance_mode=vae_codebook_distance_mode,
        rqvae_sim_vq=vae_sim_vq,
        enable_count_token=vae_enable_token_count,
    )

    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    test_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split
    )

    print(f"Prepare Dataset Users: {num_users} Items: {num_items}")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_dataloader = accelerator.prepare(
        test_dataloader
    )
    # =========== INIT DATASET OVER ==============

    # =========== INIT MODEL ==================
    model = Recommender(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        sem_ids_dim=tokenizer.sem_ids_dim,
        # inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        max_pos=test_dataset.max_seq_len*tokenizer.sem_ids_dim,
        num_users=num_users,
        num_items=num_items,
        token_embedding_dim=decoder_token_embed_dim
    )

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)

    model = accelerator.prepare(
        model
    )

    # =========== INIT MODEL OVER ================

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")
    count = 0
    # test
    # load best
    model.eval()
    preds = []
    preds2 = []
    with torch.no_grad():
        with tqdm(test_dataloader, desc=f"Test", disable=not accelerator.is_main_process) as pbar_eval:
            for batch in pbar_eval:
                B, N = batch.ids.shape
                _, D = tokenizer.cached_ids.shape
                ids = batch.ids
                ids_fut = batch.ids_fut

                sem_ids = tokenizer._tokenize_seq_batch_from_cached(batch.ids)
                ids[~batch.seq_mask] = num_items
                seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
                token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N).to(device)
                target = batch.label.to(device)
                eval_batch = TokenizedSeqBatch(
                    user_ids=batch.user_ids.to(device),
                    sem_ids=sem_ids.to(device),
                    sem_ids_fut=None,
                    ids=ids,
                    ids_fut=ids_fut,
                    seq_mask=seq_mask.to(device),
                    token_type_ids=token_type_ids.to(device),
                    token_type_ids_fut=None
                )
                logits, _ = model(eval_batch)
                rows_ids = torch.arange(ids.shape[0], dtype=torch.long, device=ids.device)
                # last_item_idx = (ids != -1).sum(axis=1) - 1
                last_item_idx = -1
                logits = logits[rows_ids, last_item_idx, :]
                ranks = logits.shape[1] - logits.argsort().argsort()
                ranks = torch.gather(
                    ranks,
                    dim=1,
                    index=target
                )
                preds.append(ranks)
                # preds2.append(ranks_2)
        preds = torch.cat(preds, dim=0).squeeze(dim=1)
        metrics, score = calculate_metrics(preds, return_score=True)
        print(metrics)

if __name__ == "__main__":
    start = time.time()
    print("parsing config ...")
    args = parse_args()
    print(args)
    print("parsing config over ...")
    train(
        iterations=args.rec_iterations,
        batch_size=args.rec_batch_size,
        learning_rate=args.rec_learning_rate,
        weight_decay=args.rec_weight_decay,
        warmup_steps=args.rec_warmup_steps,
        dataset_folder=args.raw_dataset_folder,
        save_dir_root=args.rec_save_dir_root,
        dataset=args.dataset,
        num_users=args.num_users,
        num_items=args.num_items,
        pretrained_rqvae_path=args.pretrained_rqvae_path,
        pretrained_decoder_path=args.pretrained_decoder_path,
        split_batches=args.split_batches,
        amp=args.amp,
        wandb_logging=args.wandb_logging,
        force_dataset_process=args.force_dataset_process,
        mixed_precision_type=args.mixed_precision_type,
        gradient_accumulate_every=args.rec_gradient_accumulate_every,
        save_model_every=args.rec_save_model_every,
        partial_eval_every=args.rec_partial_eval_every,
        full_eval_every=args.rec_full_eval_every,
        vae_input_dim=args.vae_input_dim,
        vae_embed_dim=args.vae_embed_dim,
        vae_hidden_dims=args.vae_hidden_dims,
        vae_codebook_size=args.vae_codebook_size,
        vae_codebook_normalize=args.vae_codebook_normalize,
        vae_codebook_mode=args.vae_codebook_mode,
        vae_codebook_distance_mode=args.vae_codebook_distance_mode,
        vae_sim_vq=args.vae_sim_vq,
        vae_n_cat_feats=args.vae_n_cat_feats,
        vae_n_layers=args.vae_n_layers,
        vae_enable_token_count=args.vae_enable_token_count,
        decoder_embed_dim=args.rec_decoder_embed_dim,
        dropout_p=args.rec_dropout_p,
        attn_heads=args.rec_attn_heads,
        attn_embed_dim=args.rec_attn_embed_dim,
        attn_layers=args.rec_attn_layers,
        dataset_split=args.dataset_split,
        train_data_subsample=args.rec_train_data_subsample,
        decoder_token_embed_dim=args.rec_decoder_token_embed_dim,
        lamb_aux=args.rec_aux_lambda
    )
    end = time.time()
    print(f"use time: {end - start}s")