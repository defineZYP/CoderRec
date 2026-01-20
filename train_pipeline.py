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

from itertools import chain

from modules.rec.model import Recommender
from modules.rec.utils import calculate_metrics
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.rqvae.quantize import QuantizeForwardMode, QuantizeDistance
from modules.rqvae.loss import ReconstructionLoss

from modules.pipeline import ReasoningRecommender
from modules.llm.modeling_qwen3 import Qwen3Model

from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer

from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from tqdm import tqdm

from accelerate import DistributedDataParallelKwargs

import time

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    pretrained_pipeline_path=None,
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
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    n_passes=1,
    dataset_split="beauty",
    train_data_subsample=True,
    lamb_aux=0.1,
    lamb_lm=0.001,
    use_lora=False,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_rank=8,
    lora_bias='none',
    lora_target_modules=None,
    enable_reasoning=False
):
    # prepare backend of WANDB and accelerator
    if dataset != RecDataset.AMAZON and dataset != RecDataset.AMAZON_QWEN:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no',
        kwargs_handlers=[ddp_kwargs]
    )
    setup_seed(42 + accelerator.process_index)

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
    )
    
    tokenizer = accelerator.prepare(tokenizer)
    accelerator.unwrap_model(tokenizer).precompute_corpus_ids(item_dataset)

    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=True, 
        split=dataset_split
    )

    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=False, 
        split=dataset_split
    )

    test_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split
    )

    # 统计结果
    if num_users == -1 or num_items == -1:
        with tqdm(train_dataset) as bar:
            for data in bar:
                num_users = max(num_users, data.user_ids.item())
                num_items = max(num_items, data.ids.max().item())
        with tqdm(eval_dataset) as bar:
            for data in bar:
                num_users = max(num_users, data.user_ids.item())
                num_items = max(num_items, data.ids.max().item())

    print(f"Prepare Dataset Users: {num_users} Items: {num_items}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    train_dataloader = cycle(train_dataloader)

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size * 4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False)
    
    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    train_dataloader_iterator = iter(train_dataloader)
    # =========== INIT DATASET OVER ==============

    # =========== INIT MODEL ==================
    sem_ids_dim = accelerator.unwrap_model(tokenizer).sem_ids_dim
    model = Recommender(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        sem_ids_dim=sem_ids_dim,
        max_pos=train_dataset.max_seq_len * sem_ids_dim,
        num_users=num_users,
        num_items=num_items
    )

    start_iter = 0
    if pretrained_decoder_path is not None and pretrained_pipeline_path is None:
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)

    model_name = "Qwen/Qwen3-4B"
    llm = Qwen3Model.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm.requires_grad_(False) # due to limited resources, it is no possible for us to train the whole model

    if use_lora:
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_rank,
            bias=lora_bias,
            target_modules=lora_target_modules
        )
        llm = get_peft_model(llm, peft_config)
    
    pipeline = ReasoningRecommender(
        llm=llm,
        llm_tokenizer=llm_tokenizer,
        rec_head=model,
        rqvae_bridge=accelerator.unwrap_model(tokenizer).rq_vae,
        num_items=num_items,
        num_users=num_users,
        enable_reasoning=enable_reasoning,
        n_passes=n_passes
    ).to(accelerator.device)

    if pretrained_pipeline_path is not None:
        pretrained_lora_path = os.path.join(pretrained_pipeline_path, 'lora')
        pretrained_rec_path = os.path.join(pretrained_pipeline_path, 'rec')
        pretrained_adapter_path = os.path.join(pretrained_pipeline_path, 'adapter')
        pipeline.load_state_dict(
            pretrained_lora_path,
            pretrained_rec_path,
            pretrained_adapter_path
        )

    optimizer = AdamW(
        params=chain(
            model.parameters(), 
            pipeline.talk_head.parameters(), 
            pipeline.lm_head.parameters(),
            [param for param in llm.parameters() if param.requires_grad]
        ),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps
    )

    pipeline, optimizer, lr_scheduler = accelerator.prepare(
        pipeline, optimizer, lr_scheduler
    )

    # raise NotImplementedError()

    candidates = item_dataset.item_data.to(device).transpose(0, 1)

    # =========== INIT MODEL OVER ================

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")
    count = 0

    # save info
    best_score = -1e10
    best_step = 0
    saved_lora_path = os.path.join(save_dir_root, 'lora')
    saved_rec_path = os.path.join(save_dir_root, 'rec')
    saved_adapter_path = os.path.join(save_dir_root, 'adapter')
    os.makedirs(saved_lora_path, exist_ok=True)
    os.makedirs(saved_rec_path, exist_ok=True)
    os.makedirs(saved_adapter_path, exist_ok=True)
    max_pos = train_dataset.max_seq_len
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    gumbel_warmup = warmup_steps / 10
    gumbel_step = 1
    _, D = accelerator.unwrap_model(tokenizer).cached_ids.shape

    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for it in range(iterations):
            model.train()
            total_loss = 0
            checkpoint_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                lr = optimizer.param_groups[0]["lr"]
                # print(lr)
                batch = next(train_dataloader_iterator)
                B, N = batch.ids.shape

                x = batch.x.to(device)
                x_fut = batch.x_fut.to(device)

                # prepare tokenized data
                ids = batch.ids 
                ids_fut = batch.ids_fut

                sem_ids = accelerator.unwrap_model(tokenizer)._tokenize_seq_batch_from_cached(ids)
                sem_ids_fut = accelerator.unwrap_model(tokenizer)._tokenize_seq_batch_from_cached(ids_fut)
                _lm_seq_mask = batch.seq_mask.to(device)

                # print(lm_seq_mask.shape)
                lm_seq_mask = _lm_seq_mask.unsqueeze(1) * _lm_seq_mask.unsqueeze(2)
                # print(D)
                causal_mask = torch.tril(torch.ones((N, N), device=lm_seq_mask.device)).repeat(B, 1, 1).unsqueeze(1)

                seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)

                sem_ids_fut[~seq_mask] = -1

                token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N).to(device)

                batch = TokenizedSeqBatch(
                    user_ids=batch.user_ids.to(device),
                    sem_ids=sem_ids.to(device),
                    sem_ids_fut=sem_ids_fut.to(device),
                    ids=ids.to(device),
                    ids_fut=ids_fut.to(device),
                    seq_mask=seq_mask.to(device),
                    token_type_ids=token_type_ids.to(device),
                    token_type_ids_fut=token_type_ids.to(device),
                )

                logits, rec_logits, x_hat = pipeline(
                    batch,
                    input_embeds=x,
                    attention_mask=causal_mask,
                    valid_mask=lm_seq_mask.unsqueeze(1),
                    use_cache=False
                )

                logits = logits.matmul(candidates)
                loss = lamb_lm * loss_fn(logits.view(-1, logits.shape[-1]), ids_fut.view(-1).to(logits.device))
                loss3 = lamb_aux * (F.mse_loss(x_hat, x_fut, reduction='none') * _lm_seq_mask.unsqueeze(-1).to(x_hat.device)).mean()
                # loss = 0.05 * loss_fn(logits.view(-1, logits.shape[-1]), ids_fut.view(-1).to(logits.device))
                # print(f"loss: {loss}")
                loss2 = loss_fn(rec_logits.view(-1, rec_logits.shape[-1]), ids_fut.view(-1).to(logits.device))
                # print(f"loss2: {loss2}")
                total_loss += loss + loss2 + loss3
                checkpoint_loss += loss2
                # print(f"total_loss: {total_loss}")

            accelerator.backward(total_loss)
            # print(sem_ids.reshape(B, N, D))
            pbar.set_description(f'loss: {total_loss.item():.4f}')
            accelerator.wait_for_everyone()
            optimizer.step()
            lr_scheduler.step()
            accelerator.wait_for_everyone()

            if (it + 1) % full_eval_every == 0:
                pipeline.eval()
                preds = []
                preds2 = []
                with torch.no_grad():
                    with tqdm(eval_dataloader, desc=f"Eval {it + 1}", disable=not accelerator.is_main_process) as pbar_eval:
                        for batch in pbar_eval:
                            B, N = batch.ids.shape
                            ids = batch.ids
                            ids_fut = batch.ids_fut

                            x = batch.x.to(device)

                            sem_ids = accelerator.unwrap_model(tokenizer)._tokenize_seq_batch_from_cached(batch.ids)
                            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
                            token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N).to(device)
                            target = batch.label.to(device)
                            lm_seq_mask = batch.seq_mask.to(device)
                            # print(lm_seq_mask.shape)
                            lm_seq_mask = lm_seq_mask.unsqueeze(1) * lm_seq_mask.unsqueeze(2)
                            # print(D)
                            causal_mask = torch.tril(torch.ones((N, N), device=lm_seq_mask.device)).repeat(B, 1, 1).unsqueeze(1)

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

                            logits, _ = accelerator.unwrap_model(pipeline).rec_head(eval_batch)

                            rows_ids = torch.arange(ids.shape[0], dtype=torch.long, device=ids.device)
                            last_item_idx = -1
                            logits = logits[rows_ids, last_item_idx, :]
                            ranks = logits.shape[1] - logits.argsort().argsort()
                            ranks = torch.gather(
                                ranks,
                                dim=1,
                                index=target
                            )
                            preds.append(ranks)
                    preds = torch.cat(preds, dim=0)
                    accelerator.wait_for_everyone()
                    preds = accelerator.gather(preds).squeeze(dim=1)
                    if accelerator.is_main_process:
                        metrics, _ = calculate_metrics(preds, return_score=True)
                        score = - checkpoint_loss.item()
                        checkpoint_loss = 0
                        if score > best_score:
                            best_score = score
                            best_step = it
                            accelerator.unwrap_model(pipeline).save_state_dict(
                                saved_lora_path,
                                saved_rec_path,
                                saved_adapter_path
                            )
                        print(f"rec head: {metrics} best step: {best_step}")
                        sys.stdout.flush()
                    accelerator.wait_for_everyone()
            pbar.update(1)
            gumbel_step += 1
    
    accelerator.unwrap_model(pipeline).load_state_dict(
        saved_lora_path,
        saved_rec_path,
        saved_adapter_path
    )
    pipeline.eval()
    preds = []
    with torch.no_grad():
        with tqdm(test_dataloader, desc=f"Test", disable=not accelerator.is_main_process) as pbar_eval:
            for batch in pbar_eval:
                B, N = batch.ids.shape
                ids = batch.ids
                ids_fut = batch.ids_fut

                x = batch.x.to(device)

                sem_ids = accelerator.unwrap_model(tokenizer)._tokenize_seq_batch_from_cached(batch.ids)
                seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
                token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N).to(device)
                target = batch.label.to(device)
                lm_seq_mask = batch.seq_mask.to(device)
                # print(lm_seq_mask.shape)
                lm_seq_mask = lm_seq_mask.unsqueeze(1) * lm_seq_mask.unsqueeze(2)
                # print(D)
                causal_mask = torch.tril(torch.ones((N, N), device=lm_seq_mask.device)).repeat(B, 1, 1).unsqueeze(1)

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

                logits, _ = accelerator.unwrap_model(pipeline).rec_head(eval_batch)

                rows_ids = torch.arange(ids.shape[0], dtype=torch.long, device=ids.device)
                last_item_idx = -1
                logits = logits[rows_ids, last_item_idx, :]
                ranks = logits.shape[1] - logits.argsort().argsort()
                ranks = torch.gather(
                    ranks,
                    dim=1,
                    index=target
                )
                preds.append(ranks)
        preds = torch.cat(preds, dim=0)
        accelerator.wait_for_everyone()
        preds = accelerator.gather(preds).squeeze(dim=1)
        if accelerator.is_main_process:
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
        save_dir_root=args.pipeline_save_dir_root,
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
        decoder_embed_dim=args.rec_decoder_embed_dim,
        dropout_p=args.rec_dropout_p,
        attn_heads=args.rec_attn_heads,
        attn_embed_dim=args.rec_attn_embed_dim,
        attn_layers=args.rec_attn_layers,
        dataset_split=args.dataset_split,
        train_data_subsample=args.rec_train_data_subsample,
        use_lora=args.use_lora,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_rank=args.lora_rank,
        lora_bias=args.lora_bias,
        lora_target_modules=args.lora_target_modules,
        pretrained_pipeline_path=args.pretrained_pipeline_path,
        enable_reasoning=args.enable_reasoning,
        n_passes=args.n_passes,
        lamb_aux=args.rec_aux_lambda,
        lamb_lm=args.pipeline_aux_lambda
    )
    end = time.time()
    print(f"use time {end - start}s")