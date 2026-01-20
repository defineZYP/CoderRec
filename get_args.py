import argparse
import json

from modules.rqvae.quantize import QuantizeForwardMode, QuantizeDistance
from data.processed import RecDataset

def parse_args():
    parser = argparse.ArgumentParser()
    # MONITOR and INIT
    parser.add_argument('--split_batches', type=bool, default=True)
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--wandb_logging', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--config', type=str, default=None)

    # Dataset
    parser.add_argument('--raw_dataset_folder', type=str, default="dataset/ml-1m")
    parser.add_argument('--dataset', type=int, default=0)                            # Need Change
    parser.add_argument('--num_users', type=int, default=-1)
    parser.add_argument('--num_items', type=int, default=-1)
    parser.add_argument('--force_dataset_process', type=bool, default=False)
    parser.add_argument('--mixed_precision_type', type=str, default="fp16")
    parser.add_argument('--dataset_split', type=str, default="beauty")

    # LLM
    parser.add_argument('--pretrained_model_path', type=str, default="/DATA/DATANAS2/zhangyip/models/Qwen3-4B")
    parser.add_argument('--enable_reasoning', type=bool, default=False)

    # RQVAE
    parser.add_argument('--pretrained_rqvae_path', type=str, default=None)
    parser.add_argument('--vae_iterations', type=int, default=50000)
    parser.add_argument('--vae_warmup_steps', type=int, default=10000)
    parser.add_argument('--vae_batch_size', type=int, default=64)
    parser.add_argument('--vae_learning_rate', type=int, default=0.0001)
    parser.add_argument('--vae_min_learning_rate', type=int, default=0.0000001)
    parser.add_argument('--vae_weight_decay', type=float, default=0.01)
    parser.add_argument('--vae_save_dir_root', type=str, default="out/")
    parser.add_argument('--vae_use_kmeans_init', type=bool, default=True)
    parser.add_argument('--vae_gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--vae_commitment_weight', type=float, default=0.25)
    parser.add_argument('--vae_n_cat_feats', type=int, default=0)
    parser.add_argument('--vae_input_dim', type=int, default=18)
    parser.add_argument('--vae_embed_dim', type=int, default=16)
    parser.add_argument('--vae_hidden_dims', type=str, default="[18, 18]")                              # Need Change
    parser.add_argument('--vae_codebook_size', type=int, default=32)
    parser.add_argument('--vae_codebook_normalize', type=bool, default=0)
    parser.add_argument('--vae_codebook_normalize_type', type=str, default='rms')
    parser.add_argument('--vae_codebook_mode', type=int, default=0)                                     # Need Change
    parser.add_argument('--vae_codebook_distance_mode', type=int, default=0)                            # Need Change
    parser.add_argument('--vae_sim_vq', type=bool, default=False)
    parser.add_argument('--vae_enable_token_count', type=bool, default=True)
    parser.add_argument('--vae_n_layers', type=int, default=3)
    parser.add_argument('--vae_global_normalize', type=bool, default=False)
    parser.add_argument('--vae_save_model_every', type=int, default=1000000)
    parser.add_argument('--vae_eval_every', type=int, default=50000)

    # REC
    parser.add_argument('--pretrained_decoder_path', type=str, default=None)
    parser.add_argument('--rec_iterations', type=int, default=500000)
    parser.add_argument('--rec_circle_iterations', type=int, default=500000)
    parser.add_argument('--rec_warmup_steps', type=int, default=1000)
    parser.add_argument('--rec_batch_size', type=int, default=64)
    parser.add_argument('--rec_learning_rate', type=float, default=0.01)
    parser.add_argument('--rec_weight_decay', type=float, default=0.01)
    parser.add_argument('--rec_save_dir_root', type=str, default="out/")
    parser.add_argument('--rec_gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--rec_save_model_every', type=int, default=1000000)
    parser.add_argument('--rec_partial_eval_every', type=int, default=1000)
    parser.add_argument('--rec_full_eval_every', type=int, default=10000)
    parser.add_argument('--rec_decoder_embed_dim', type=int, default=64)
    parser.add_argument('--rec_decoder_token_embed_dim', type=int, default=None)
    parser.add_argument('--rec_dropout_p', type=float, default=0.1)
    parser.add_argument('--rec_attn_heads', type=int, default=8)
    parser.add_argument('--rec_attn_embed_dim', type=int, default=64)
    parser.add_argument('--rec_attn_layers', type=int, default=4)
    parser.add_argument('--rec_train_data_subsample', type=bool, default=True)
    parser.add_argument('--rec_aux_lambda', type=float, default=0.01)

    # LoRA and pipeline
    parser.add_argument('--use_lora', type=bool, default=0)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_bias', type=str, default='none')
    parser.add_argument('--lora_target_modules', type=str, default='["q_proj", "k_proj", "v_proj", "o_proj"]')
    parser.add_argument('--n_passes', type=int, default=1)
    parser.add_argument('--pretrained_pipeline_path', type=str, default=None)
    parser.add_argument('--pipeline_save_dir_root', type=str, default='./out/')
    parser.add_argument('--pipeline_aux_lambda', type=float, default=0.001)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config) as f:
            configs = json.load(f)
            for key in configs:
                args.__setattr__(key, configs[key])
    
    if args.dataset == 0:
        args.dataset = RecDataset.AMAZON
    elif args.dataset == 1:
        args.dataset = RecDataset.ML_1M
    elif args.dataset == 2:
        args.dataset = RecDataset.ML_32M
    else:
        raise NotImplementedError(f"Unsupported dataset mode {args.dataset}")
    
    if isinstance(args.vae_hidden_dims, str): 
        args.vae_hidden_dims = eval(args.vae_hidden_dims)

    if isinstance(args.lora_target_modules, str):
        args.lora_target_modules = eval(args.lora_target_modules)
    
    if args.vae_codebook_mode == 0:
        args.vae_codebook_mode = QuantizeForwardMode.STE
    elif args.vae_codebook_mode == 1:
        args.vae_codebook_mode = QuantizeForwardMode.ROTATION_TRICK
    elif args.vae_codebook_mode == 2:
        args.vae_codebook_mode = QuantizeForwardMode.GUMBEL_SOFTMAX
    else:
        raise NotImplementedError(f"Unsupported vae codebook mode {args.vae_codebook_mode}")
    
    if args.vae_codebook_distance_mode == 0:
        args.vae_codebook_distance_mode = QuantizeDistance.L2
    elif args.vae_codebook_distance_mode == 1:
        args.vae_codebook_distance_mode = QuantizeDistance.COSINE
    else:
        raise NotImplementedError(f"Unsupported vae codebook distance mode {args.vae_codebook_distance_mode}")
    
    return args

