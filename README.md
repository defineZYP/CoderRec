# CoderRec

This is the implementation of paper "Cross-Scale Collaboration between LLMs and Lightweight Sequential Recommenders with Domain-Specific Latent Reasoning"



### Install

Create and activate the conda environment

```bash
conda create -n coderrec Python=3.10
conda activate coderrec
```

Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies

```bash
pip install -r requirements.txt
```



### Usage

#### Training

* Train the RQVAE. 

  ```bash
  CUDA_VISIBLE_DEVICES=0 python train_rqvae.py --config ./configs/rqvae/beauty.json
  ```

  

* Train the decoder.

  ```bash
  CUDA_VISIBLE_DEVICES=0 python train_decoder.py --config ./configs/decoder/beauty.json --pretrained_rqvae_path <PATH_OF_RQVAE_CHECKPOINT>
  ```

  

* Train the overall pipeline with latent reasoning. 

  * This step requires substantially greater computational resources than the preceding two steps. However, even if these resources cannot be met, satisfactory performance can still be achieved by relying solely on the first two steps.

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train_pipeline.py --config ./configs/pipeline/beauty.json --n_passes 2 --enable_reasoning 1 --pretrained_rqvae_path <PATH_OF_RQVAE_CHECKPOINT> --pretrained_decoder_path <PATH_OF_DECODER_CHECKPOINT>
  ```

  

#### Inference

* You can download the model weights [here](https://drive.google.com/drive/folders/1iPMT5o5znHW_aO4ndiMn1VggV5qVRwLy?usp=sharing), then run

  ```bash 
  bash ./scripts/inference_beauty.sh
  ```

  

### Acknowledgements

This repository is developed based on the excellent implementation of [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender).
