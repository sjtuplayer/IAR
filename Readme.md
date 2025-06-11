# Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction

### [Paper](https://arxiv.org/abs/2501.00880) | [Page](https://sjtuplayer.github.io/projects/IAR/)

[Teng Hu](https://github.com/sjtuplayer), [Jiangning Zhang](https://zhangzjn.github.io/), [Ran Yi](https://yiranran.github.io/), [Jieyu Weng](https://github.com/sjtuplayer/MotionMaster), [Yabiao Wang](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ), [Xianfang Zeng](https://github.com/sjtuplayer/MotionMaster), [Zhucun Xue](https://github.com/sjtuplayer/MotionMaster), and [Lizhuang Ma](https://dmcv.sjtu.edu.cn/)

[![image](https://github.com/sjtuplayer/IAR/raw/main/__assets__/images/framework.png)](https://github.com/sjtuplayer/IAR/blob/main/__assets__/images/framework.png)

## Overview

This repository contains the implementation for IAR pipeline, including training, sampling, and evaluation components.

## Prerequisites

Before getting started, ensure you have:

- Python ≥ 3.7
- PyTorch ≥ 2.1
- Access to ImageNet or similar dataset for training

## Setup

Download pretrained weights of VQGAN from LlamaGen: https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt

## Training Pipeline

### 0. Data Preparation

First extract codes from your training images:

```
torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
autoregressive/train/extract_codes_c2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--data-path /path/to/imagenet/train \
--code-path /path/to/output/codes \
--ten-crop \
--crop-range 1.1 \
--image-size 384
```

### 1. Preprocess VQ Checkpoint
Reorder the codebook with balanced k-means algorithm.
It will output the reordered codebook for inference and a mapping
function for training.

```
python balance_k_means.py
```

### 2. Model Training

Train the autoregressive model:

```
PYTHONPATH=$PYTHONPATH:./ torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
autoregressive/train/train_c2i.py \
--results-dir ./results \
--code-path /path/to/output/codes \
--image-size 384 \
--gpt-model GPT-B
```

## Sampling



### Generate Images

Generate new images using a trained model:

```
PYTHONPATH=$PYTHONPATH:./ torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i-reorder-kmeans+nearest-cluster_size=128.pt \
--gpt-ckpt results/your_model_directory/checkpoints \
--gpt-model GPT-B \
--image-size 384 \
--image-size-eval 256 \
--cfg-scale 2.25 \
--num-fid-samples=50000 \
--sample-dir=samples
```

## Evaluation

Before evaluation, install required packages as specified in `evaluations/README.md`.

Evaluate generated samples:

```
python3 evaluations/c2i/evaluator.py \
evaluations/VIRTUAL_imagenet256_labeled.npz \
samples/your_generated_samples.npz
```

## Citation

If you find this code helpful for your research, please cite:


```
@inproceedings{hu2025improving,
     title={Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction},
     author={Hu, Teng and Zhang, Jiangning and Yi, Ran and Weng, Jieyu and Wang, Yabiao and Zeng, Xianfang and Xue, Zhucun and Ma, Lizhuang},
     booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
     year={2025}
}
```