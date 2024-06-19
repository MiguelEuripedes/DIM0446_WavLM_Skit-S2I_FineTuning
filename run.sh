#!/bin/bash
#SBATCH --job-name=Tst_WavLM
#SBATCH --time=1:30:00
#SBATCH --partition=gpu-8-v100
#SBATCH --nodelist=gpunode0
#SBATCH --gres=gpu:1
#_SBATCH --exclusive

export TORCH_CUDA_VERSION=cu117

srun python trainer_wavML.py
