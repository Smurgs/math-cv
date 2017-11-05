#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:lgpu:4   
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4    # There are 24 CPU cores on Cedar GPU nodes
#SBATCH --time=0-00:10
source ../tensorflow/bin/activate
python mathcv.py train
