#!/usr/bin/env bash

#SBATCH -A rbe549
#SBATCH -p academic       # partition name
#SBATCH -N 1             # number of nodes
#SBATCH -c 32            # number of CPU cores
#SBATCH --gres=gpu:2     # number of GPUs
#SBATCH -C A30
#SBATCH -t 24:00:00      # walltime (hh:mm:ss)
#SBATCH --mem=64G        # memory per node
#SBATCH --job-name="P2-Group8-Ship-Nofine"


# (1) Source your bashrc so conda is available
source /home/lfrecalde/anaconda3/etc/profile.d/conda.sh
# (2) Activate your conda environment
conda activate cv_cuda

# (3) Run your Python code
python Wrapper.py
