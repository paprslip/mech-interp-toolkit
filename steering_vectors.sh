#!/bin/bash
#SBATCH --job-name=steering_vectors
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=def-rgrosse
#SBATCH --time=1:30:00
#SBATCH --output=logs/%u-%x-%j.log
#SBATCH --error=logs/%u-%x-%j.log

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python arrow
source /home/terrence/mech-interp-toolkit/activate
python /home/terrence/mech-interp-toolkit/main.py