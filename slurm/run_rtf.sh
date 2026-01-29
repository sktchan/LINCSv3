#!/bin/bash

#SBATCH --job-name=rank_tf
#SBATCH --output=out/rank_tf.out
#SBATCH --error=out/rank_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

#SBATCH --gres=gpu:L40S:1
#SBATCH --time=3-00:00:00

cd /home/golem/scratch/chans/lincsv3
julia scripts/rank_tf.jl
