#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junbin.cheng@ontariotechu.net
#SBATCH --output-outfile/myfile-%j.out

# Load the necessary modules
module load python/3.10
