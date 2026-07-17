#!/bin/bash

# BEGIN MANDATORY OPTIONS
#SBATCH --time=0-10:00:00       # Time limit
#SBATCH --qos=short            # priority/quality of service
#SBATCH --account="physioFM"      # Replace with your CLAM project name
# END MANDATORY OPTIONS

# BEGIN INFORMATIONAL OPTIONS
#SBATCH --job-name="Ablation study on NormWear and FiLM"        # Name for your job
#SBATCH --comment=""  # Comment for your job
#SBATCH --output=tmp/cogload_eval.out      # Output file
#SBATCH --error=tmp/cogload_eval.err       # Error file
#SBATCH --mail-type=BEGIN,END,FAIL   # Mail on start and end job
#SBATCH --mail-user=loqman.ouagague@ec-nantes.fr  # Email address for the job
# END INFORMATIONAL OPTIONS

# BEGIN RESOURCES OPTIONS
#SBATCH --partition=standard      # partition standard
#SBATCH --ntasks=32            # How many CPUs to use
##SBATCH --gres=gpu:1           # How many GPUs to use
# END RESOURCES OPTIONS


# BEGIN JOB

# Load the necessary modules (e.g., Python, CUDA, etc.)
module purge

# Recharger le .bashrc
source ~/.bashrc

cd /scratch/nautilus/projects/physioFM/NormWear

source .venv/bin/activate

# Run your Python script
python3 Experiments/NormWear\ +\ FiLM/run_ablation.py
exit
