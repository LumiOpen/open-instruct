#!/bin/bash
#SBATCH --job-name=env_setup
#SBATCH --partition=dev-g 
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1    
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=7
#SBATCH --time=00:10:00
#SBATCH --account=project_462000353
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err

mkdir -p logs

# Load modules
module load LUMI #Loads correct compilers for the accelerators, propably not needed
module use /appl/local/csc/modulefiles/ #Add the module path needed for csc modules in Lumi
module load pytorch


#Create venv
python -m venv .tulu_venv --system-site-packages

#Activate
source .tulu_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install build dependencies
pip install setuptools wheel

# Install the package in editable mode
cd open-instruct && pip install -e .
pip install wandb



