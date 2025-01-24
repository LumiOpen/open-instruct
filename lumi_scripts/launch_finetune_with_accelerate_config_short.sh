#!/bin/bash
#SBATCH --job-name=short_tulu_3_sft
#SBATCH --account=project_462000353  
#SBATCH --partition=dev-g
#SBATCH --cpus-per-task=56
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file

echo "JOB NAME" $SLURM_JOB_NAME

module use /appl/local/csc/modulefiles/ 
module load pytorch 
source /scratch/project_462000444/zosaelai2/.tulu_venv/bin/activate


export HF_DATASETS_CACHE="/scratch/project_462000444/zosaelai2/datasets_cache"
export HF_HOME="/scratch/project_462000444/cache"
export PYTHONPATH="/scratch/project_462000444/zosaelai2/.tulu_venv/lib/python3.10/site-packages"
#pip show transformers


#Distributed variables
#Distributed variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
#export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

echo "WORLD_SIZE: $WORLD_SIZE"
echo "LOCAL_RANK: $LOCAL_RANK"


#LOGGING/DEBUGGING
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#HF_HUB_ENABLE_HF_TRANSFER=1 #Speeds up loading from hf hub, i think
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 #This might not work with rccl
#export HSA_FORCE_FINE_GRAIN_PCIE=1 #Supposedly improves performance/prevents hanging
#export HIP_LAUNCH_BLOCKING=1 #Removes async operations
#export TRANSFORMERS_VERBOSITY=error
#export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
#export ACCELERATE_LOG_LEVEL=DEBUG
export OMP_NUM_THREADS=1 #This could be increased
export TOKENIZERS_PARALLELISM=false #Removes error involved with the FastTokenizer and rust/python parallelism.
                                    #See more:https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996


ACCELERATE_CONFIG_FILE=configs/ds_configs/deepspeed_zero3.yaml
CONFIG_FILE=configs/train_configs/tulu3/tulu3_sft_short.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $CONFIG_FILE
pwd -P

export CMD=" \
    open_instruct/finetune.py $CONFIG_FILE 
    "


#LAUNCHERS
export ACC_LAUNCHER="singularity_wrapper exec accelerate launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "


SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"

# # NUM_GPUS=32
# # CONFIG_FILE=configs/train_configs/tulu3/tulu3_sft.yaml

# # Generate CUDA_VISIBLE_DEVICES as a range from 0 to NUM_GPUS-1
# CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
# export CUDA_VISIBLE_DEVICES

# echo "Number of GPUs: $NUM_GPUS"
# echo "Using config file: $CONFIG_FILE"
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# # You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# # but it will trade off speed.
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     --use_deepspeed \
#     --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
#     open_instruct/finetune.py \
#     "configs/train_configs/tulu3/tulu3_sft.yaml" #\
#     #--report_to=tensorboard,wandb