# set environments
# set your own path to LEROBOT_DATASET
export HF_LEROBOT_HOME="/mnt/mnt/public_zgc/datasets/IPEC-COMMUNITY/"
# set your own path to TRITON_CACHE_DIR
export TRITON_CACHE_DIR="/mnt/mnt/public_zgc/home/mjwei/repo/hume/tmp/.triton"
export TOKENIZERS_PARALLELISM=false

# set your own WANDB_API_KEY
export WANDB_API_KEY="506abb8bf792fd0f7d24334644539afe72855cec"
export WANDB_PROJECT="hume-vla"

# set your own WANDB_ENTITY
export WANDB_ENTITY="winstonwei"

source "$(pwd)/.venv/bin/activate"
