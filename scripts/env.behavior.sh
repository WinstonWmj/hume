# set environments
# set your own path to LEROBOT_DATASET
export HF_LEROBOT_HOME="/mnt/project_rlinf/mjwei/download_models"
# set your own path to TRITON_CACHE_DIR
export TRITON_CACHE_DIR="/mnt/project_rlinf/mjwei/repo/hume/tmp/.triton"
export TOKENIZERS_PARALLELISM=false

# set your own WANDB_API_KEY
# export WANDB_API_KEY="506abb8bf792fd0f7d24334644539afe72855cec" # using key set in .bashrc
export WANDB_PROJECT="hume-vla"

# CRITICAL: Prevent wandb from blocking in multi-GPU training
# Use "disabled" to completely skip wandb initialization (no network calls)
# Use "offline" to log locally without network, or "online" if API key is set
export WANDB_MODE=${WANDB_MODE:-"offline"}

# set your own WANDB_ENTITY
export WANDB_ENTITY="winstonwei"

source "$(pwd)/.venv/bin/activate"
