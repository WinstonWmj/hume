DEBUG=false
if [ "$DEBUG" = true ]; then
  GPUS=1
  PER_DEVICE_BATCH_SIZE=8
  wandb_enable=false
  ACCELERATE_ARGS="--num_machines 1 --num_processes ${GPUS} --mixed_precision=bf16 --dynamo_backend=no"
  num_workers=8
  save_freq=10
  steps=2000
fi

# distributed settings
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-48}
wandb_enable=${wandb_enable:-true}
wandb_mode=${wandb_mode:-offline}
num_workers=${num_workers:-8}  # 2 workers per GPU for 8 GPUs

# set environments
source scripts/env.behavior.sh
# distritubed training
find_free_port() {
    while true; do
        port=$(shuf -i 20000-65535 -n 1)
        if ! netstat -tna | grep -q ":${port}.*LISTEN"; then
            echo $port
            break
        fi
    done
}
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(find_free_port)
ACCELERATE_ARGS=${ACCELERATE_ARGS:-"--main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --num_machines ${NODES} --num_processes=${GPUS} --multi_gpu \
  --mixed_precision=bf16 --dynamo_backend=no"}

# WANDB


# dataset mapping
declare -A data_map
data_map["libero_spatial"]=libero_spatial_no_noops_1.0.0_lerobot
data_map["libero_object"]=libero_object_no_noops_1.0.0_lerobot
data_map["libero_goal"]=libero_goal_no_noops_1.0.0_lerobot
data_map["libero_10"]=libero_10_no_noops_1.0.0_lerobot
data_map["behavior"]=behavior

# Comet weights path
declare -A pretrained_comet_map
pretrained_comet_map["behavior"]=/mnt/project_rlinf/mjwei/download_models/openpi_comet/sunshk/comet_weights_pytorch_2/pi05-b1kpt12-cs32/

data_name=behavior
dataset=${data_map[$data_name]}
echo "dataset: ${dataset}"
# Use comet-specific config file
cfg="behavior_comet"

# Hyper Parameters for CometPolicy
# NOTE: chunk_size should match PI0's action_horizon (32 for comet)
chunk_size=32
# VQH chunk size: should match chunk_size for no-receding-horizon control
# CalQL action_dim = vqh_chunk_size * 23 = 32 * 23 = 736
vqh_chunk_size=32

# PI0 settings
freeze_pi0=true

lr=2e-4
critic_lr=3e-4
actor_lr=3e-4
temp_lr=3e-4

steps=${steps:-$((GPUS * 20000))}
save_freq=${save_freq:-1000}

# exp names
train_args="${freeze_pi0:+--freeze_pi0=${freeze_pi0}} "

job_name="comet_${data_name}_ck${chunk_size}_vqh${vqh_chunk_size}_"\
"${freeze_pi0:+frozen_}"\
"gpu${GPUS}_lr${lr}_${critic_lr}_${actor_lr}_${temp_lr}_"\
"bs${PER_DEVICE_BATCH_SIZE}_"\
"s$((steps / 1000))k"

pretrained_comet_path=${pretrained_comet_map[$data_name]}
echo "pretrained_comet_path: ${pretrained_comet_path}"

# Launch training
echo "train_args: ${train_args}"
echo "job_name: ${job_name}"
CMD="accelerate launch $ACCELERATE_ARGS src/hume/training/train_vqh_s1.py ${train_args} \
  --pretrained_comet_path=${pretrained_comet_path} \
  --config_path=config/${cfg}.json \
  --dataset.repo_id=${dataset} \
  --dataset.video_backend="pyav" \
  --dataset.image_transforms.enable=true \
  --num_workers=${num_workers} \
  --policy_optimizer_lr=${lr} \
  --s2_chunk_size=${chunk_size} \
  --steps=${steps} \
  --batch=${PER_DEVICE_BATCH_SIZE} \
  --save_freq=${save_freq} \
  --log_freq=1 \
  --job_name=${job_name} \
  --wandb.enable=${wandb_enable} \
  --wandb.mode=${wandb_mode} \
  --wandb.disable_artifact=true \
  --wandb.project=${WANDB_PROJECT} \
  --wandb.entity=${WANDB_ENTITY} \
  --next_obs_offset=${vqh_chunk_size} \
  --vqh_chunk_size=${vqh_chunk_size} \
  --critic_lr=${critic_lr} \
  --actor_lr=${actor_lr} \
  --temp_lr=${temp_lr} \
  --checkpoints_total_limit=0 \
  --output_base=comet_vqh
"
export REPO_PATH=$(dirname "$(dirname "${BASH_SOURCE[0]}")")
LOG_DIR="${REPO_PATH}/outputs/logs/$(date +'%Y%m%d-%H:%M:%S')-${job_name}" #/$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${LOG_DIR}"
MEGA_LOG_FILE="${LOG_DIR}/run_vqh_comet.log"
echo MEGA_LOG_FILE: ${MEGA_LOG_FILE}
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
