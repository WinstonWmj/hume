export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

export CUDA_VISIBLE_DEVICES=4 
# export PYTHONPATH=/mnt/project_rlinf/mjwei/repo/BEHAVIOR-1K-latest-comet/OmniGibson:$PYTHONPATH
export OMNIGIBSON_DATA_PATH=/mnt/public/quanlu/BEHAVIOR-1K-datasets
# PYTHONUNBUFFERED=1 is used to prevent buffering of stdout and stderr, so that the output is displayed immediately.
PYTHONUNBUFFERED=1 python src/hume/serve_b1k_comet.py \
    --port=8000 \
    --task-name=turning_on_radio \
    --control-mode=receeding_horizon \
    --replan-steps=32  \
    --s2-replan-steps=32 \
    --s2-candidates-num=5 \
    --no-post-process-action \
    --ckpt_path=/mnt/project_rlinf/mjwei/repo/hume/outputs/comet_vqh/2026-01-22/05-11-37_comet_behavior_ck32_vqh32_frozen_gpu8_lr2e-4_3e-4_3e-4_3e-4_bs48_s160k/checkpoints/002000/pretrained_model
