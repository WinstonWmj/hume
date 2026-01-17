export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

export CUDA_VISIBLE_DEVICES=1 
export PYTHONPATH=/mnt/project_rlinf/mjwei/repo/BEHAVIOR-1K-latest-comet/OmniGibson:$PYTHONPATH

# PYTHONUNBUFFERED=1 is used to prevent buffering of stdout and stderr, so that the output is displayed immediately.
PYTHONUNBUFFERED=1 python src/hume/serve_b1k.py \
    --port=8000 \
    --task-name=turning_on_radio   \
    --control-mode=receeding_horizon   \
    --task-name=turning_on_radio \
    --max-len=32  \
    --ckpt_path=/mnt/project_rlinf/mjwei/repo/hume/outputs/hume/2026-01-16/07-58-44_behavior_ck15-30-15_sh-1_theta1.0-1.0_eps0.0_alp0.0_gpu4_lr5e-5_1e-5_1e-5_2e-5_bs8_s800k/checkpoints/000010/pretrained_model

# export TORCHDYNAMO_DISABLE=1
# export TORCH_COMPILE_DISABLE=1

# CUDA_VISIBLE_DEVICES=1 
# python scripts/serve_b1k.py  \
#     --task-name=turning_on_radio   \
#     --control-mode=receeding_horizon   \
#     --max-len=32   \
#     policy:checkpoint   \
#     --policy.config=pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k   \
#     --policy.dir=/mnt/public/quanlu/openpai_comet_model/pi05-b1kpt12-cs32