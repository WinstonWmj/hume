export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

export CUDA_VISIBLE_DEVICES=4 
# export PYTHONPATH=/mnt/project_rlinf/mjwei/repo/BEHAVIOR-1K-latest-comet/OmniGibson:$PYTHONPATH
export OMNIGIBSON_DATA_PATH=/mnt/public/quanlu/BEHAVIOR-1K-datasets
# PYTHONUNBUFFERED=1 is used to prevent buffering of stdout and stderr, so that the output is displayed immediately.
PYTHONUNBUFFERED=1 python src/hume/serve_b1k.py \
    --port=8000 \
    --task-name=turning_on_radio \
    --control-mode=receeding_horizon \
    --replan-steps=15  \
    --s2-replan-steps=30 \
    --s2-candidates-num=5 \
    --no-post-process-action \
    --ckpt_path=/mnt/public/mjwei/download_models/hume_ckpts/2026-01-20/18-06-31_behavior_ck15-30-15_sh-1_theta1.0-1.0_eps0.0_alp0.0_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs8_s1600k/checkpoints/0010000/pretrained_model
