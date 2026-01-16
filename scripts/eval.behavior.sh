export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

export CUDA_VISIBLE_DEVICES=1 

python -u src/hume/serve_b1k.py \
    --ckpt_path /mnt/project_rlinf/mjwei/repo/hume/outputs/hume/2026-01-16/07-58-44_behavior_ck15-30-15_sh-1_theta1.0-1.0_eps0.0_alp0.0_gpu4_lr5e-5_1e-5_1e-5_2e-5_bs8_s800k/checkpoints/000010/pretrained_model \
    --port 8000