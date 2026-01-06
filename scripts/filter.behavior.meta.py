import json
from pathlib import Path

# ================= 配置区域 =================
DATASET_ROOT = Path("/mnt/mnt/public_zgc/datasets/behavior/")  # 数据集根目录
OLD_META_FILE = DATASET_ROOT / "meta/episodes.jsonl"
# ===========================================

filtered_episodes = []
kept_count = 0
total_count = 0

print(f"正在读取并过滤 {OLD_META_FILE} ...")

with open(OLD_META_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        
        total_count += 1
        item = json.loads(line)
        
        # 获取 episode_index
        ep_idx = item["episode_index"]
        
        # --- 关键步骤：构建路径 ---
        # 规则：data/task-{task_idx}/episode_{ep_idx}.parquet
        # task是每200个episode一个文件夹，但是除10000才能得到对应的task_idx
        # 因此，格式是：data/task-0000/episode_00000010.parquet
        task_idx = ep_idx // 10000
        filename = f"episode_{ep_idx:08d}.parquet"
        file_rel_path = Path(f"data/task-{task_idx:04d}/{filename}")
        full_file_path = DATASET_ROOT / file_rel_path
        # breakpoint()
        # 检查文件是否存在
        if full_file_path.exists():
            filtered_episodes.append(item)
            kept_count += 1
        # 如果你只下载了 videos 而不是 data (虽然不常见)，也可以检查 videos 路径：
        # video_path = DATASET_ROOT / f"videos/chunk-{task_idx:03d}/episode_{ep_idx:06d}.mp4"
        
print(f"处理完成！")
print(f"原始条目总数: {total_count}")
print(f"本地存在条目: {kept_count}")

if kept_count == 0:
    print("❌ 警告：没有找到任何匹配的本地文件！请检查 `DATASET_ROOT` 设置或 chunk 文件夹结构。")
    print(f"脚本尝试寻找的路径示例: {full_file_path}")
else:
    # 覆盖写入原来的 meta 文件 (建议先备份)
    backup_path = DATASET_ROOT / "meta/episodes_backup.jsonl"
    if not backup_path.exists():
        import shutil
        shutil.copy(OLD_META_FILE, backup_path)
        print(f"已备份原文件到: {backup_path}")

    with open(DATASET_ROOT / "meta/episodes.jsonl", "w", encoding="utf-8") as f:
        for item in filtered_episodes:
            f.write(json.dumps(item) + "\n")
    
    print("✅ meta/episodes.jsonl 更新成功！现在应该可以加载了。")