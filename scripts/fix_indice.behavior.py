#!/usr/bin/env python3
"""
Fix behavior dataset episode indices to be consecutive starting from 0.

This script:
1. Renames parquet files (data/*.parquet)
2. Updates parquet content (episode_index column)
3. Renames video files (videos/*/*.mp4)
4. Updates episodes.jsonl
5. Updates episodes_stats.jsonl
6. Updates meta/info.json (chunk structure)
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

# Configuration
BEHAVIOR_ROOT = Path('/mnt/project_rlinf/mjwei/download_models/behavior')
BACKUP_ROOT = Path('/mnt/mnt/public_zgc/datasets/behavior_backup')
DRY_RUN = False  # Set to False to actually perform changes
NEW_CHUNKS_SIZE = 1000  # Use standard lerobot chunk size
DO_BACKUP = False  # Backup before making changes

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    if DRY_RUN:
        print(f"  [DRY_RUN] Would save: {path}")
        return
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def save_jsonl(path, items):
    if DRY_RUN:
        print(f"  [DRY_RUN] Would save: {path}")
        return
    with open(path, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')

def main():
    print("=" * 70)
    print("FIX BEHAVIOR DATASET EPISODE INDICES")
    print("=" * 70)
    print(f"Root: {BEHAVIOR_ROOT}")
    print(f"Dry run: {DRY_RUN}")
    print()

    # Load metadata
    info = load_json(BEHAVIOR_ROOT / 'meta/info.json')
    episodes = load_jsonl(BEHAVIOR_ROOT / 'meta/episodes.jsonl')
    episodes_stats = load_jsonl(BEHAVIOR_ROOT / 'meta/episodes_stats.jsonl')
    
    old_chunks_size = info['chunks_size']
    print(f"Old chunks_size: {old_chunks_size}")
    print(f"New chunks_size: {NEW_CHUNKS_SIZE}")
    print(f"Total episodes: {len(episodes)}")
    print()

    # Build old->new episode index mapping
    old_to_new = {}
    for new_idx, ep in enumerate(episodes):
        old_idx = ep['episode_index']
        old_to_new[old_idx] = new_idx
    
    print("Episode index mapping (first 10):")
    for old_idx, new_idx in list(old_to_new.items())[:10]:
        print(f"  {old_idx:>8} -> {new_idx}")
    print()

    # Determine video keys from info
    video_keys = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
    print(f"Video keys ({len(video_keys)}):")
    for vk in video_keys:
        print(f"  - {vk}")
    print()

    # Prepare file operations
    file_ops = []  # list of (old_path, new_path, type)
    
    # 1. Parquet files - need to rename AND update content
    print("=" * 70)
    print("PARQUET FILE OPERATIONS")
    print("=" * 70)
    for old_idx, new_idx in old_to_new.items():
        old_chunk = old_idx // old_chunks_size
        new_chunk = new_idx // NEW_CHUNKS_SIZE
        
        old_path = BEHAVIOR_ROOT / info['data_path'].format(
            episode_chunk=old_chunk, episode_index=old_idx
        )
        
        new_data_path = f"data/chunk-{new_chunk:03d}/episode_{new_idx:06d}.parquet"
        new_path = BEHAVIOR_ROOT / new_data_path
        
        if old_path.exists():
            file_ops.append((old_path, new_path, 'parquet', old_idx, new_idx))
        else:
            print(f"  WARNING: Missing parquet: {old_path}")
    
    print(f"Parquet files to process: {len([f for f in file_ops if f[2] == 'parquet'])}")
    for op in file_ops[:5]:
        print(f"  {op[0].name} -> {op[1].parent.name}/{op[1].name}")
    print("  ...")
    print()

    # 2. Video files
    print("=" * 70)
    print("VIDEO FILE OPERATIONS")
    print("=" * 70)
    video_ops = []
    for old_idx, new_idx in old_to_new.items():
        old_chunk = old_idx // old_chunks_size
        new_chunk = new_idx // NEW_CHUNKS_SIZE
        
        for vid_key in video_keys:
            old_video_path_template = info['video_path'].format(
                episode_chunk=old_chunk, video_key=vid_key, episode_index=old_idx
            )
            old_video_path = BEHAVIOR_ROOT / old_video_path_template
            
            new_video_path = BEHAVIOR_ROOT / f"videos/chunk-{new_chunk:03d}/{vid_key}/episode_{new_idx:06d}.mp4"
            
            if old_video_path.exists():
                video_ops.append((old_video_path, new_video_path, 'video'))
            else:
                print(f"  WARNING: Missing video: {old_video_path}")
    
    print(f"Video files to process: {len(video_ops)}")
    for op in video_ops[:5]:
        rel_old = op[0].relative_to(BEHAVIOR_ROOT)
        rel_new = op[1].relative_to(BEHAVIOR_ROOT)
        print(f"  {rel_old} -> {rel_new}")
    print("  ...")
    print()
    
    file_ops.extend(video_ops)

    # 3. Update episodes.jsonl
    print("=" * 70)
    print("UPDATE EPISODES.JSONL")
    print("=" * 70)
    new_episodes = []
    for new_idx, ep in enumerate(episodes):
        new_ep = ep.copy()
        new_ep['episode_index'] = new_idx
        new_episodes.append(new_ep)
    print(f"Episodes to update: {len(new_episodes)}")
    for ep in new_episodes[:5]:
        print(f"  {ep}")
    print("  ...")
    print()

    # 4. Update episodes_stats.jsonl
    print("=" * 70)
    print("UPDATE EPISODES_STATS.JSONL")
    print("=" * 70)
    # Build mapping from old episode_index to stats
    old_stats = {s['episode_index']: s for s in episodes_stats}
    new_episodes_stats = []
    for new_idx, ep in enumerate(episodes):
        old_idx = ep['episode_index']
        if old_idx in old_stats:
            new_stat = old_stats[old_idx].copy()
            new_stat['episode_index'] = new_idx
            new_episodes_stats.append(new_stat)
        else:
            print(f"  WARNING: No stats for old episode {old_idx}")
    print(f"Episode stats to update: {len(new_episodes_stats)}")
    print()

    # 5. Update info.json
    print("=" * 70)
    print("UPDATE INFO.JSON")
    print("=" * 70)
    new_info = info.copy()
    new_info['chunks_size'] = NEW_CHUNKS_SIZE
    new_info['data_path'] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    new_info['video_path'] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    new_info['total_chunks'] = (len(episodes) + NEW_CHUNKS_SIZE - 1) // NEW_CHUNKS_SIZE
    
    # Remove custom paths
    if 'metainfo_path' in new_info:
        del new_info['metainfo_path']
    if 'annotation_path' in new_info:
        del new_info['annotation_path']
    
    print(f"New data_path: {new_info['data_path']}")
    print(f"New video_path: {new_info['video_path']}")
    print(f"New chunks_size: {new_info['chunks_size']}")
    print(f"New total_chunks: {new_info['total_chunks']}")
    print()

    if DRY_RUN:
        print("=" * 70)
        print("DRY RUN - No changes made")
        print("=" * 70)
        print("To apply changes, set DRY_RUN = False and run again.")
        return

    # Backup before making changes
    if DO_BACKUP:
        print("=" * 70)
        print("BACKING UP DATA")
        print("=" * 70)
        print(f"Backup location: {BACKUP_ROOT}")
        
        if BACKUP_ROOT.exists():
            print(f"Backup already exists at {BACKUP_ROOT}, skipping backup.")
        else:
            BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
            
            # Backup meta folder
            print("Backing up meta/...")
            shutil.copytree(BEHAVIOR_ROOT / 'meta', BACKUP_ROOT / 'meta')
            
            # Backup data folder
            print("Backing up data/...")
            shutil.copytree(BEHAVIOR_ROOT / 'data', BACKUP_ROOT / 'data')
            
            # Backup videos folder (this may take a while)
            print("Backing up videos/ (this may take a while)...")
            shutil.copytree(BEHAVIOR_ROOT / 'videos', BACKUP_ROOT / 'videos')
            
            print("Backup completed!")
        print()

    # Execute changes
    print("=" * 70)
    print("EXECUTING CHANGES")
    print("=" * 70)
    
    # Create new chunk directories
    for chunk in range((len(episodes) + NEW_CHUNKS_SIZE - 1) // NEW_CHUNKS_SIZE):
        (BEHAVIOR_ROOT / f"data/chunk-{chunk:03d}").mkdir(parents=True, exist_ok=True)
        for vid_key in video_keys:
            (BEHAVIOR_ROOT / f"videos/chunk-{chunk:03d}/{vid_key}").mkdir(parents=True, exist_ok=True)
    
    # Process parquet files (rename and update content)
    parquet_ops = [op for op in file_ops if op[2] == 'parquet']
    print(f"\nProcessing {len(parquet_ops)} parquet files...")
    for op in tqdm(parquet_ops, desc="Parquet files"):
        old_path, new_path, _, old_idx, new_idx = op
        
        # Read parquet, update episode_index, write to new location
        table = pq.read_table(old_path)
        
        # Update episode_index column
        arrays = []
        names = []
        for col_name in table.column_names:
            if col_name == 'episode_index':
                # Replace with new index
                new_col = pa.array([new_idx] * len(table), type=pa.int64())
                arrays.append(new_col)
            else:
                arrays.append(table.column(col_name))
            names.append(col_name)
        
        new_table = pa.table(dict(zip(names, arrays)))
        new_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, new_path)
        
        # Delete old file after successful write
        old_path.unlink()
    
    # Process video files (just rename)
    video_ops_list = [op for op in file_ops if op[2] == 'video']
    print(f"\nProcessing {len(video_ops_list)} video files...")
    for op in tqdm(video_ops_list, desc="Video files"):
        old_path, new_path, _ = op
        
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
    
    # Save metadata
    print("\nSaving metadata...")
    save_jsonl(BEHAVIOR_ROOT / 'meta/episodes.jsonl', new_episodes)
    save_jsonl(BEHAVIOR_ROOT / 'meta/episodes_stats.jsonl', new_episodes_stats)
    save_json(BEHAVIOR_ROOT / 'meta/info.json', new_info)
    
    # Cleanup old directories
    print("\nCleaning up old directories...")
    for d in (BEHAVIOR_ROOT / 'data').iterdir():
        if d.is_dir() and d.name.startswith('task-'):
            if not any(d.iterdir()):
                d.rmdir()
                print(f"  Removed empty: {d}")
    
    for d in (BEHAVIOR_ROOT / 'videos').iterdir():
        if d.is_dir() and d.name.startswith('task-'):
            # Remove empty subdirs
            for subd in d.iterdir():
                if subd.is_dir() and not any(subd.iterdir()):
                    subd.rmdir()
            if not any(d.iterdir()):
                d.rmdir()
                print(f"  Removed empty: {d}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()

