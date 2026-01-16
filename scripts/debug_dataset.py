#!/usr/bin/env python3
"""Debug script to trace the episode index issue"""

import sys
sys.path.insert(0, 'src')

from hume.training.dataset import LeRobotDataset
from torch.utils.data import DataLoader

# Monkey patch _query_videos to add debugging
original_query_videos = LeRobotDataset._query_videos

def debug_query_videos(self, query_timestamps, ep_idx):
    print(f"\n=== _query_videos called ===")
    print(f"ep_idx: {ep_idx}")
    print(f"ep_idx type: {type(ep_idx)}")
    print(f"chunks_size: {self.meta.chunks_size}")
    episode_chunk = self.meta.get_episode_chunk(ep_idx)
    print(f"episode_chunk: {episode_chunk}")
    
    for vid_key in list(query_timestamps.keys())[:1]:  # Just print first key
        video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
        print(f"Video path: {video_path}")
        print(f"Exists: {video_path.exists()}")
        break
    
    return original_query_videos(self, query_timestamps, ep_idx)

LeRobotDataset._query_videos = debug_query_videos

# Create dataset
dataset = LeRobotDataset(
    repo_id='behavior',
    root='/mnt/project_rlinf/mjwei/download_models/behavior',
    episodes=None,
    slide=1,
    s1_action_steps=5,
    video_backend='pyav',
    download_videos=False,
)

print(f"Dataset length: {len(dataset)}")
print(f"Total episodes: {dataset.num_episodes}")

# Try to load first batch
dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

print("\n===  Trying to load first batch ===")
try:
    batch = next(iter(dataloader))
    print(f"Success! Episode index: {batch['episode_index'].item()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

