from hume import HumePolicy
import numpy as np

# load policy
pretrained_policy="/mnt/mnt/public_zgc/models/Hume-vla/Hume-System2/"
hume = HumePolicy.from_pretrained(pretrained_policy).to("cuda")

# config Test-Time Computing args
hume.init_infer(
    infer_cfg=dict(
        replan_steps=8,
        s2_replan_steps=16,
        s2_candidates_num=5,
        noise_temp_lower_bound=1.0,
        noise_temp_upper_bound=1.0,
        time_temp_lower_bound=0.9,
        time_temp_upper_bound=1.0,
        post_process_action=True,
        device="cuda",
    )
)

# prepare observations
observation = {
    "observation.images.image": np.zeros((1,224,224,3), dtype = np.uint8), # (B, H, W, C)
    "observation.images.wrist_image": np.zeros((1,224,224,3), dtype = np.uint8), # (B, H, W, C)
    "observation.state": np.zeros((1, 8)), # (B, state_dim) match the state dimension of the corresponding model
    "task": ["Lift the papper"],
}

# Infer the action
action = hume.infer(observation) # (B, action_dim)
print(action)