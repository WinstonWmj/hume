import dataclasses
import logging
import socket

import tyro

# from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from hume.models import CometPolicy
from hume.serving import websocket_policy_server
from hume.shared.eval_b1k_wrapper import B1KPolicyWrapper


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script for CometPolicy."""

    # If provided, will be used to retrieve the prompt of the task, otherwise use turning_on_radio as default.
    task_name: str | None = None
    
    # Port to serve the policy on.
    port: int = 8000

    ckpt_path: str | None = None

    # Specifies the fine-grained level of the policy.
    fine_grained_level: int = 0

    # Specifies the control mode of the policy.
    control_mode: str = "receeding_horizon" # receeding_horizon | temporal_ensemble | receeding_temporal

    # Specifies the action horizon of the policy.
    max_len: int = 32  # receeding horizon | receeding temporal mode
    action_horizon: int = 5  # temporal ensemble mode
    temporal_ensemble_max: int = 3  # receeding temporal mode

    resize_size: int = 224
    replan_steps: int = 32  # CometPolicy uses chunk_size=32

    post_process_action: bool = True

    # VQH candidate selection parameters (now used by CometPolicy)
    use_vqh_selection: bool = True  # Enable VQH-based candidate selection
    s2_candidates_num: int = 5      # Number of candidate actions to generate
    noise_temp_lower_bound: float = 1.0
    noise_temp_upper_bound: float = 2.0
    time_temp_lower_bound: float = 0.9
    time_temp_upper_bound: float = 1.0
    

def main(args: Args) -> None:
    logging.info(f"Using task_name: {args.task_name}")
    logging.info(f"Loading CometPolicy from: {args.ckpt_path}")
    
    # Load CometPolicy instead of HumePolicy
    comet_policy = CometPolicy.from_pretrained(args.ckpt_path).to("cuda")
    comet_policy.init_infer(
        infer_cfg=dict(
            replan_steps=args.replan_steps,
            post_process_action=args.post_process_action,
            device="cuda",
            # VQH candidate selection parameters
            use_vqh_selection=args.use_vqh_selection,
            s2_candidates_num=args.s2_candidates_num,
            noise_temp_bounds=(args.noise_temp_lower_bound, args.noise_temp_upper_bound),
            time_temp_bounds=(args.time_temp_lower_bound, args.time_temp_upper_bound),
        )
    )
    
    policy = B1KPolicyWrapper(
        comet_policy,
        task_name=args.task_name,
        control_mode=args.control_mode,
        max_len=args.replan_steps,
        action_horizon=args.action_horizon,
        temporal_ensemble_max=args.temporal_ensemble_max,
        fine_grained_level=args.fine_grained_level,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # for debug
    import pickle
    with open("/mnt/project_rlinf/mjwei/repo/hume/.cursor/debug_obs.pkl", "rb") as f:
        obs = pickle.load(f)
    # then directly call policy.act(obs) to debug
    action = policy.act(obs)
    print(action)
    return
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={},
    )
    logging.info("server running")
    server.serve_forever()


if __name__ == "__main__":
    # import debugpy
    # # 5678 是监听端口，你可以随便改
    # # wait_for_client() 会暂停程序直到你连上 Debugger，防止错过启动时的断点
    # print("Waiting for debugger attach on port 5678...")
    # debugpy.listen(("localhost", 5678))
    # debugpy.wait_for_client() 
    # print("Debugger attached!")
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
