import dataclasses
import logging
import socket

import tyro

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from hume.models import HumePolicy
from hume.serving import websocket_policy_server
from hume.shared.eval_b1k_wrapper import B1KPolicyWrapper


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

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


def main(args: Args) -> None:
    logging.info(f"Using task_name: {args.task_name}")
    hume = HumePolicy.from_pretrained(args.ckpt_path).to("cuda")
    hume = B1KPolicyWrapper(
        hume,
        task_name=args.task_name,
        control_mode=args.control_mode,
        max_len=args.max_len,
        action_horizon=args.action_horizon,
        temporal_ensemble_max=args.temporal_ensemble_max,
        fine_grained_level=args.fine_grained_level,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=hume,
        host="0.0.0.0",
        port=args.port,
        metadata={},
    )
    logging.info("server running")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
