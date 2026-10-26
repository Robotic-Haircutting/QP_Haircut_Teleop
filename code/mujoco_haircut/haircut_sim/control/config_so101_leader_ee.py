from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig

from .config_so101_leader_home import SO101LeaderHomeConfig


@TeleoperatorConfig.register_subclass("haircut_so101_leader_ee")
@dataclass
class SO101LeaderEEConfig(SO101LeaderHomeConfig):
    """SO-101 leader arm that outputs the end-effector pose (forward kinematics) + gripper opening. Auto-homing at start/end uses the same parameters as so101_leader_home.

    Shares the calibration file with the stock so101_leader: the same --teleop.id reads
    ~/.cache/huggingface/lerobot/calibration/teleoperators/{so_leader or so101_leader}/<id>.json
    """

    # Extra per-joint offsets (deg) added before FK, to fine-tune the difference between the calibration zero and the URDF zero; normally 0
    joint_offsets_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # Also output the raw joint angles (shoulder_pan.pos etc.), handy when switching piper_follower to joint mode or comparing recorded data
    include_joints: bool = False
    output_gripper: bool = True  # False: send only six pose fields
    # Runs without a leader arm: outputs a slow sinusoidal joint motion (integration testing only)
    mock: bool = False
