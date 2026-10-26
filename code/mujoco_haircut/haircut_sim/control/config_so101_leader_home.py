from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig


@TeleoperatorConfig.register_subclass("haircut_so101_leader_home")
@dataclass
class SO101LeaderHomeConfig(TeleoperatorConfig, SOLeaderConfig):
    """Stock so101_leader + automatic return to the home pose at start and end.

    Homing enables torque, interpolates slowly to home_joints_deg, then releases torque to the operator's hand once there. Shares the calibration file with so101_leader.
    """

    # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] (deg; gripper is 0-100)
    # Should put the arm in the same posture as Piper's start_joints_deg
    home_joints_deg: list[float] = field(default_factory=lambda: [-32.0, -49.7, 46.7, 3.4, -106.5, 0.0])
    # Homing duration (s); larger is slower
    home_time_s: float = 3.0
    home_on_connect: bool = True
    home_on_disconnect: bool = True
    # Torque limit while holding after homing (0-1000); lower makes it easier to push by hand
    hold_torque_limit: int = 800
    # At start, torque is released once any joint is pushed more than this angle (deg) away from the home pose
    release_threshold_deg: float = 4.0
    # Pose the leader returns to at the end. Servos holding against gravity for long trip their overload protection and drop torque, so by default it returns to a folded pose that rests on itself before releasing
    exit_joints_deg: list[float] = field(default_factory=lambda: [-28.3, -103.3, 96.9, 65.2, -39.8, 0.0])
    # Keep torque on when exiting (only recommended if the exit pose itself is not loaded by gravity)
    hold_after_exit: bool = False
