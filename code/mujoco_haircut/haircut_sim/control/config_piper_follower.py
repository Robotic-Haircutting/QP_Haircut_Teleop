from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from .mrnn_ik import MRNNIKParams
from .qp_ik import QPIKParams
from ..pnd_ik import PNDIKParams


@RobotConfig.register_subclass("haircut_piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    """AgileX Piper follower arm (piper_sdk, CAN)."""

    # CAN interface name (activate it first with can_activate.sh from piper_sdk)
    can_port: str = "can0"
    # No hardware: run the pipeline with a fake in-memory arm
    mock: bool = False

    # "ee": with so101_leader_ee, sends the end-effector pose (Piper solves IK internally)
    # "joint": with the stock so101_leader, maps joint angles directly (5 SO-101 joints → Piper J1 J2 J3 J5 J6, J4 fixed)
    control_mode: str = "ee"
    # "relative": use each arm's pose at the moment teleoperation starts as its origin and follow increments only (recommended; no zero alignment needed)
    # "absolute": in ee mode p_piper = scale * R_align * p_leader + ee_offset_m; in joint mode q = sign * q_leader + offset
    mapping: str = "relative"

    # Motion speed percentage inside the Piper controller (move_spd_rate_ctrl of MotionCtrl_2, 0-100)
    speed_rate: int = 50

    # ---------- ee mode ----------
    # Leader displacement scale (Piper's reach is about 1.5-2x that of the SO-101)
    position_scale: float = 1.5
    # Rotation (deg) about z that aligns the leader base frame with the Piper base frame (change this when the arms face different directions)
    base_yaw_offset_deg: float = 0.0
    # Translation for absolute mapping (m)
    ee_offset_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Control point in qp mode: tip position relative to the Piper flange (link6) (m). When non-zero, the clipper tip (instead of the flange) follows the leader end effector;
    # teleop_sim fills it in automatically from tool.tip_offset_m in scene_config.json
    tool_offset_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Workspace clipping (Piper base frame, m)
    workspace_min_m: list[float] = field(default_factory=lambda: [0.05, -0.40, 0.08])
    workspace_max_m: list[float] = field(default_factory=lambda: [0.55, 0.40, 0.55])
    # Maximum combined horizontal + vertical distance from the base z axis (m); Piper's reach is about 0.626 m
    max_reach_m: float = 0.58
    # Minimum distance from the shoulder joint (m). The joint safety ranges (J2>=15°, J3<=-15°) do not let the arm fold completely,
    # so targets too close to the shoulder may be unreachable; when >0, rd is first pushed outside this spherical shell. Real-robot replay showed lowering qp.w_rot works better, so off by default
    min_reach_m: float = 0.0
    # End-effector speed limits
    max_linear_speed_m_s: float = 0.15
    max_angular_speed_deg_s: float = 90.0

    # ---------- joint mode ----------
    # Signs for SO-101 [pan, lift, elbow, wrist_flex, wrist_roll] → Piper [J1, J2, J3, J5, J6]; calibrate after assembly as described in the README
    joint_signs: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])
    # Offsets for absolute mapping (deg)
    joint_offsets_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # Fixed angle of J4 (forearm roll) (deg); in relative mode the starting angle is kept
    joint4_deg: float = 0.0
    max_joint_speed_deg_s: float = 120.0
    # Allowed range of each Piper joint during teleoperation (deg, [[min,max]]*6). Tighter than the official ranges:
    # J2 not below 15° and J3 not above -15°, avoiding the fully folded pose (when folded, wrist J5 hits the forearm and stalls); J5 limited to ±50°
    joint_limits_deg: list[list[float]] = field(
        default_factory=lambda: [[-150, 150], [15, 150], [-160, -15], [-100, 100], [-50, 50], [-100, 100]]
    )

    # After connecting, first move slowly at start_speed_rate% to this joint pose (deg, J1-J6), then start teleoperation; None = do not move
    # The folded pose (all 0) sits at the J2/J3 limits and the wrist can hit the forearm; a half-extended pose is recommended, e.g. [0, 40, -40, 0, 0, 0]
    start_joints_deg: list[float] | None = None
    start_speed_rate: int = 15
    # Whether to also return slowly to start_joints_deg before disconnecting (Ctrl+C to end)
    return_on_disconnect: bool = True

    # ---------- qp mode ----------
    # With so101_leader_ee: leader end-effector target → local quadratic-programming IK (qp_ik.py) → joint commands.
    # Example command-line overrides: --robot.qp.w_rot=0.5 --robot.qp.max_joint_speed_deg_s=45
    # Beyond the QP joint ranges, commands are clipped once more by joint_limits_deg above before being sent
    qp: QPIKParams = field(default_factory=QPIKParams)
    # Solver used in qp mode: "mrnn" = momentum recurrent neural network from the paper (mrnn_ik.py, default), "qp" = box-constrained QP active-set method (qp_ik.py)
    ik_solver: str = "pnd"
    pnd: PNDIKParams = field(default_factory=PNDIKParams)
    # MRNN parameters, command-line example: --robot.mrnn.momentum=0.8 --robot.mrnn.task=position
    mrnn: MRNNIKParams = field(default_factory=MRNNIKParams)
    # Each frame writes rd / actual end effector r / errors to a CSV (qp_time.csv) in this folder; None disables logging. Plot with: python tools/plot_tracking.py
    log_dir: str | None = None

    # ---------- Gripper ----------
    # Piper gripper opening (mm) corresponding to leader gripper.pos 100
    gripper_max_mm: float = 70.0
    gripper_effort: float = 1.0  # N·m, 0-5
    invert_gripper: bool = False

    # Whether to disable the motors on disconnect. Note: once disabled, Piper falls under gravity, so they stay enabled by default
    disable_on_disconnect: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.control_mode not in ("ee", "joint", "qp"):
            raise ValueError(f"control_mode must be ee, joint or qp, got {self.control_mode}")
        if self.ik_solver not in ("mrnn", "qp", "pnd"):
            raise ValueError(f"ik_solver must be mrnn, qp or pnd, got {self.ik_solver}")
        if self.mapping not in ("relative", "absolute"):
            raise ValueError(f"mapping must be relative or absolute, got {self.mapping}")
