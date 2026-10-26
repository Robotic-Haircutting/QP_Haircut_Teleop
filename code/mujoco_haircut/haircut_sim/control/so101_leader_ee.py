import logging
import math
import time

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config_so101_leader_ee import SO101LeaderEEConfig
from .so101_kinematics import JOINT_NAMES, so101_fk
from .so101_leader_home import SO101LeaderHome
from .transforms import matrix_to_rotvec

logger = logging.getLogger(__name__)

EE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"]
# Old and new lerobot versions use different calibration directory names for the SO-101 leader; search both
_CALIB_DIR_NAMES = ["so_leader", "so101_leader", "so100_leader"]


class SO101LeaderEE(SO101LeaderHome):
    """Read the SO-101 leader joint angles and output the TCP pose in base_link via forward kinematics.

    action: ee.x/y/z (m), ee.wx/wy/wz (rotation vector, rad), gripper.pos (0-100, 0 = closed)
    """

    config_class = SO101LeaderEEConfig
    name = "so_leader"  # selects the default calibration directory, same as the stock SO-101 leader

    def __init__(self, config: SO101LeaderEEConfig):
        if not config.use_degrees:
            raise ValueError("so101_leader_ee requires use_degrees=true (forward kinematics works in degrees)")
        if config.calibration_dir is None and config.id is not None:
            for dir_name in _CALIB_DIR_NAMES:
                d = HF_LEROBOT_CALIBRATION / TELEOPERATORS / dir_name
                if (d / f"{config.id}.json").is_file():
                    config.calibration_dir = d
                    logger.info(f"Reusing leader calibration file {d / f'{config.id}.json'}")
                    break
        super().__init__(config)
        self.config = config
        self._mock_connected = False
        self._t0 = time.perf_counter()

    @property
    def action_features(self) -> dict[str, type]:
        ft = {k: float for k in EE_KEYS}
        if self.config.output_gripper:
            ft["gripper.pos"] = float
        if self.config.include_joints:
            ft.update({f"{j}.pos": float for j in JOINT_NAMES})
        return ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._mock_connected if self.config.mock else super().is_connected

    @property
    def is_calibrated(self) -> bool:
        return True if self.config.mock else super().is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.config.mock:
            self._mock_connected = True
            self._t0 = time.perf_counter()
            logger.info(f"{self} connected (mock).")
            return
        super().connect(calibrate)

    def read_joints(self) -> dict[str, float]:
        if self.config.mock:
            t = time.perf_counter() - self._t0
            return {
                "shoulder_pan.pos": 20 * math.sin(0.5 * t),
                "shoulder_lift.pos": 10 * math.sin(0.3 * t),
                "elbow_flex.pos": 10 * math.sin(0.4 * t),
                "wrist_flex.pos": 15 * math.sin(0.6 * t),
                "wrist_roll.pos": 30 * math.sin(0.2 * t),
                "gripper.pos": 50 + 50 * math.sin(0.7 * t),
            }
        return super().get_action()

    def get_action(self) -> dict[str, float]:
        joints = self.read_joints()
        T = so101_fk([joints[f"{j}.pos"] for j in JOINT_NAMES], self.config.joint_offsets_deg)
        rv = matrix_to_rotvec(T[:3, :3])
        action = dict(zip(EE_KEYS, [*T[:3, 3].tolist(), *rv.tolist()], strict=True))
        if self.config.output_gripper:
            action["gripper.pos"] = float(joints["gripper.pos"])
        if self.config.include_joints:
            action.update({f"{j}.pos": float(joints[f"{j}.pos"]) for j in JOINT_NAMES})
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        if self.config.mock:
            self._mock_connected = False
            return
        super().disconnect()
