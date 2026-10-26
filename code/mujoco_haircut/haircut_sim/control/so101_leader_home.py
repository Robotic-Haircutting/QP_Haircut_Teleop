import logging
import time

import numpy as np
from lerobot.teleoperators.so_leader.so_leader import SOLeader

from .config_so101_leader_home import SO101LeaderHomeConfig

logger = logging.getLogger(__name__)

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_FULL_TORQUE = 1000


class SO101LeaderHome(SOLeader):
    """Same output as the stock so101_leader, but returns to the home pose automatically after connecting and before disconnecting.

    Start: after homing, holds with low torque (so it does not fall under gravity); torque is released automatically as soon as the operator pushes any joint more than release_threshold_deg.
    End: after homing, exits with torque on and the leader stays in place; the next run takes over again.
    """

    config_class = SO101LeaderHomeConfig
    name = "so_leader"  # shares the calibration directory with the stock leader

    def __init__(self, config: SO101LeaderHomeConfig):
        super().__init__(config)
        self.config = config
        self._holding = False
        self._home = np.asarray(config.home_joints_deg, dtype=float)

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        if self.config.home_on_connect:
            self.go_home()
            self._set_torque_limit(self.config.hold_torque_limit)
            time.sleep(0.3)
            # Use the position where the arm actually settles after lowering torque as the reference, so the steady-state sag under gravity is not mistaken for a "hand push"
            settled = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
            self._hold_ref = np.array([settled[m] for m in MOTORS], dtype=float)
            self._holding = True
            logger.info("Leader homed and holding; grab it and push any joint to release")

    def get_action(self) -> dict[str, float]:
        action = super().get_action()
        if self._holding:
            q = np.array([action[f"{m}.pos"] for m in MOTORS], dtype=float)
            if np.max(np.abs(q - self._hold_ref)) > self.config.release_threshold_deg:
                self._release()
                logger.info("Hand on the leader detected; torque released, teleoperation starts")
        return action

    def disconnect(self) -> None:
        keep = False
        if self.is_connected and self.config.home_on_disconnect:
            try:
                self.go_home(self.config.exit_joints_deg)
                keep = self.config.hold_after_exit
            except Exception as e:  # close the serial port even if homing fails
                logger.warning(f"Leader homing failed: {e}")
        if keep:
            self._set_torque_limit(self.config.hold_torque_limit)
            self.bus.disconnect(disable_torque=False)
            logger.info(f"{self} homed; exiting with torque on (the next run takes over automatically)")
        else:
            self._release()
            super().disconnect()

    def _release(self) -> None:
        self.bus.disable_torque()
        self._set_torque_limit(_FULL_TORQUE)
        self._holding = False

    def _set_torque_limit(self, value: int) -> None:
        for m in MOTORS:
            self.bus.write("Torque_Limit", m, int(value), normalize=False)

    def go_home(self, target=None) -> None:
        """Enable torque, interpolate smoothly from the current position to target (home_joints_deg by default), and return with torque on."""
        goal = self._home if target is None else np.asarray(target, dtype=float)
        start = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        q0 = np.array([start[m] for m in MOTORS], dtype=float)
        logger.info(f"Leader homing: {np.round(q0, 1)} → {np.round(goal, 1)}; do not pull on the leader")
        self._set_torque_limit(_FULL_TORQUE)
        # Set the goal to the current position before enabling torque, to avoid a jump at the moment torque comes on
        self.bus.sync_write("Goal_Position", dict(zip(MOTORS, q0.tolist(), strict=True)))
        self.bus.enable_torque()
        n = max(1, int(self.config.home_time_s * 50))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)  # slow at both ends, fast in the middle
            q = q0 + s * (goal - q0)
            self.bus.sync_write("Goal_Position", dict(zip(MOTORS, q.tolist(), strict=True)))
            time.sleep(1 / 50)
        time.sleep(0.4)
        self._holding = False
