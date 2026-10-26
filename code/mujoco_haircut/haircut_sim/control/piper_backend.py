"""Piper hardware abstraction: PiperSDKBackend uses piper_sdk + CAN, MockPiperBackend simulates in memory.

Common units: position in m, orientation as rpy in rad (Piper convention R = Rz·Ry·Rx), joints in deg, gripper opening in mm.
"""

import logging
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

# Piper end-effector pose after power-on (piper_sdk C_PiperForwardKinematics.init_pos, mm / deg)
_PIPER_INIT_POSE = [56.128, 0.0, 213.266, 0.0, 85.0, 0.0]


def _open_gs_usb_bus(can_port: str):
    import os

    import can
    import libusb_package

    # pyusb finds libusb-1.0.dll through PATH; use the one bundled with libusb-package
    dll_dir = os.path.dirname(str(libusb_package.get_library_path()))
    if dll_dir not in os.environ["PATH"]:
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
    digits = "".join(c for c in can_port if c.isdigit())
    index = int(digits) if digits else 0
    return can.Bus(interface="gs_usb", channel=index, index=index, bitrate=1000000)


class PiperSDKBackend:
    def __init__(self, can_port: str):
        self.can_port = can_port
        self.piper = None

    def connect(self, timeout_s: float = 5.0) -> None:
        from piper_sdk import C_PiperInterface_V2

        if sys.platform == "win32":
            # Windows has no socketcan: the candleLight/gs_usb adapter is accessed directly via libusb. can_port "0" and "can0" both mean the first adapter
            from piper_sdk.hardware_port import C_STD_CAN

            class _GsUsbCAN(C_STD_CAN):
                def _create_bus(self):
                    return _open_gs_usb_bus(self.channel_name)

            self.piper = C_PiperInterface_V2(self.can_port, judge_flag=False, can_auto_init=False)
            arm_can = _GsUsbCAN(self.can_port, "gs_usb", 1000000, False, False, self.piper.ParseCANFrame)
            self.piper._C_PiperInterface_V2__arm_can = arm_can
        else:
            self.piper = C_PiperInterface_V2(self.can_port)
        self.piper.ConnectPort()

        t0 = time.perf_counter()
        while self.piper.GetArmJointMsgs().time_stamp == 0:
            if time.perf_counter() - t0 > timeout_s:
                raise ConnectionError(
                    f"No Piper feedback within {timeout_s}s; check that {self.can_port} is activated (can_activate.sh) and Piper is powered on"
                )
            time.sleep(0.01)

        self.recover(timeout_s)
        # Enable the gripper at its current opening so it does not clamp or open right after connecting
        self.send_gripper(self.get_gripper_mm(), 1.0, enable=True)
        logger.info(f"Piper on {self.can_port} enabled.")

    def recover(self, timeout_s: float = 5.0) -> None:
        """Clear joint driver faults (stall/overcurrent etc.) and re-enable all joints. set_zero must stay 0."""
        self.piper.JointConfig(joint_num=7, set_zero=0x00, acc_param_is_effective=0x00, clear_err=0xAE)
        time.sleep(0.1)
        t0 = time.perf_counter()
        while not self.piper.EnablePiper():
            if time.perf_counter() - t0 > timeout_s:
                raise ConnectionError(f"Piper could not be enabled within {timeout_s}s; check the E-stop and power")
            time.sleep(0.01)

    def faulted_joints(self) -> list[int]:
        """Return the numbers (1-6) of joints that are disabled or in error."""
        low = self.piper.GetArmLowSpdInfoMsgs()
        bad = []
        for i in range(1, 7):
            st = getattr(low, f"motor_{i}").foc_status
            if st.driver_error_status or st.stall_status or not st.driver_enable_status:
                bad.append(i)
        return bad

    def disconnect(self, disable: bool) -> None:
        if self.piper is None:
            return
        if disable:
            t0 = time.perf_counter()
            while self.piper.DisablePiper() and time.perf_counter() - t0 < 2.0:
                time.sleep(0.01)
        self.piper.DisconnectPort()
        self.piper = None

    def get_joints_deg(self) -> np.ndarray:
        js = self.piper.GetArmJointMsgs().joint_state
        return np.array([js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]) / 1000.0

    def get_ee(self) -> tuple[np.ndarray, np.ndarray]:
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        pos = np.array([ep.X_axis, ep.Y_axis, ep.Z_axis]) / 1e6  # 0.001mm -> m
        rpy = np.deg2rad(np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis]) / 1000.0)
        return pos, rpy

    def get_gripper_mm(self) -> float:
        return self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0

    def send_ee(self, pos_m, rpy_rad, speed_rate: int) -> None:
        x, y, z = (round(v * 1e6) for v in pos_m)
        rx, ry, rz = (round(v * 1000) for v in np.rad2deg(rpy_rad))
        self.piper.MotionCtrl_2(0x01, 0x00, int(speed_rate), 0x00)  # CAN control, MOVE P
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)

    def send_joints(self, joints_deg, speed_rate: int) -> None:
        j = [round(v * 1000) for v in joints_deg]
        self.piper.MotionCtrl_2(0x01, 0x01, int(speed_rate), 0x00)  # CAN control, MOVE J
        self.piper.JointCtrl(*j)

    def send_gripper(self, opening_mm: float, effort_nm: float, enable: bool = True) -> None:
        self.piper.GripperCtrl(
            round(abs(opening_mm) * 1000), round(np.clip(effort_nm, 0, 5) * 1000), 0x01 if enable else 0x00, 0
        )


class MockPiperBackend:
    """Fake Piper whose commands take effect immediately. In joint mode the end-effector pose is updated with piper_sdk forward kinematics (when piper_sdk is installed)."""

    def __init__(self):
        self.joints = np.zeros(6)
        self.pos = np.array(_PIPER_INIT_POSE[:3]) / 1000.0
        self.rpy = np.deg2rad(_PIPER_INIT_POSE[3:])
        self.gripper_mm = 0.0
        self._fk = None
        self.n_ee_cmds = 0
        self.n_joint_cmds = 0

    def connect(self) -> None:
        try:
            from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

            self._fk = C_PiperForwardKinematics(dh_is_offset=0x01)
        except ImportError:
            self._fk = None
        logger.info("Mock Piper connected.")

    def disconnect(self, disable: bool) -> None:
        pass

    def recover(self, timeout_s: float = 5.0) -> None:
        pass

    def faulted_joints(self) -> list[int]:
        return []

    def get_joints_deg(self) -> np.ndarray:
        return self.joints.copy()

    def get_ee(self) -> tuple[np.ndarray, np.ndarray]:
        return self.pos.copy(), self.rpy.copy()

    def get_gripper_mm(self) -> float:
        return self.gripper_mm

    def send_ee(self, pos_m, rpy_rad, speed_rate: int) -> None:
        self.pos = np.asarray(pos_m, dtype=float).copy()
        self.rpy = np.asarray(rpy_rad, dtype=float).copy()
        self.n_ee_cmds += 1

    def send_joints(self, joints_deg, speed_rate: int) -> None:
        self.joints = np.asarray(joints_deg, dtype=float).copy()
        if self._fk is not None:
            tcp = self._fk.CalFK(np.deg2rad(self.joints).tolist())[-1]
            self.pos = np.array(tcp[:3]) / 1000.0
            self.rpy = np.deg2rad(tcp[3:])
        self.n_joint_cmds += 1

    def send_gripper(self, opening_mm: float, effort_nm: float, enable: bool = True) -> None:
        self.gripper_mm = float(opening_mm)
