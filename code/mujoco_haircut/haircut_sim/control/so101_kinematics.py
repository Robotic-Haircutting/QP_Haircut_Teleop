"""SO-101 forward kinematics (no placo / URDF parsing needed).

Joint origins are taken from Simulation/SO101/so101_new_calib.urdf in TheRobotStudio/SO-ARM100,
consistent with lerobot's built-in EE pipeline (RobotKinematics + so101_new_calib.urdf, target_frame=gripper_frame_link):
the inputs are lerobot-calibrated joint angles (deg, use_degrees=True), which map directly to the URDF joint angles.
"""

import numpy as np

from .transforms import make_tf, rot_z, rpy_to_matrix

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# (xyz, rpy): fixed transform from the parent link frame to the joint frame; each joint rotates about its own z axis
_JOINT_ORIGINS = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),  # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),  # shoulder_lift
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),  # elbow_flex
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),  # wrist_flex
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),  # wrist_roll
]
# gripper_link -> gripper_frame_link (TCP near the center of the gripper fingertips)
_TCP_ORIGIN = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))

_FIXED = [make_tf(rpy_to_matrix(*rpy), xyz) for xyz, rpy in _JOINT_ORIGINS]
_TCP = make_tf(rpy_to_matrix(*_TCP_ORIGIN[1]), _TCP_ORIGIN[0])


def so101_fk(joints_deg, joint_offsets_deg=None) -> np.ndarray:
    """joints_deg: 5 joint angles (deg, ordered as JOINT_NAMES). Returns the 4x4 TCP pose in base_link (m)."""
    q = np.deg2rad(np.asarray(joints_deg, dtype=float)[:5])
    if joint_offsets_deg is not None:
        q = q + np.deg2rad(np.asarray(joint_offsets_deg, dtype=float)[:5])
    T = np.eye(4)
    for fixed, qi in zip(_FIXED, q, strict=True):
        T = T @ fixed @ make_tf(rot_z(qi), (0, 0, 0))
    return T @ _TCP
