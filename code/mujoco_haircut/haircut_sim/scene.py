"""Build the haircut simulation scene: Piper follower + SO-101 leader + mannequin head, all configured in scene_config.json.

World frame = Piper base frame (x forward, z up, in m).
The model is kinematic display only: joint angles are written externally, then mj_forward is called; no dynamics simulation.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DEFAULT_CONFIG = ROOT / "scene_config.json"

PIPER_JOINTS = [f"piper/joint{i}" for i in range(1, 7)]
PIPER_FINGERS = ("piper/joint7", "piper/joint8")
PIPER_NEW_FW_OFFSETS_DEG = [0.0, 2.0, -2.0, 0.0, 0.0, 0.0]
PIPER_FINGER_MAX_M = 0.035  # travel per finger; both fingers open to 70 mm in total
SO101_JOINTS = [f"so101/{n}" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
SO101_GRIPPER = "so101/gripper"
SO101_GRIPPER_MAX_RAD = 0.785398  # leader trigger 0-45°, corresponds to lerobot gripper.pos 0-100


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_dir"] = str(path.resolve().parent)
    return cfg


def _quat_from_euler_deg(euler_deg) -> list[float]:
    """Rotate about the fixed axes x→y→z in turn (R = Rz·Ry·Rx) and return a MuJoCo wxyz quaternion."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.deg2rad(np.asarray(euler_deg, dtype=float)), "XYZ")
    return q.tolist()


def _resolve(cfg: dict, p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = Path(cfg["_dir"]) / path
    return str(path.resolve())


def _attach(world_spec: mujoco.MjSpec, xml: Path, prefix: str, pos, euler_deg) -> None:
    child = mujoco.MjSpec.from_file(str(xml))
    for key in list(child.keys):
        child.delete(key)
    for light in list(child.lights):
        child.delete(light)
    frame = world_spec.worldbody.add_frame(pos=list(pos), quat=_quat_from_euler_deg(euler_deg))
    child.option.impratio = world_spec.option.impratio
    child.option.integrator = world_spec.option.integrator
    child.option.cone = world_spec.option.cone
    world_spec.attach(child, prefix=prefix, frame=frame)


def add_head(spec: mujoco.MjSpec, head: dict, cfg: dict) -> None:
    """Head: any STL/OBJ mesh. Shown only as a static object in the scene (with collision geometry for future contact work)."""
    pos = np.asarray(head["pos"], dtype=float)
    stand_h = float(head.get("stand_height_m", 0.0))
    if stand_h > 0:
        spec.worldbody.add_geom(
            name="head_stand",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[float(head.get("stand_radius_m", 0.04)), stand_h / 2, 0],
            pos=[pos[0], pos[1], stand_h / 2],
            rgba=[0.35, 0.35, 0.38, 1],
        )
    body = spec.worldbody.add_body(name="head", pos=[pos[0], pos[1], pos[2] + stand_h], quat=_quat_from_euler_deg(head.get("euler_deg", [0, 0, 0])))
    file = head.get("file")
    if file:
        scale = float(head.get("scale", 0.001))
        spec.add_mesh(name="head_mesh", file=_resolve(cfg, file), scale=[scale] * 3)
        body.add_geom(name="head", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="head_mesh", rgba=head.get("rgba", [0.93, 0.80, 0.70, 1]))
    else:
        # Placeholder head: ellipsoid head + cylinder neck
        body.add_geom(name="neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.05, 0], pos=[0, 0, 0.05], rgba=head.get("rgba", [0.93, 0.80, 0.70, 1]))
        body.add_geom(name="head", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.10, 0.08, 0.12], pos=[0, 0, 0.19], rgba=head.get("rgba", [0.93, 0.80, 0.70, 1]))


def add_clipper(spec: mujoco.MjSpec, tool: dict) -> None:
    """Clipper: a cylinder attached to Piper link6 along the flange z axis, with its tip at tool.tip_offset_m. Display only, no collisions."""
    c = tool.get("clipper", {})
    if not c.get("enabled", False):
        return
    tip = np.asarray(tool.get("tip_offset_m", [0.0, 0.0, 0.14]), dtype=float)
    half = float(c.get("length_m", 0.14)) / 2
    spec.body("piper/link6").add_geom(
        name="clipper",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[float(c.get("radius_m", 0.02)), half, 0],
        pos=(tip - np.array([0.0, 0.0, half])).tolist(),
        rgba=c.get("rgba", [0.15, 0.15, 0.17, 1.0]),
        contype=0,
        conaffinity=0,
    )


def build_spec(cfg: dict) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "haircut_scene"
    spec.compiler.degree = False
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]

    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.84, 0.86], rgb2=[0.74, 0.76, 0.78], width=300, height=300)
    mat = spec.add_material(name="floor", texrepeat=[8, 8], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05], material="floor")
    spec.worldbody.add_light(name="top", pos=[0.3, 0.2, 2.0], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])

    p, s = cfg["piper"], cfg["so101_leader"]
    _attach(spec, ASSETS / "piper" / "piper.xml", "piper/", p["pos"], [0, 0, p.get("yaw_deg", 0.0)])
    _attach(spec, ASSETS / "so101" / "so101_leader.xml", "so101/", s["pos"], [0, 0, s.get("yaw_deg", 0.0)])
    add_clipper(spec, cfg.get("tool", {}))
    if cfg.get("head", {}).get("enabled", True):
        add_head(spec, cfg["head"], cfg)
    return spec


class Scene:
    """Compiled scene + an interface for writing joint angles in real-robot units."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.spec = build_spec(cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        adr = lambda name: int(self.model.jnt_qposadr[self.model.joint(name).id])  # noqa: E731
        self._piper = [adr(n) for n in PIPER_JOINTS]
        self._piper_fingers = [adr(n) for n in PIPER_FINGERS]
        self._so101 = [adr(n) for n in SO101_JOINTS]
        self._so101_grip = adr(SO101_GRIPPER)
        self._so101_signs = np.asarray(cfg["so101_leader"].get("joint_signs", [1] * 5), dtype=float)
        self._so101_offsets = np.deg2rad(np.asarray(cfg["so101_leader"].get("joint_offsets_deg", [0] * 5), dtype=float))
        # Menagerie's Piper geometry follows the old piper_sdk DH (J2/J3 zero offset of 2°); new firmware (dh_is_offset=1) needs [0,2,-2,0,0,0] added
        self._piper_offsets = np.deg2rad(np.asarray(cfg["piper"].get("joint_offsets_deg", PIPER_NEW_FW_OFFSETS_DEG), dtype=float))
        self.set_piper(cfg["piper"].get("init_joints_deg", [0, 0, 0, 0, 0, 0]), 0.0)
        self.set_so101([0, 0, 0, 0, 0], 0.0)
        mujoco.mj_forward(self.model, self.data)

    def set_piper(self, joints_deg, gripper_mm=None) -> None:
        """Piper joint feedback (deg, J1-J6, same signs and zeros as piper_sdk GetArmJointMsgs) + gripper opening (mm)."""
        self.data.qpos[self._piper] = np.deg2rad(np.asarray(joints_deg, dtype=float)[:6]) + self._piper_offsets
        if gripper_mm is not None:
            f = float(np.clip(gripper_mm / 1000.0 / 2.0, 0.0, PIPER_FINGER_MAX_M))
            self.data.qpos[self._piper_fingers[0]] = f
            self.data.qpos[self._piper_fingers[1]] = -f

    def set_so101(self, joints_deg, gripper_pos=None) -> None:
        """SO-101 leader joint angles after lerobot calibration (deg, use_degrees, consistent with the URDF joint angles) + gripper.pos (0-100)."""
        q = np.deg2rad(np.asarray(joints_deg, dtype=float)[:5])
        self.data.qpos[self._so101] = self._so101_signs * q + self._so101_offsets
        if gripper_pos is not None:
            self.data.qpos[self._so101_grip] = float(np.clip(gripper_pos, 0, 100)) / 100.0 * SO101_GRIPPER_MAX_RAD

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
