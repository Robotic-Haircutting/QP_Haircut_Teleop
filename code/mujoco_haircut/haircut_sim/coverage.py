"""Haircut coverage: how much of the hair area the clipper-tip trajectory has swept.

- Clipper tip = Piper flange (link6) offset by tool.tip_offset_m along its own z axis (default 0.14 m, gripper fingertips).
- Hair region = the part of the head mesh above hair.min_height_frac, minus the face (the patch facing hair.face_axis_local).
  When hair.front_hairline_frac is set, a "hairline" is used instead: anything on the face side beyond that line (forehead, face) is not hair,
  and the rest is split into four haircut regions (top / back / left / right), each with its own coverage.
- A vertex counts as "cut" when it is closer than hair.cover_radius_m to any trajectory point; coverage is weighted by the area each vertex represents.
"""

import mujoco
import numpy as np

DEFAULT_HAIR = {
    "min_height_frac": 0.55,  # hair is above 55% of the total head height (from the top of the ears up)
    "face_axis_local": [0, 1, 0],  # direction the face points in the head model's own frame (+y for HeadScan.stl)
    "face_exclude_deg": 100,  # within 100° left/right of the face direction (face and sideburns)...
    "face_exclude_below_frac": 0.85,  # ...and below 85% height is excluded, leaving only the top and back of the head
    "cover_radius_m": 0.03,  # tip within 3 cm of the hair surface counts as cut (application check, not a precise measure)
    "grid_m": 0.0053,  # hair region sampled on a ~5 mm grid (~2000 points), one point per cell
}
REGIONS = ["top", "back", "left", "right"]  # left/right from the person's own point of view
REGION_NAMES = {"top": "Top", "back": "Back", "left": "Left", "right": "Right"}
DEFAULT_TIP_OFFSET_M = [0.0, 0.0, 0.14]


def tip_offset(cfg: dict) -> np.ndarray:
    return np.asarray(cfg.get("tool", {}).get("tip_offset_m", DEFAULT_TIP_OFFSET_M), dtype=float)


def tip_pos(model, data, offset: np.ndarray) -> np.ndarray:
    b = model.body("piper/link6").id
    return data.xpos[b] + data.xmat[b].reshape(3, 3) @ offset


class HairCoverage:
    def __init__(self, model, data, cfg: dict):
        h = {**DEFAULT_HAIR, **cfg.get("hair", {})}
        self.radius = float(h["cover_radius_m"])
        gid = model.geom("head").id
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Coverage needs a head mesh (head.file must not be empty)")
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = (data.geom_xmat[gid].reshape(3, 3) @ model.mesh_vert[va:va + vn].T).T + data.geom_xpos[gid]
        faces = model.mesh_face[fa:fa + fn]
        tri = verts[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        vert_area = np.zeros(vn)
        for k in range(3):
            np.add.at(vert_area, faces[:, k], area / 3.0)

        z = verts[:, 2]
        frac = (z - z.min()) / max(z.max() - z.min(), 1e-9)
        self.center = np.array([(verts[:, 0].min() + verts[:, 0].max()) / 2, (verts[:, 1].min() + verts[:, 1].max()) / 2, z.min()])
        R_head = data.xmat[model.body("head").id].reshape(3, 3)
        f = R_head @ np.asarray(h["face_axis_local"], dtype=float)
        f[2] = 0.0
        self.face_dir = f / max(np.linalg.norm(f), 1e-9)
        horiz = verts[:, :2] - self.center[:2]
        horiz /= np.maximum(np.linalg.norm(horiz, axis=1, keepdims=True), 1e-9)
        cos_face = horiz @ self.face_dir[:2]
        rel = verts[:, :2] - self.center[:2]
        fwd = rel @ self.face_dir[:2]  # distance toward the face
        lat = rel @ np.array([-self.face_dir[1], self.face_dir[0]])  # distance toward the person's left (z × face)
        if h.get("front_hairline_frac") is not None:
            # Hairline: anything beyond (half the head depth × front_hairline_frac) toward the face is not hair (no hair on the forehead by default)
            fwd_n = fwd / max(fwd.max(), 1e-9)
            hair = (frac >= h["min_height_frac"]) & (fwd_n <= float(h["front_hairline_frac"]))
        else:
            is_face = (cos_face > np.cos(np.deg2rad(h["face_exclude_deg"]))) & (frac < h["face_exclude_below_frac"])
            hair = (frac >= h["min_height_frac"]) & ~is_face
        # Haircut regions: above top_frac is the top; the rest is split horizontally into back / left / right
        back = -fwd >= np.abs(lat) * np.tan(np.deg2rad(float(h.get("back_half_angle_deg", 45))))
        region = np.where(frac >= float(h.get("top_frac", 0.85)), 0, np.where(back, 1, np.where(lat > 0, 2, 3)))

        self.all_verts = verts
        pts, area, reg = verts[hair], vert_area[hair], region[hair]
        self.hair_verts = pts  # full-resolution hair vertices (for plotting)
        self.vert_cell = np.arange(len(pts))  # which grid point each hair vertex belongs to
        grid = float(h.get("grid_m", 0.0))
        if grid > 0 and len(pts):
            # Merge each grid cell into one point: area-weighted mean position of its vertices, areas summed
            _, inv = np.unique(np.floor(pts / grid).astype(np.int64), axis=0, return_inverse=True)
            inv = inv.ravel()
            n = inv.max() + 1
            a = np.bincount(inv, weights=area, minlength=n)
            p = np.stack([np.bincount(inv, weights=area * pts[:, k], minlength=n) for k in range(3)], axis=1) / np.maximum(a, 1e-12)[:, None]
            counts = np.zeros((n, len(REGIONS)))
            np.add.at(counts, (inv, reg), 1)
            pts, area, reg = p, a, counts.argmax(axis=1)  # a cell belongs to the region of most of its vertices
            self.vert_cell = inv
        self.pts = pts
        self.area = area
        self.region = np.asarray(reg, dtype=int)  # region of each point, indexing REGIONS
        self.total_area = float(self.area.sum())
        self.covered = np.zeros(len(self.pts), dtype=bool)

    def reset(self) -> None:
        self.covered[:] = False

    def update(self, p: np.ndarray) -> tuple[np.ndarray, float]:
        """Add a clipper-tip position. Returns (indices of newly cut hair vertices, nearest distance from the tip to the hair surface in m)."""
        d2 = np.sum((self.pts - p) ** 2, axis=1)
        new = np.flatnonzero(~self.covered & (d2 < self.radius**2))
        self.covered[new] = True
        return new, float(np.sqrt(d2.min()))

    @property
    def vert_covered(self) -> np.ndarray:
        """Whether each full-resolution hair vertex is cut (follows its grid point)."""
        return self.covered[self.vert_cell]

    @property
    def ratio(self) -> float:
        return float(self.area[self.covered].sum() / self.total_area) if self.total_area > 0 else 0.0

    def region_ratios(self) -> dict[str, float]:
        """Coverage of each haircut region (top/back/left/right)."""
        out = {}
        for k, name in enumerate(REGIONS):
            m = self.region == k
            tot = float(self.area[m].sum())
            out[name] = float(self.area[m & self.covered].sum() / tot) if tot > 0 else 0.0
        return out
