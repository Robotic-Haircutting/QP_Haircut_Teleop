# Data Format and Analysis

## 1. Tracking CSV (shared by both experiments)

`pnd_*.csv` from Experiment 1 and `signals.csv` from Experiment 2 have the same column order (Experiment 2 adds extra columns). Units: position in m, orientation as a rotation vector in rad, joint angles in °.

| Column | Meaning |
| --- | --- |
| `t` | Seconds since start |
| `rd_x … rd_wz` | Desired end-effector pose (from the leader arm or the phone, converted to the Piper base frame) |
| `rc_x … rc_wz` | End-effector pose from forward kinematics of the joint command (command-side tracking) |
| `r_x … r_wz` | End-effector pose from forward kinematics of the measured Piper joint angles (actual tracking, including motor lag) |
| `e_pos_mm`, `e_rot_deg` | Position error (mm) and orientation error (norm of the relative rotation vector, °) of r relative to rd |
| `ec_pos_mm`, `ec_rot_deg` | Error of rc relative to rd |
| `qc1 … qc6` / `q1 … q6` | Joint commands / measured joint angles |
| `solve_ms`, `qp_iters` | QP solve time and number of PND iterations in this cycle |

## 2. Statistics

- Position error $e_{p,i} = 1000\,\|p_{r,i} - p_{d,i}\|$ (mm); for the command-side error, $p_r$ is replaced by $p_c$.
- Orientation error is the norm of the rotation vector of the relative rotation $R_d^\top R$ (°), computed from rotation matrices rather than by subtracting Euler angles.
- Near a limit: any **commanded** joint angle within 0.5° of its limit. The limits must match `--robot.joint_limits_deg` of that run (J5 is ±65° in the α experiment); otherwise the grouping changes.
- Coverage: fraction of hair sample points on the head model within 30 mm of the clipper tip (`cover_radius_m` in `scene_config.json`). This is geometric coverage on the model, not the result of a real haircut.

## 3. Analysis scripts

| Script | Purpose |
| --- | --- |
| `mujoco_haircut/pnd_alpha_experiment/analyze_alpha.py <results directory>` | Summarizes the sampling period, mean errors, limit grouping, joint acceleration RMS and solve time of each run, and writes `summary.csv` |
| `analysis/plot_alpha_box.py A1 A02 OUT` | Box plots of position/orientation errors for α=1 and α=0.2 (whiskers at 5–95%, triangles mark the mean) |
| `analysis/plot_alpha_compare.py A1 A02 OUT` | Cumulative error distribution curves; also prints the tracking lag (ms) estimated from the rd→rc cross-correlation |
| `analysis/plot_tracking_error.py A1 A02 OUT` | Position/orientation error over time for the two runs |
| `mujoco_haircut/haircut_sim/long_tracking_report.py`, `figures.py` | Report and paper figures generated automatically at the end of each run (can also be regenerated from an existing CSV) |
| `AprilTag_Precision_Lab/session_report.py` | Session report for the phone experiment, in the same output format as Experiment 1 |

Figures follow IEEE journal requirements: single column 3.5 in, double column 7.16 in, PDF with embedded fonts, PNG at 600 dpi.

## 4. Data not included in the repository

Raw CSV files (2–20 MB each) and videos (over 100 MB per segment) are not in the repository. If they are to be published, consider Zenodo, Figshare or a GitHub Release, and add the download link to the README on the repository front page.
