# Experiment 1: Heterogeneous Leader-Follower Teleoperation, SO-101 Leader → Piper Follower

The SO-101 leader sends only the **end-effector position and orientation** (6 values) to the follower; its joint angles are used only for its own forward kinematics and the MuJoCo display, never for follower control. The Piper follower uses the clipper tip as its control point: each cycle it builds a velocity-level QP from its own joint feedback, solves it with PND, and sends joint commands.

Code directory: [`mujoco_haircut/`](../mujoco_haircut)

## 1. Setup

- Piper follower; its base frame is the world frame (x forward, z up).
- SO-101 leader placed about 0.55 m to the left of Piper (`so101_leader.pos` in `scene_config.json`, display only).
- The clipper is held in the Piper gripper, with its tip 0.20 m along the flange z axis; the gripper is locked and does not follow the leader trigger (`--teleop.output_gripper=false`).
- The scanned mannequin head is fixed in front of Piper; its position is `head.pos` in `scene_config.json`, obtained by the calibration in the next section.
- An external USB camera faces the head and Piper (`real_camera.index` in `scene_config.json`, default 1) to record the real scene.

## 2. Control parameters

`run_haircut.bat` (baseline experiment, 60 Hz, PND iterated to convergence):

| Parameter | Command line | Value |
| --- | --- | --- |
| Feedback gain k_e | `--robot.pnd.gain` | 30 |
| Damping λ | `--robot.pnd.damping` | 0.001 |
| Position / orientation weights w_p, w_r | `PNDIKParams` defaults | 1, 0.1 |
| Joint speed limit | `--robot.pnd.max_joint_speed_deg_s` | 90 °/s |
| Control rate | `--fps` | 60 Hz |
| Follower start joint angles | `--robot.start_joints_deg` | [0, 53, -63, 0, 40, 0]° |
| Leader home joint angles | `--teleop.home_joints_deg` | [-32, -49.7, 46.7, 3.4, -106.5, 0]° |

The α comparison experiment in the paper (`pnd_alpha_experiment/run_alpha1.0.bat`, `run_alpha0.2.bat`) changes only the following relative to the table above:

| Parameter | Command line | Value |
| --- | --- | --- |
| Relaxation coefficient α | `--robot.pnd.relaxation` | 1.0 or 0.2 |
| Iterations per cycle N_max | `--robot.pnd.iterations` | 1 |
| Control rate | `FPS` | 100 Hz |
| Follower start joint angles | `--robot.start_joints_deg` | [0, 48, -44, 0, 10, 0]° (flange 5 cm lower than in the baseline) |
| Joint limits | `--robot.joint_limits_deg` | J1 ±150, J2 [15,150], J3 [-160,-15], J4 ±100, J5 ±65, J6 ±100 (°) |

## 3. Procedure

### 3.1 Preparation

1. Set up the environment as in [01_installation.md](01_installation.md) and calibrate the SO-101 leader (id `leader`).
2. Connect Piper's USB-CAN adapter and the SO-101 serial port; make sure the port matches `--teleop.port` in the `.bat` files.
3. Make sure nobody is inside Piper's workspace, the clipper is mounted, and a hand is on the E-stop.
4. Run `run_haircut.bat mock` once to check that the MuJoCo window, recording and result files all work.

### 3.2 Calibrating the head position (redo after moving the head)

1. Double-click `mujoco_haircut/calibrate_head.bat`. Piper first moves slowly to the start pose, then follows the leader.
2. Guide the clipper tip with the leader to lightly touch **the highest point at the center of the top of the head**. The top is fairly flat, so touching 1–2 cm off turns directly into horizontal position error; be as precise as possible.
3. Press **Enter** in the MuJoCo window to record (U undoes, R restarts); once the top of the model head lines up with the tip, press **Y** to save. The head is translated as a whole in MuJoCo and the result is written back to `head.pos` in `scene_config.json` (the old file is backed up automatically).
4. Press Ctrl+C to finish.

### 3.3 Teleoperated haircut (baseline experiment)

1. Double-click `mujoco_haircut/run_haircut.bat`. It opens the MuJoCo digital-twin window and starts recording automatically; Piper moves to the start pose and then follows the leader.
2. The operator moves the leader so the clipper tip passes close to the head surface over the four regions in turn: top, back, left and right. The top-left of the window shows the total and per-region coverage and the tip-to-scalp distance in real time.
3. MuJoCo window shortcuts: **V** starts/stops recording, **C** clears the trajectory, **F** toggles follow/free view.
4. When the haircut is done, press **Ctrl+C** in the console; the follower homes on exit, and the program stops recording and generates the reports. The MuJoCo window stays open.

### 3.4 α comparison experiment (data for the table / box plots in the paper)

1. Double-click `mujoco_haircut/pnd_alpha_experiment/run_alpha1.0.bat`. Teleoperation starts once the follower reaches the ready pose; after cutting over the whole head, press Ctrl+C to finish.
2. Double-click `run_alpha0.2.bat` and perform similar motions. With a small α the follower lags, so move the leader more slowly and keep a hand on the E-stop.
3. Double-click `analyze_alpha_compare.bat`; the console prints a comparison table and writes `summary.csv`. Columns:
   - `dt_ms`: median actual sampling period, which should be close to 10 ms at 100 Hz; clearly above 12 ms means the loop cannot keep up, so change `FPS` in both bat files back to 60 and rerun.
   - `pos_mm` / `rot_deg`: mean position / orientation tracking error (feedback pose relative to the desired pose).
   - `near_pct`, `pos_near`, `pos_away`, etc.: a sample counts as "near a limit" when any commanded joint angle is within 0.5° of its limit; errors are grouped accordingly.
   - `acc_rms_meas`: RMS of the measured joint acceleration; smaller is smoother.
   - `iters_max`: should be 1.
4. Generate the comparison figures in the paper:

   ```bat
   python analysis\plot_alpha_box.py      <csv for α=1> <csv for α=0.2> <output dir>
   python analysis\plot_alpha_compare.py  <csv for α=1> <csv for α=0.2> <output dir>
   python analysis\plot_tracking_error.py <csv for α=1> <csv for α=0.2> <output dir>
   ```

Results are saved under `pnd_alpha_experiment/results/a<α>_fps100_j5wide_low5cm/pnd_<date>_<time>/`, separate from `results/` of the baseline experiment.

## 4. Output

Each run creates the following under `results/pnd_<date>_<time>/`:

- `pnd_*.csv`: per-cycle log (desired, commanded and feedback poses, commanded/measured joint angles, errors, solve time, QP iterations)
- `pnd_*.html`: status and tracking report; `pnd_*_ieee.html`: compact figure cards
- `pnd_*_haircut.html`: hair coverage report (four views + coverage curve + per-region coverage)
- `pnd_*_figures/`: paper figures, PDF (embedded fonts) + 600 dpi PNG: position, orientation, errors, solver performance, J1–J3, J4–J6, four coverage views, coverage ratio
- `video_*.mp4`: gripper view and haircut progress view in the left column, main view in the middle, real camera on the right (2560×720)

Column definitions are in [04_data_format_and_analysis.md](04_data_format_and_analysis.md).

## 5. Changing the algorithm

- `haircut_sim/pnd_ik.py`: `build_qp()` sets H, f and the velocity bounds; `solve()` implements the projection neurodynamics.
- `haircut_sim/control/piper_follower.py`: reads follower feedback, maps the target, solves and sends commands.
- `haircut_sim/control/qp_ik.py`, `mrnn_ik.py`: earlier algorithms kept for comparison, selected with `--robot.ik_solver`.
- `scene_config.json`: head model, clipper offset, hair region and views.

## 6. Notes

- Coverage is geometric: the fraction of hair sample points on the head surface within 30 mm of the clipper tip. It is not the same as the result of a real haircut.
- The cutting path comes entirely from the operator; the program does no automatic path planning.
- If Piper does not move and the console shows `SendCanMessage(SEND_MESSAGE_FAILED)`, the USB-CAN adapter is usually held by another program; close it and try again.
