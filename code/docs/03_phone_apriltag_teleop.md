# Experiment 2: Teleoperating Piper with a Phone and AprilTags

A phone replaces the dedicated leader arm. The phone's rear camera looks at a four-AprilTag board fixed on the desk; the phone web page detects the Tags in real time and the computer computes **the phone pose relative to the Tag board**, which after clutching, filtering and scaling becomes Piper's desired pose. On the Piper side, the same form of velocity-level QP as in Experiment 1 is iterated to convergence by PND before commands are sent. The phone page also shows the MuJoCo digital twin and the real camera view.

Code directory: [`AprilTag_Precision_Lab/`](../AprilTag_Precision_Lab) (depends on `mujoco_haircut/` and `Client_RHCR/piper/` in the same repository)

## 1. Setup

- The Piper follower, clipper, mannequin head and external USB camera are the same as in Experiment 1; the head position is shared through `mujoco_haircut/scene_config.json`.
- Four-Tag board: tag36h11, IDs 0 and 1 in the top row, IDs 2 and 3 in the bottom row, all patterns in the same orientation; black-border outer side 50 mm, 100 mm gap between borders (150 mm center spacing). The sizes can be changed to measured values in the web page. The world origin is the board center.
- The phone and computer are on the same LAN; use Chrome or Safari (not the WeChat built-in browser).
- The control point is the clipper tip; moving the phone forward maps to base +X, left to +Y, up to +Z (initial camera-to-base axes are z, −x, −y).

## 2. Control parameters

| Item | Value | Location |
| --- | --- | --- |
| Feedback gain k_e | 6 | `pnd_tracking.py` `PNDTracker(gain=6)` |
| Orientation weight w_r | 0.3 (w_p = 1) | `pnd_tracking.py` |
| Damping λ | 0.001 | `pnd_tracking.py` |
| PND iteration cap / convergence residual | 1500 / 1e-5 rad/s, iterated to convergence | `pnd_tracking.py` |
| Joint acceleration ramp | 240 °/s², velocity continuity weight 0.003 | `pnd_tracking.py` `SmoothPNDIK` |
| Tracking speed limits | end effector 100 mm/s, joints 45 °/s, orientation 30 °/s | `robot_bridge.py` |
| Tracking joint limits | J1 ±148, J2 [0,178], J3 [−168,0], J4 ±98, J5 ±68, J6 ±118 (°) | `robot_bridge.py` |
| Motion scale (phone : tip) | default 10:30, adjustable in the page from 10:1.5 to 10:200 | Motion scale in the web page |
| Initial joint angles | [14.352, 49.32, −57.868, 0, 38.629, 0]° | `initial_pose.py` |
| Homing speed | 15 °/s | `robot_bridge.py` |
| Head guard clearance | 10 mm (based on the MuJoCo head model) | `head_guard.py` |

## 3. Procedure

### 3.1 First-time preparation

1. Generate the HTTPS certificate and open firewall port 8443 as described in [01_installation.md](01_installation.md).
2. Print the four-Tag board, fix it flat on the desk, and measure the black-border side length and spacing with a ruler.
3. Double-click `AprilTag_Precision_Lab/start_phone_teleop_piper_pnd.bat`. It only starts the web service; it does not connect to or enable the arm. The console prints the address for the phone, `https://<PC LAN IP>:8443/piper/`.
4. Open that address on the phone; when warned that the certificate is not trusted, choose to proceed, and allow camera access. **Control from only one page at a time**; do not open the control page on the computer at the same time.

### 3.2 Camera intrinsics and Tag board

In **Camera & Tag Settings** on the web page:

1. Enter the measured black-border side length and the horizontal and vertical gaps.
2. Camera intrinsics, either:
   - Recommended: checkerboard calibration (default 9×6 inner corners, 20 mm squares; change to match your printout). Keeping the same lens, resolution and zoom, capture at least 15 views from different positions, tilts and distances, press **Calibrate**, then **Export Calibration** to save the JSON.
   - Quick: keep the Tag board fixed, view it from several angles, and press **Estimate Intrinsics** to estimate the focal length from the Tags (marked as an estimate; it does not replace checkerboard calibration).
3. Optional: if the spacing or sticker orientation is inaccurate, pause, keep IDs 0–3 visible together and slowly tilt the phone, then press **Calibrate Board Layout** to capture 16 frames and jointly optimize the board layout; a median reprojection error above 2 px is rejected.
4. Check in the **Camera Pose** panel that the reprojection error is below 3 px and at least 3 Tags are visible.

### 3.3 Haircut experiment

1. Make sure the clipper is mounted, nobody is in the workspace, and a hand is on the E-stop.
2. Press **Start** (the button then becomes the Resume / Pause clutch) and confirm the safety prompt. The page opens the phone camera and the real camera, connects to Piper (the backend defaults to the real robot; it can be switched to Mock under Connection & Maintenance), starts the MuJoCo twin, and Piper moves slowly at 15 °/s to the initial joint angles. It stays paused once there.
3. **Head alignment** (redo after moving the head; after a service restart the previous `results/calibration/head_alignment.json` is reused automatically):
   1. Press **Resume** to start following, and guide the clipper tip with the phone to **10 mm above** the center of the top of the head.
   2. Press **Align Head**: the arm holds its position, the program reads the CAN feedback while still, and translates the MuJoCo head to the corresponding position.
4. Press **Resume** with the phone still and the Tags in view; after 3 consecutive stable frames the reference is set (current phone pose ↔ current tip pose) and following starts.
5. Move the phone so the tip passes close to the head over the top, back, left and right in turn. When the phone reaches an uncomfortable position, press **Pause**, move it back to a comfortable position and press **Resume** (clutch); the arm does not jump back to the old reference.
6. When done, press **Finish & Save**: the arm returns to the initial pose, data is saved, reports and figures are generated, and the recording of this session is cut out.

### 3.4 Automatic pause and protection

- If Tag detection is lost for more than 0.8 s, the pose jumps in 3 consecutive frames, the reprojection error exceeds 3 px, the camera or intrinsics change, PND does not converge, CAN feedback goes stale or a driver reports an error, the arm holds still and pauses; when tracking recovers it **does not resume automatically**, and Resume must be pressed by hand.
- Every command first passes the head guard check (aligned head model + gripper + clipper): commands that would come within 10 mm of the head are not sent, while motion away from the head is always allowed.
- **Emergency Stop** stops immediately; **Return to Initial** returns to the initial pose and pauses.

## 4. Simulation-only mode (no real robot)

Double-click `start_tag_teleop_sim.bat` and open `https://<PC IP>:8443/` on the computer or phone. You can hold a Tag in front of the laptop camera, or film a fixed Tag with the phone, to drive Piper in MuJoCo without loading the SDK or connecting CAN. On the Piper page, the Mock backend can also be selected under **Connection & Maintenance** to run the full workflow on simulated hardware.

## 5. Output

- `results/experiments/hardware_pnd_<timestamp>/` (`mock_pnd_<timestamp>/` for Mock):
  - `signals.csv`: rd target, rc command FK, r measured end effector (position in m, rotation vector in rad), position/orientation errors, qc commanded / q measured joint angles (°), `solve_ms`, `qp_iters`
  - `events.jsonl`: raw/filtered phone poses, configuration, clutch and fault events, head guard interventions
  - `solver.jsonl`: PND residual, convergence flag, feedback age, speed-limit factor
  - `session.json`: data provenance; `figures/`: overview and individual paper figures (PDF + 600 dpi PNG)
- `results/pnd_<date>_<time>/`: reports in the same format as Experiment 1, `_figures/`, coverage and `video_*.mp4`
- `results/piper_videos/`: screen recordings made from the web page (H.264 MP4)

## 6. Notes

- A single planar marker has a two-fold pose ambiguity. The program picks, among candidates with similar error, the one continuous with the previous frame, but the initial anchoring can still be uncertain; it is more stable when 3 or more Tags of the board are visible.
- Low jitter at rest does not mean the absolute accuracy is right; a wrong Tag size or wrong intrinsics introduces scale error.
- The median phone sampling interval is about 24 ms, clearly slower than the 10 ms of the leader arm, and the mean position error with the phone is also clearly larger than in leader-arm mode.
- The head guard only covers the aligned head model in MuJoCo; it does not account for real human motion, calibration error or stopping distance.
