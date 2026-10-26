"""lerobot-teleoperate with a synchronized MuJoCo display. Arguments are exactly those of lerobot-teleoperate, plus two optional ones:

  python -m haircut_sim.teleop_sim <all lerobot-teleoperate arguments> [--sim.config=xxx.json] [--sim.no_viewer]

What it does:
  1. Opens the MuJoCo window (haircut_sim.viewer) in a separate process.
  2. Piper: a background thread reads the joint feedback and gripper opening already cached by piper_sdk at 60 Hz and sends them to the window over UDP.
     Read-only, no control commands are sent; homing after connecting, teleoperation and homing on exit are all visible.
  3. SO-101 leader: each time the teleoperation loop reads the joint angles it also sends one UDP packet (about 10 µs); no extra serial reads.
Control and algorithms come from haircut_sim/control and haircut_sim/pnd_ik.py in this project; no external robot plugin is needed.
"""

import logging
import sys
import threading
import time

from .publisher import DEFAULT_PORT, SimPublisher, spawn_viewer
from .scene import load_config

logger = logging.getLogger("haircut_sim")
SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _poll_piper(robot, pub: SimPublisher, stop: threading.Event, hz: float) -> None:
    period = 1.0 / hz
    while not stop.is_set():
        try:
            pub.send_piper(robot.backend.get_joints_deg(), robot.backend.get_gripper_mm())
        except Exception:
            pass  # not connected yet / already disconnected
        stop.wait(period)


def _hook_piper(pub: SimPublisher, hz: float) -> None:
    from .control.piper_follower import PiperFollower

    orig_connect, orig_disconnect = PiperFollower.connect, PiperFollower.disconnect

    def connect(self, *args, **kwargs):
        self._sim_stop = threading.Event()
        threading.Thread(target=_poll_piper, args=(self, pub, self._sim_stop, hz), daemon=True, name="piper-sim").start()
        return orig_connect(self, *args, **kwargs)

    def disconnect(self, *args, **kwargs):
        try:
            return orig_disconnect(self, *args, **kwargs)
        finally:
            if getattr(self, "_sim_stop", None) is not None:
                self._sim_stop.set()

    PiperFollower.connect, PiperFollower.disconnect = connect, disconnect


def _leader_degrees(leader, action: dict) -> list[float]:
    """Convert the leader output to URDF joint angles (deg). With use_degrees=False (-100 to 100) the calibration range is used to convert back to angles."""
    q = [float(action[f"{j}.pos"]) for j in SO101_JOINTS]
    if getattr(leader.config, "use_degrees", True):
        return q
    cal = leader.calibration
    return [v / 200.0 * (cal[j].range_max - cal[j].range_min) * 360.0 / 4095.0 for v, j in zip(q, SO101_JOINTS)]


def _hook_leader(pub: SimPublisher) -> None:
    try:
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
    except ImportError:  # older lerobot
        from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader as SOLeader

    orig_get_action = SOLeader.get_action

    def get_action(self, *args, **kwargs):
        action = orig_get_action(self, *args, **kwargs)
        try:
            if all(f"{j}.pos" in action for j in SO101_JOINTS):
                pub.send_so101(_leader_degrees(self, action), action.get("gripper.pos"))
        except Exception:
            pass
        return action

    SOLeader.get_action = get_action

    # mock mode of so101_leader_ee does not go through SOLeader; hook it separately
    try:
        from .control.so101_leader_ee import SO101LeaderEE
    except ImportError:
        return
    if hasattr(SO101LeaderEE, "read_joints"):
        orig_read = SO101LeaderEE.read_joints

        def read_joints(self, *args, **kwargs):
            joints = orig_read(self, *args, **kwargs)
            if self.config.mock:
                try:
                    pub.send_so101([joints[f"{j}.pos"] for j in SO101_JOINTS], joints.get("gripper.pos"))
                except Exception:
                    pass
            return joints

        SO101LeaderEE.read_joints = read_joints


def install_hooks(port: int = DEFAULT_PORT, piper_hz: float = 60.0) -> SimPublisher:
    pub = SimPublisher(port)
    _hook_piper(pub, piper_hz)
    _hook_leader(pub)
    return pub


def _teleoperate_main():
    # Register only this project's devices; never scan external robot plugins.
    from . import control
    control.register_devices()
    from lerobot.scripts import lerobot_teleoperate

    def register_local_devices():
        return control

    lerobot_teleoperate.register_third_party_plugins = register_local_devices
    return lerobot_teleoperate.main


def _make_plots(log_dir: str, since: float, sim_config: str | None) -> None:
    """After teleoperation ends: plot the tracking figures for this run's log (tools/plot_tracking.py bundled with the plugin) and the haircut coverage report."""
    from pathlib import Path

    files = [p for p in Path(log_dir).glob("*.csv") if p.stat().st_mtime >= since]
    if not files:
        logger.info("No tracking log for this run; skipping plots")
        return
    csv_path = max(files, key=lambda p: p.stat().st_mtime)
    try:
        from .long_tracking_report import make_long_report
        make_long_report(csv_path)
    except Exception as e:
        logger.warning(f"Long state report generation failed: {e}")
    figures_ready = False
    try:
        from .figures import make_figures
        make_figures(csv_path, sim_config)
        figures_ready = True
    except Exception as e:
        logger.warning(f"Paper figure generation failed: {e}")
    try:
        from .tracking_report import make_tracking_report
        make_tracking_report(csv_path, figures_ready)
    except Exception as e:
        logger.warning(f"State report generation failed: {e}")
    try:
        from .report import make_report
        make_report(csv_path, sim_config, open_browser=False)
    except Exception as e:
        logger.warning(f"Coverage report generation failed: {e}")


def _make_run_dir(base: str | None, args: list[str]):
    """One subfolder per run, <base>/<solver>_<date>_<time>, holding the CSV, report, _figures and mp4. base defaults to mujoco_haircut/results."""
    from datetime import datetime
    from pathlib import Path

    opts = dict(a[2:].split("=", 1) for a in args if a.startswith("--robot.") and "=" in a)
    mode = opts.get("robot.control_mode", "qp")
    algo = opts.get("robot.ik_solver", "pnd") if mode == "qp" else mode
    root = Path(base) if base else Path(__file__).resolve().parent.parent / "results"
    run_dir = root / f"{algo}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results of this run are saved to {run_dir}", flush=True)
    return run_dir


def _tidy_results(root, keep) -> None:
    """Clean up previous runs: delete empty folders; for folders with a CSV but no report (the console was closed last time), generate the missing report."""
    from pathlib import Path

    root, keep = Path(root), Path(keep).resolve()
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.resolve() != keep):
        if not any(d.iterdir()):
            d.rmdir()
            continue
        csvs = [c for c in d.glob("*.csv") if c.stat().st_size > 0]
        if csvs and not any(d.glob("*_haircut.html")):
            logger.info(f"Generating the report missed last time: {d.name}")
            _make_plots(str(d), 0.0, None)


def main() -> None:
    sim_config, viewer, plots, calibrate = None, True, True, False
    rest = []
    log_dir = None
    extra: list[str] = []
    for a in sys.argv[1:]:
        if a.startswith("--sim.config="):
            sim_config = a.split("=", 1)[1]
        elif a == "--sim.no_viewer":
            viewer = False
        elif a == "--sim.no_plots":
            plots = False
        elif a == "--sim.record":
            extra.append("--record")
        elif a == "--sim.calibrate":
            calibrate, plots = True, False
        else:
            if a.startswith("--robot.log_dir="):
                log_dir = a.split("=", 1)[1].strip('"')
            rest.append(a)
    t_start = time.time() - 1.0
    port = load_config(sim_config).get("udp_port", DEFAULT_PORT)
    # The clipper tip is Piper's control point: without a manual --robot.tool_offset_m, use tool.tip_offset_m from scene_config.json
    if any(a == "--robot.type=haircut_piper_follower" for a in rest) and not any(a.startswith("--robot.tool_offset_m=") for a in rest):
        tip = load_config(sim_config).get("tool", {}).get("tip_offset_m")
        if tip is not None:
            rest.append(f"--robot.tool_offset_m=[{','.join(str(float(v)) for v in tip)}]")
    # Without --robot.log_dir (e.g. self-test scripts) no results folder is created and nothing is logged, as before
    if not calibrate and log_dir:
        log_dir = str(_make_run_dir(log_dir, rest))
        print("To finish, press Ctrl+C in this console window (closing the console directly leaves no time to generate the report; it will be generated on the next run)", flush=True)
        rest = [a for a in rest if not a.startswith("--robot.log_dir=")] + [f"--robot.log_dir={log_dir}"]
        extra += ["--out_dir", log_dir]

    logging.basicConfig(level=logging.INFO)
    proc = spawn_viewer(sim_config, port, extra + (["--calibrate"] if calibrate else [])) if viewer else None
    pub = install_hooks(port)
    time.sleep(0.5)  # let the window process bind its port first
    sys.argv = ["lerobot-teleoperate", *rest]
    try:
        _teleoperate_main()()
    except KeyboardInterrupt:
        pass
    finally:
        # Finalize the MP4 before generating reports; the twin stays open.
        for _ in range(3):
            pub.stop_recording()
            time.sleep(0.02)
        if plots and log_dir:
            logger.info("Generating tracking figures and the haircut coverage report...")
            _make_plots(log_dir, t_start, sim_config)
            try:
                from pathlib import Path
                _tidy_results(Path(log_dir).parent, log_dir)
            except Exception as e:
                logger.warning(f"Failed to tidy up previous result folders: {e}")
        if not calibrate and log_dir:
            from pathlib import Path
            try:
                Path(log_dir).rmdir()  # delete the empty folder only if nothing was produced this run (e.g. hardware never connected)
            except OSError:
                pass
        if proc is not None:
            logger.info("Teleoperation finished; the MuJoCo window stays open, close it manually")


if __name__ == "__main__":
    main()
