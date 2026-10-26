# Installation

The experiments were run on Windows 10/11. The steps below assume Miniconda with an environment named `lerobot`; every `.bat` file in the repository looks for Python at `%USERPROFILE%\miniconda3\envs\lerobot\python.exe`. If conda is installed elsewhere, change that line in the `.bat` files.

## 1. Python environment

```bat
conda create -n lerobot python=3.12 -y
conda activate lerobot
cd <repository root>\code
pip install -r requirements.txt
pip install -e mujoco_haircut
```

`pip install -e mujoco_haircut` installs the `haircut_sim` package in editable mode; the phone experiment imports it as well (coverage, recording, paper figure templates).

## 2. FFmpeg (needed for recording)

Both experiments record the MuJoCo view and the real camera view as H.264 MP4, which requires `ffmpeg` on PATH or installed in the conda environment:

```bat
conda install -n lerobot -c conda-forge ffmpeg -y
```

## 3. USB-CAN (Piper follower arm)

- Piper is connected through a candleLight / gs_usb compatible USB-CAN adapter; the interface name in the code is `can0`.
- On Windows, install the WinUSB driver for the adapter with [Zadig](https://zadig.akeo.ie/); `gs_usb`, `pyusb` and `libusb-package` are already in requirements.txt.
- Only one program can open the USB-CAN adapter at a time; close the previous program before switching experiments.

## 4. SO-101 leader arm (Experiment 1)

- The leader's serial port defaults to `COM6`; check it in Device Manager and change `--teleop.port=COM6` in the `.bat` files.
- Calibrate the leader with LeRobot first, using the id `leader`:

  ```bat
  lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=leader
  ```

  The calibration file is saved as `so101_leader\leader.json` under `%USERPROFILE%\.cache\huggingface\lerobot\calibration\teleoperators\` (depending on the LeRobot version it may also be `so_leader\` or `so100_leader\`); the code searches these directories in turn.

## 5. HTTPS certificate for the phone web page (Experiment 2)

Phone browsers only allow HTTPS pages to open the camera, so the service runs with a self-signed certificate. The certificate holds a local private key; **never commit it** (`.gitignore` already excludes `certs/`). Generate it before first use:

```bat
cd AprilTag_Precision_Lab
python tools\make_cert.py
```

This creates `AprilTag_Precision_Lab/certs/cert.pem` and `key.pem`. The first time the phone opens the page it warns that the certificate is not trusted; choose to proceed.

Windows Firewall must allow LAN access to TCP 8443. Run this once from an administrator PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File AprilTag_Precision_Lab\allow_phone_access.ps1
```

## 6. Self-tests

```bat
cd mujoco_haircut
python tests\test_standalone.py
python tests\test_pnd_ik.py
python tests\test_scene.py
python tests\test_calibrate.py
run_haircut.bat mock

cd ..\AprilTag_Precision_Lab
python -B tests\verify_geometry.py
python -B tests\verify_pnd_qp.py
python -B tests\verify_head_guard.py
```

A few cases in `verify_paper.py` and `verify_simulation.py` read result files that existed on the original development machine; they fail in a fresh environment without those files, which is expected.
