@echo off
chcp 65001 >nul
rem ===== Haircut teleoperation, PND single iteration per control cycle (N_max = 1) =====
rem Same settings as run_haircut.bat (gain 30, damping 0.001, 90 deg/s), except alpha, N_max, fps, start pose [0,48,-44,0,10,0] (flange 5 cm lower than before, q5 = 10 deg) and joint 5 limit +-65 deg.
set ALPHA=0.2
set FPS=100
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" lerobot
if errorlevel 1 goto error
cd /d "%~dp0.."
set TAG=a%ALPHA%_fps%FPS%_j5wide_low5cm
if not exist "%~dp0results\%TAG%" mkdir "%~dp0results\%TAG%"
echo Haircut: SO-101 leader -^> PND single iteration (alpha=%ALPHA%, fps=%FPS%) -^> Piper. Keep a hand on the E-stop. Ctrl+C to quit.
python -m haircut_sim.teleop_sim --robot.type=haircut_piper_follower --robot.id=piper --robot.can_port=can0 --robot.control_mode=qp --robot.ik_solver=pnd --robot.speed_rate=50 --robot.position_scale=1.0 --robot.start_joints_deg="[0,48,-44,0,10,0]" --robot.joint_limits_deg="[[-150,150],[15,150],[-160,-15],[-100,100],[-65,65],[-100,100]]" --robot.pnd.max_joint_speed_deg_s=90 --robot.pnd.gain=30 --robot.pnd.damping=0.001 --robot.pnd.relaxation=%ALPHA% --robot.pnd.iterations=1 --robot.log_dir="%~dp0results\%TAG%" --teleop.type=haircut_so101_leader_ee --teleop.include_joints=false --teleop.output_gripper=false --teleop.port=COM6 --teleop.id=leader --teleop.home_joints_deg="[-32,-49.7,46.7,3.4,-106.5,0]" --fps=%FPS% --sim.record
pause
exit /b
:error
echo Could not activate the lerobot environment.
pause
