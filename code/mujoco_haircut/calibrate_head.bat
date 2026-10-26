@echo off
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" lerobot
if errorlevel 1 goto error
cd /d "%~dp0"
echo Touch the TOP of the head with the clipper tip. ENTER=record, U=undo, R=restart, Y=save.
python -m haircut_sim.teleop_sim --robot.type=haircut_piper_follower --robot.id=piper --robot.can_port=can0 --robot.control_mode=qp --robot.ik_solver=pnd --robot.speed_rate=50 --robot.position_scale=1.0 --robot.start_joints_deg="[0,53,-63,0,40,0]" --robot.pnd.max_joint_speed_deg_s=90 --robot.pnd.gain=30 --robot.pnd.damping=0.001 --teleop.type=haircut_so101_leader_ee --teleop.include_joints=false --teleop.output_gripper=false --teleop.port=COM6 --teleop.id=leader --teleop.home_joints_deg="[-32,-49.7,46.7,3.4,-106.5,0]" --fps=60 --sim.calibrate
pause
exit /b
:error
echo Could not activate the lerobot environment.
pause
