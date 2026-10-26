"""Run in-process mock teleoperation while forbidding external robot code."""
import importlib.abc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

class BlockExternalRobot(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('lerobot_robot_piper', 'piper_teleop')):
            raise AssertionError('External robot project import forbidden: ' + fullname)

sys.meta_path.insert(0, BlockExternalRobot())
from haircut_sim.control import PiperFollower, PiperFollowerConfig, SO101LeaderEE, SO101LeaderEEConfig
from haircut_sim.pnd_ik import PNDIK
assert Path(sys.modules[PNDIK.__module__].__file__).resolve().is_relative_to(ROOT)
assert Path(sys.modules[PiperFollower.__module__].__file__).resolve().is_relative_to(ROOT)
leader = SO101LeaderEE(SO101LeaderEEConfig(port='none', mock=True, output_gripper=False))
leader.connect()
assert set(leader.get_action()) == {'ee.x','ee.y','ee.z','ee.wx','ee.wy','ee.wz'}
leader.disconnect()
from haircut_sim.teleop_sim import main
sys.argv = ['teleop_sim', '--robot.type=haircut_piper_follower', '--robot.id=test',
    '--robot.mock=true', '--robot.control_mode=qp', '--robot.ik_solver=pnd',
    '--robot.start_joints_deg=[0,50,-50,0,30,0]', '--robot.return_on_disconnect=false',
    '--teleop.type=haircut_so101_leader_ee', '--teleop.port=none', '--teleop.id=test',
    '--teleop.mock=true', '--teleop.output_gripper=false', '--teleop.include_joints=false',
    '--fps=60', '--teleop_time_s=3', '--sim.no_viewer', '--sim.no_plots']
main()
print('PASS: self-contained imports and pose-only mock control')
