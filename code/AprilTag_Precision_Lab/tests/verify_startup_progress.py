import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_bridge import Simulation
sim=Simulation()
def await_save(path):
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        if path.exists():
            with np.load(path) as data:
                assert not data['covered'].any() and len(data['segments'])==0,'new viewer started with old cuts'
            return
        if sim.viewer.poll() is not None:raise RuntimeError('viewer exited')
        time.sleep(.15)
    raise RuntimeError('viewer did not persist initial empty scene')
try:
    sim.start('startup-test');first=sim.coverage_path;assert first.stem==sim.session_path.stem
    assert '--coverage-path' in sim.viewer.args and str(first) in sim.viewer.args
    await_save(first);assert sim.paused
    sim.stop();assert first.exists();saved=first.read_bytes()
    sim.start('startup-test');second=sim.coverage_path;assert first!=second
    await_save(second);assert first.read_bytes()==saved,'new startup changed previous coverage'
    print('REAL MUJOCO INITIALIZATION PASS: two new starts show empty coverage/trails, use separate session files, and preserve previous data',flush=True)
finally:sim.close()
