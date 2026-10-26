"""Dedicated launcher for today's independent AprilTag lab (not Client_RHCR)."""
import json,ssl,subprocess,sys,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
URL='https://127.0.0.1:8443/'
print('NEW AprilTag control project - port 8443',flush=True)
print('Project: '+str(ROOT),flush=True)
print('Previous Client_RHCR project has its own BAT and port 8000.',flush=True)
try:
    with urllib.request.urlopen(URL+'api/sim/state',context=ssl._create_unverified_context(),timeout=2) as response:
        state=json.load(response)
    if not {'solver','moving','scale','paused'}.issubset(state):raise ValueError('Unexpected service on port 8443')
except (OSError,ValueError):
    print('Starting this project only. Keep this console open.',flush=True)
    raise SystemExit(subprocess.call([sys.executable,'-B',str(ROOT/'server.py'),'--port','8443'],cwd=ROOT))
else:
    print('This project is already running. Open '+URL,flush=True)
    print('Phone: use the computer LAN address with port 8443.',flush=True)
