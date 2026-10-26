import sys,subprocess,ssl,urllib.request,time,socket
from pathlib import Path
root=Path(__file__).parent;url='https://127.0.0.1:8443/piper/'
try:
    with urllib.request.urlopen(url,context=ssl._create_unverified_context(),timeout=2) as response:assert b'robotConnect' in response.read()
except Exception:
    subprocess.Popen([sys.executable,'-B',str(root/'server.py'),'--port','8443'],cwd=root,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    for _ in range(40):
        try:
            with urllib.request.urlopen(url,context=ssl._create_unverified_context(),timeout=1) as response:assert b'robotConnect' in response.read()
            break
        except Exception:time.sleep(.25)
    else:raise SystemExit('Service not ready; check port 8443 and results/service.log before retrying.')
# No browser is opened on this computer: control from the phone only, one page at a time.
def lan_ip():
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:s.connect(('8.8.8.8',80));return s.getsockname()[0]
    except OSError:return '<PC LAN IP>'
    finally:s.close()
print(f'Service ready. Open on the PHONE only: https://{lan_ip()}:8443/piper/')
print('Do not open the control page on this computer at the same time.')
