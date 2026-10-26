import asyncio,sys,ssl,time
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiohttp import web,ClientSession,TCPConnector
from server import create_app,ROOT
from tag_board import board_points
from phone_pose import CAMERA_TO_BASE
async def main():
 app=create_app();runner=web.AppRunner(app);await runner.setup();ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(ROOT/'certs/cert.pem',ROOT/'certs/key.pem')
 site=web.TCPSite(runner,'127.0.0.1',0,ssl_context=ctx);await site.start();url=f'https://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
 cal=dict(width=960,height=1280,fx=1160,fy=1160,cx=480,cy=640,calibrated=True);k=np.array([[1160.,0,480],[0,1160,640],[0,0,1]])
 geo=dict(tag_mm=50,gap_x_mm=100,gap_y_mm=100);r=Rotation.from_euler('xyz',[8,-6,2],degrees=True).as_matrix();initial=-r.T@np.array([0,0,.65])
 try:
  async with ClientSession(connector=TCPConnector(ssl=False)) as c:
   async def action(name):
    async with c.post(url+'/api/sim',json=dict(client_id='board-test',action=name)) as response:assert response.status==200,await response.text();return await response.json()
   async def feed(offset,ids):
    p=initial+r.T@np.array(offset)/1000;tv=-r@p;rv=cv2.Rodrigues(r)[0]
    tags=[dict(id=i,corners=cv2.projectPoints(board_points(i,geo),rv,tv,k,None)[0].reshape(4,2).tolist()) for i in ids]
    payload=dict(client_id='board-test',camera_id='board-camera',tracking_mode='board',moving='phone',tag_id=0,tag_mm=50,board=geo,detections=tags,width=960,height=1280,calibration=cal)
    async with c.post(url+'/api/pose',json=payload) as response:assert response.status==200,await response.text();return await response.json()
   await feed([0,0,0],[0,1,2,3]);await action('clutch');sim=app['simulation'];assert sim.scale==.35;home=np.array(sim.state()['position_mm']);reference=sim.ref.copy()
   for ids in ([1,2,3],[2],[0,3],[0,1,2,3]):
    for _ in range(8):await feed([0,0,0],ids);await asyncio.sleep(.035)
    assert np.linalg.norm(np.array(sim.state()['position_mm'])-home)<.1
    assert np.linalg.norm(sim.ref-reference)<1e-6,'tag subset changed reference'
   for _ in range(25):await feed([-20,0,0],[0,1,2,3]);await asyncio.sleep(.04)
   actual=np.array(sim.state()['position_mm'])-home;assert np.linalg.norm(actual-[0,7,0])<.3,actual
   await action('clutch');q=sim.twin.q.copy();await feed([100,40,20],[1,3]);await action('clutch')
   assert np.max(np.abs(sim.twin.q-q))<1e-8
   print('HTTPS + REAL MUJOCO BOARD PASS: partial occlusion keeps origin and joints; phone-left 20mm -> robot-left 7mm; clutch recenter has no jump',flush=True)
 finally:await runner.cleanup()
asyncio.run(main())
