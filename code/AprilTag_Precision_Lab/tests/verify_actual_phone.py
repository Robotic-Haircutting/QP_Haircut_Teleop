"""Replay this phone's captured tag corners, not ideal projected corners."""
import asyncio,json,ssl,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiohttp import web,ClientSession,TCPConnector
from server import create_app,ROOT

async def main():
    frames=json.loads(Path(__file__).with_name('phone_rejections.json').read_text(encoding='utf-8'))
    app=create_app();runner=web.AppRunner(app);await runner.setup()
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(ROOT/'certs/cert.pem',ROOT/'certs/key.pem')
    site=web.TCPSite(runner,'127.0.0.1',0,ssl_context=ctx);await site.start()
    url=f'https://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    try:
      async with ClientSession(connector=TCPConnector(ssl=False)) as client:
        async def post(path,data):
            async with client.post(url+path,json=data) as r:
                text=await r.text();assert r.status==200,(r.status,text);return json.loads(text)
        async def action(name):return await post('/api/sim',dict(client_id='captured-phone',action=name))
        async def feed(frame,cal):
            return await post('/api/pose',dict(client_id='captured-phone',moving='phone',camera_id='captured-camera',tag_id=0,corners=frame['corners'],width=720,height=1280,tag_mm=80,calibration=cal))
        await action('start');sim=app['simulation'];q=sim.twin.q.copy();old=frames[0]['calibration']
        await feed(frames[0],old);s=await action('clutch');assert s['paused'] and s['resume_pending']
        for _ in range(3):await feed(frames[0],old);await asyncio.sleep(.04)
        assert sim.paused and np.max(np.abs(sim.twin.q-q))<1e-9
        fitted=await post('/api/estimate-focal',dict(frames=[x['corners'] for x in frames],calibration=old,width=720,height=1280,tag_mm=80))
        await feed(frames[0],fitted)
        assert not sim.paused and not sim.resume_pending and sim.input_mode=='phone'
        assert np.max(np.abs(sim.twin.q-q))<1e-9,'new intrinsics anchoring jumped'
        anchor=sim.twin.home_pose()[0]
        end=time.monotonic()+1
        while time.monotonic()<end:await feed(frames[1],fitted);await asyncio.sleep(.05)
        assert not sim.paused,sim.reason
        actual=sim.twin.home_pose()[0];expected=anchor+sim.delta()
        assert np.linalg.norm(actual-anchor)>.2,'actual phone input did not move simulated arm'
        assert np.linalg.norm(actual-expected)<2,(actual,expected)
        print('ACTUAL PHONE -> HTTPS -> MUJOCO PASS: bad fx612 waits without motion; fitted lens consumes explicit request without jump; actual next captured view moves endpoint',np.round(actual-anchor,3).tolist(),flush=True)
        await action('pause');saved=sim.twin.q.copy()
        await feed(frames[8],fitted);await action('clutch')
        assert not sim.paused and np.max(np.abs(sim.twin.q-saved))<1e-9
        print('ACTUAL PHONE CLUTCH PASS: recenter with captured phone pose, resume preserves joints',flush=True)
        await action('stop')
    finally:await runner.cleanup()

asyncio.run(main())
