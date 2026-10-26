"""End-to-end synthetic image corners -> HTTPS -> IK -> actual viewer frame."""
import asyncio,json,ssl,sys,time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiohttp import web,ClientSession,TCPConnector
from server import create_app,ROOT
from pose_core import object_corners

async def main():
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(ROOT/'certs/cert.pem',ROOT/'certs/key.pem')
    app=create_app();runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0,ssl_context=ctx);await site.start()
    url=f'https://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    calibration=dict(width=640,height=480,fx=800,fy=800,cx=320,cy=240,calibrated=True)
    k=np.array([[800.,0,320],[0,800,240],[0,0,1]])
    rotation=np.array([.15,.1,.02])
    control_mode='tag'
    try:
      async with ClientSession(connector=TCPConnector(ssl=False)) as client:
        async def action(name,**kwargs):
            async with client.post(url+'/api/sim',json=dict(client_id='test',action=name,**kwargs)) as r:
                body=await r.text();assert r.status==200,(name,body);return json.loads(body)
        async def feed(p):
            corners=cv2.projectPoints(object_corners(.08),rotation,np.array(p,float)/1000,k,None)[0].reshape(4,2)
            async with client.post(url+'/api/pose',json=dict(client_id='test',moving=control_mode,tag_id=0,corners=corners.tolist(),tag_mm=80,calibration=calibration,width=640,height=480)) as r:
                assert r.status==200,await r.text()
        async def hold(p,seconds=.8):
            end=time.monotonic()+seconds
            while time.monotonic()<end:await feed(p);await asyncio.sleep(.04)
        def pos():return np.array(app['simulation'].state()['position_mm'])
        await action('sensitivity',step=.25)
        await feed([0,0,550]);await action('clutch')
        assert not app['simulation'].paused and app['simulation'].anchored
        await action('clutch');assert app['simulation'].paused
        print('UNIFIED CLUTCH PASS: pre-start detection, one-click start/anchor and pause',flush=True)
        # Wait for genuine MuJoCo rendering; no renderer mocks.
        frame=None
        for _ in range(100):
            async with client.get(url+'/api/sim/frame.jpg') as r:
                if r.status==200:frame=await r.read();break
            await asyncio.sleep(.2)
        assert frame,'MuJoCo viewer did not render'
        image=cv2.imdecode(np.frombuffer(frame,np.uint8),cv2.IMREAD_COLOR)
        assert image.shape==(384,1536,3),image.shape
        (ROOT/'results/verified_mujoco.jpg').write_bytes(frame)
        await feed([0,0,550]);home=pos();await action('resume')
        # Move 20mm toward camera, then operator left, then upward.
        cases=[([0,0,530],[5,0,0]),([20,0,530],[5,5,0]),([20,-20,530],[5,5,5])]
        for p,expected in cases:
            await hold(p);actual=pos()-home
            assert np.linalg.norm(actual-expected)<1.5,(actual,expected)
            print('DIRECTION/SCALE PASS',actual.round(3).tolist(),flush=True)
        await action('clutch');stopped=pos();q=app['simulation'].twin.q.copy()
        await hold([-80,50,620]);assert np.linalg.norm(pos()-stopped)<1e-9
        await action('clutch');await hold([-80,50,620],.4)
        assert np.max(np.abs(app['simulation'].twin.q-q))<1e-9,'resume jumped'
        await hold([-80,50,600]);assert np.linalg.norm((pos()-stopped)-[5,0,0])<1.5
        print('CLUTCH PASS: pause, large recenter, resume unchanged joints, subsequent movement',flush=True)
        await asyncio.sleep(.9);assert not app['simulation'].paused and app['simulation'].tracking_hold
        stopped=pos();await hold([-40,20,560]);assert np.linalg.norm(pos()-stopped)<1e-9
        assert not app['simulation'].paused and not app['simulation'].tracking_hold
        await hold([-40,20,540]);assert np.linalg.norm(pos()-stopped)>1
        await action('pause');saved=pos();await hold([-20,30,520]);assert app['simulation'].paused and np.linalg.norm(pos()-saved)<1e-9
        print('LOSS RECOVERY PASS: holds without clutch pause, reacquires without jump and follows; manual pause never auto-resumes',flush=True)
        await feed([-40,20,560]);await action('align',step='origin')
        for step,p in [('forward',[-40,20,500]),('left',[20,20,560]),('up',[-40,-40,560])]:
            await feed(p);await action('align',step=step)
        assert np.max(np.abs(app['simulation'].axis-np.array([[0,0,-1],[1,0,0],[0,-1,0]])))<1e-6
        await feed([-40,20,560]);await action('resume')
        # A changed size/intrinsics configuration must break the old anchor.
        calibration['fx']=810;saved=pos();await feed([-40,20,560]);assert not app['simulation'].paused and np.linalg.norm(pos()-saved)<1e-9
        print('ALIGNMENT/CONFIG PASS: configuration reanchors without automatic pause; genuine rendered JPEG saved',flush=True)
        control_mode='phone';calibration['fx']=800
        await feed([0,0,550]);anchor=pos();await action('resume')
        await hold([0,0,530]);assert np.linalg.norm(pos()-anchor-[5,0,0])<1.5
        await hold([20,0,530]);assert np.linalg.norm(pos()-anchor-[5,5,0])<1.5
        await hold([20,20,530]);assert np.linalg.norm(pos()-anchor-[5,5,5])<1.5
        from scipy.spatial.transform import Rotation
        camera_r=Rotation.from_rotvec(rotation).as_matrix().T
        camera_p=-camera_r@np.array([20.,20.,530.])
        async def feed_camera(p,r):
            rv=cv2.Rodrigues(r.T)[0]
            corners=cv2.projectPoints(object_corners(.08),rv,-r.T@p/1000,k,None)[0].reshape(4,2)
            async with client.post(url+'/api/pose',json=dict(client_id='test',moving='phone',tag_id=0,corners=corners.tolist(),tag_mm=80,calibration=calibration,width=640,height=480)) as response:
                assert response.status==200,await response.text()
        # 2026-10-07: phone steers the clipper tip, so a pure rotation must keep the TIP fixed (the flange swings around it).
        tip=lambda:app['simulation']._tip(*app['simulation'].twin.home_pose())
        before_r=app['simulation'].twin.home_pose()[1];before_p=tip()
        for angle in np.linspace(0,5,21):
            changed_r=camera_r@Rotation.from_euler('z',angle,degrees=True).as_matrix()
            await feed_camera(camera_p,changed_r);await asyncio.sleep(.04)
        for _ in range(25):await feed_camera(camera_p,changed_r);await asyncio.sleep(.04)
        actual_r=app['simulation'].twin.home_pose()[1]
        assert not app['simulation'].paused,app['simulation'].reason
        rotated=Rotation.from_matrix(actual_r@before_r.T).magnitude()*180/np.pi
        assert abs(rotated-5)<.5,rotated
        assert np.linalg.norm(tip()-before_p)<1.5,'rotation changed the clipper-tip target'
        await action('pause')
        recentered_r=camera_r@Rotation.from_euler('z',25,degrees=True).as_matrix()
        await feed_camera(camera_p,recentered_r);q=app['simulation'].twin.q.copy()
        await action('resume')
        for _ in range(10):await feed_camera(camera_p,recentered_r);await asyncio.sleep(.04)
        assert np.max(np.abs(app['simulation'].twin.q-q))<1e-9
        await action('pause')
        from phone_pose import select_phone_pose
        good=Rotation.from_euler('y',15,degrees=True).as_matrix()
        bad=Rotation.from_euler('y',-15,degrees=True).as_matrix()
        cp=np.array([0.,0.,-500.])
        ambiguous=dict(candidate_poses=[dict(R=bad.T.tolist(),t_mm=(-bad.T@cp).tolist()),dict(R=good.T.tolist(),t_mm=(-good.T@cp).tolist())],candidate_errors_px=[.05,.06])
        chosen_p,chosen_r,index=select_phone_pose(ambiguous,(cp,good))
        assert index==1 and np.linalg.norm(chosen_r-good)<1e-9
        print('PHONE 6D PASS: xyz scale 10:2.5, pure camera roll -> arm roll 5deg, no position drift, pose clutch no jump, continuous ambiguity selection',flush=True)
        import csv,io
        from tracking_csv import HEADER
        async with client.get(url+'/api/sim/export?kind=csv') as response:
            assert response.status==200
            rows=list(csv.DictReader(io.StringIO(await response.text())))
            assert list(rows[0])==HEADER
            assert float(rows[0]['qp_iters'])!=float(rows[0]['qp_iters'])
            assert abs(float(rows[0]['r_x'])-float(rows[0]['rc_x']))<1e-12
            assert all(float(row['solve_ms'])>=0 for row in rows)
        async with client.get(url+'/api/sim/export?kind=jsonl') as response:
            assert response.status==200
            events=[json.loads(line) for line in (await response.text()).splitlines()]
            assert any(e['event']=='resume_anchor' for e in events)
            assert any(e['event']=='pause' for e in events)
            assert all('t' in e for e in events)
        saved=app['simulation'].twin.q.copy()
        viewer=app['simulation'].viewer
        viewer.terminate();await asyncio.to_thread(viewer.wait,5)
        await asyncio.sleep(.15)
        assert not app['simulation'].state()['running'],'closed viewer leaves Start disabled'
        assert app['simulation'].owner is None
        for _ in range(100):
            if not app['simulation'].plotting:break
            await asyncio.sleep(.2)
        assert app['simulation'].plot_path,app['simulation'].plot_error
        async with client.get(url+'/api/sim/export?kind=png') as response:
            assert response.status==200
            png=await response.read();assert png.startswith(b'\x89PNG')
        import subprocess
        from video_mp4 import executable,verify_seek
        fixture=ROOT/'results/test_recording_input.webm'
        import shutil
        existing=next((Path.home()/'Desktop/HHE-robot/AprilTag_Precision_Lab/results/videos').glob('*.webm'))
        shutil.copyfile(existing,fixture)
        async with client.post(url+'/api/recordings?ext=webm',data=fixture.read_bytes()) as response:
            assert response.status==200
            saved_video=await response.json();path=Path(saved_video['path'])
            assert path.parent==ROOT/'results/videos'
            assert path.suffix=='.mp4'
            info=await asyncio.to_thread(verify_seek,path);assert float(info['format']['duration'])>.5
            async with client.get(url+saved_video['url'],headers={'Range':'bytes=0-100'}) as download:
                assert download.status==206 and len(await download.read())==101
            path.unlink();path.with_suffix('.json').unlink()
        fixture.unlink()
        async with client.post(url+'/api/recordings?ext=exe',data=b'bad') as response:assert response.status==400
        print('DATA/VIDEO PASS: legacy CSV, JSONL, PNG chart, real WebM -> H264 MP4, seek decode and HTTP range',flush=True)
        await action('start')
        assert app['simulation'].paused
        assert np.max(np.abs(app['simulation'].twin.q-saved))<1e-9
        print('RECOVERY PASS: viewer exit clears running/owner; restart preserves joints and remains paused',flush=True)
        # Waiting only follows an explicit click, with cancel and expiry.
        await feed_camera(camera_p,recentered_r)
        sim=app['simulation'];before=sim.twin.q.copy()
        waiting=sim.action(dict(action='clutch',client_id='phone-tab'))
        assert waiting['paused'] and waiting['resume_pending']
        cached=sim.pose_cache['test'];data=dict(cached[2],client_id='phone-tab')
        sim.update(cached[1],data)
        assert not sim.paused and not sim.resume_pending and sim.owner=='phone-tab'
        assert np.max(np.abs(sim.twin.q-before))<1e-9
        sim.action(dict(action='clutch',client_id='test'));assert sim.paused
        bad=dict(cached[1],reprojection_px=4)
        sim.update(bad,data)
        waiting=sim.action(dict(action='clutch',client_id='phone-tab'))
        assert waiting['paused'] and waiting['resume_pending']
        sim.update(bad,data);assert sim.paused and sim.resume_pending
        sim.update(cached[1],data);assert not sim.paused and not sim.resume_pending
        sim.action(dict(action='clutch',client_id='phone-tab'));assert sim.paused
        sim.update(bad,data);sim.action(dict(action='clutch',client_id='phone-tab'))
        sim.action(dict(action='clutch',client_id='phone-tab'))
        assert sim.paused and not sim.resume_pending,'second click did not cancel waiting'
        sim.update(cached[1],data);assert sim.paused,'cancelled wait restarted'
        sim.update(bad,data);sim.action(dict(action='clutch',client_id='phone-tab'))
        sim.pending_until=float('inf');await asyncio.sleep(.1)
        assert sim.paused and sim.resume_pending
        sim.update(cached[1],data);assert not sim.paused and not sim.resume_pending,'waiting did not automatically start with valid input'
        print('WAITING CLUTCH PASS: explicit request, bad pose holds still, valid pose anchors without jump, cancel stays paused; pending request waits until valid input',flush=True)
        sim.action(dict(action='stop',client_id='phone-tab'));assert not sim.state()['running']
    finally:await runner.cleanup()

asyncio.run(main())
