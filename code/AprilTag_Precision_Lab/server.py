import argparse,asyncio,base64,json,socket,ssl,uuid,time,io,zipfile
from pathlib import Path
import cv2
import numpy as np
from aiohttp import web,ClientSession,ClientTimeout
from sim_bridge import Simulation
from video_mp4 import convert_mp4
from pose_core import solve_pose,metrics
from tag_board import solve_board
from board_layout import calibrate_layout
from pose_continuity import continuous_pose
from robot_bridge import RobotExperiment
from observer_camera import ObserverCamera
from tag_intrinsics import estimate_focal
ROOT=Path(__file__).resolve().parent

async def index(request):return web.FileResponse(ROOT/('web/piper.html' if request.path.startswith('/piper') else 'web/index.html'),headers={'Cache-Control':'no-store'})
async def pose(request):
    data=await request.json()
    try:
        if data.get('tracking_mode')=='board':
            data['tag_mm']=float(data['board']['tag_mm'])
            result=await asyncio.to_thread(solve_board,data['detections'],data['board'],data['calibration'],data['width'],data['height'])
        else:result=await asyncio.to_thread(solve_pose,data['corners'],data['tag_mm'],data['calibration'],data['width'],data['height'])
        if data.get('moving')=='phone' and data.get('client_id'):
            history=request.app['pose_history'];client=data['client_id']
            config=json.dumps([data.get(k) for k in ('camera_id','tracking_mode','board','tag_id','tag_mm','calibration','width','height')],sort_keys=True)
            last=history.get(client);now=time.monotonic()
            previous=last[2] if last and last[0]==config and now-last[1]<2 else None
            result,selected=continuous_pose(result,data,previous)
            if result['reprojection_px']<=3:
                history.pop(client,None);history[client]=(config,now,selected)
                while len(history)>10:history.pop(next(iter(history)))
        await asyncio.gather(asyncio.to_thread(request.app['robot'].feed,result,data),asyncio.to_thread(request.app['simulation'].update,result,data))
        return web.json_response(result)
    except (KeyError,ValueError,TypeError,cv2.error) as e:raise web.HTTPBadRequest(text=str(e))
async def analyze(request):
    data=await request.json()
    try:return web.json_response(await asyncio.to_thread(metrics,data['samples']))
    except (KeyError,ValueError,TypeError) as e:raise web.HTTPBadRequest(text=str(e))

async def tag_layout(request):
    data=await request.json()
    if request.app['simulation'].state()['running'] and not request.app['simulation'].state()['paused']:raise web.HTTPBadRequest(text='Pause the simulation before calibrating the board layout')
    try:return web.json_response(await asyncio.to_thread(calibrate_layout,data))
    except (KeyError,ValueError,TypeError,cv2.error) as e:raise web.HTTPBadRequest(text=str(e))

def find_board(data):
    raw=base64.b64decode(data['image']);img=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
    if img is None:raise ValueError('Invalid image')
    cols,rows=int(data['cols']),int(data['rows']);square=float(data['square_mm'])
    if not (3<=cols<=20 and 3<=rows<=20 and 1<=square<=100):raise ValueError('Invalid chessboard geometry')
    found,corners=cv2.findChessboardCornersSB(img,(cols,rows),cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:raise ValueError('Not all inner checkerboard corners were found; adjust the angle or lighting and sample again')
    obj=np.zeros((cols*rows,3),np.float32);obj[:,:2]=np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square
    return obj,corners,(img.shape[1],img.shape[0]),(cols,rows,square)
async def board(request):
    data=await request.json()
    try:obj,corners,size,geometry=await asyncio.to_thread(find_board,data)
    except (ValueError,KeyError,TypeError,cv2.error) as e:raise web.HTTPBadRequest(text=str(e))
    sessions=request.app['boards'];token=data.get('session')
    if token not in sessions:
        if len(sessions)>=10:raise web.HTTPBadRequest(text='Too many calibration sessions; restart the test service')
        token=uuid.uuid4().hex;sessions[token]=[]
    frames=sessions[token]
    if frames and (frames[0][2]!=size or frames[0][3]!=geometry):raise web.HTTPBadRequest(text='Resolution or checkerboard settings changed; clear the samples and start again')
    if frames and min(float(np.mean(np.linalg.norm(corners-x[1],axis=2))) for x in frames) < 8:
        raise web.HTTPBadRequest(text='Checkerboard view is too similar to an existing sample; change the tilt, distance or position')
    if len(frames)>=40:raise web.HTTPBadRequest(text='At most 40 samples')
    frames.append((obj,corners,size,geometry))
    return web.json_response({'session':token,'count':len(frames),'size':size})
def calibrate_frames(frames):
    rms,k,d,rv,tv=cv2.calibrateCamera([x[0] for x in frames],[x[1] for x in frames],frames[0][2],None,None)
    if not np.isfinite(k).all() or not np.isfinite(d).all() or not (frames[0][2][0]*.1 < k[0,0] < frames[0][2][0]*10):
        raise ValueError('Calibration result is implausible; recollect checkerboard images at different tilts and positions')
    errors=[]
    for f,r,t in zip(frames,rv,tv):
        projected=cv2.projectPoints(f[0],r,t,k,d)[0];errors.append(float(np.sqrt(np.mean(np.sum((projected-f[1])**2,axis=2)))))
    return {'width':frames[0][2][0],'height':frames[0][2][1],'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2]),'distortion':d.ravel().tolist(),'calibrated':True,'rms_px':float(rms),'view_errors_px':errors,'views':len(frames)}
async def calibrate(request):
    data=await request.json();frames=request.app['boards'].get(data.get('session'),[])
    if len(frames)<15:raise web.HTTPBadRequest(text='At least 15 samples are required, covering different tilts, distances and image positions')
    try:return web.json_response(await asyncio.to_thread(calibrate_frames,list(frames)))
    except (ValueError,cv2.error) as e:raise web.HTTPBadRequest(text=str(e))
async def sim_action(request):
    try:
        data=await request.json();sim=request.app['simulation'];robot=request.app['robot']
        if data.get('action') in ('resume','clutch') and robot.backend:raise ValueError('Use Arm Control when connected; MuJoCo mirrors measured joints.')
        if data.get('action')=='stop' and robot.backend:robot.pause('Simulation display stopped; arm held.')
        if data.get('prepare_robot') is True and data.get('bench_clear') is not True:raise ValueError('Before starting, confirm that no tool is mounted and nobody is near the arm')
        if data.get('action') in ('start','clutch') and sim.twin is None:
            sim.solver_kind='PND' if data.get('solver')=='PND' else 'DLS'
            port=request.transport.get_extra_info('sockname')[1]
            sim.observer_url=f'https://127.0.0.1:{port}/api/observer/frame.jpg' if data.get('observer') is True else None
        state=await asyncio.to_thread(sim.action,data)
        if data.get('action')=='start' and data.get('prepare_robot') is True:
            await asyncio.to_thread(robot.action,dict(action='prepare',client_id=data['client_id'],bench_clear=data['bench_clear']))
        return web.json_response(state)
    except (ValueError,OSError) as e:raise web.HTTPBadRequest(text=str(e))

async def sim_state(request):return web.json_response(await asyncio.to_thread(request.app['simulation'].state))

async def estimate_tag_intrinsics(request):
    try:return web.json_response(await asyncio.to_thread(estimate_focal,await request.json()))
    except (ValueError,KeyError,TypeError,cv2.error) as e:raise web.HTTPBadRequest(text=str(e))

async def save_recording(request):
    ext=request.query.get('ext','webm')
    if ext not in ('webm','mp4'):raise web.HTTPBadRequest(text='Unsupported video format')
    directory=ROOT/('results/piper_videos' if request.query.get('entry')=='piper' else 'results/videos');originals=directory/'originals';originals.mkdir(parents=True,exist_ok=True)
    basename=f'interface_{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    path=originals/f'{basename}.{ext}';output=directory/f'{basename}.mp4'
    size=0;prefix=bytearray();limit=2*1024**3;valid=False
    try:
        with path.open('xb') as uploaded_file:
            async for chunk in request.content.iter_chunked(256*1024):
                size+=len(chunk)
                if size>limit:raise web.HTTPRequestEntityTooLarge(max_size=limit,actual_size=size)
                if len(prefix)<12:prefix.extend(chunk[:12-len(prefix)])
                await asyncio.to_thread(uploaded_file.write,chunk)
        valid=bytes(prefix[:4])==b'\x1a\x45\xdf\xa3' if ext=='webm' else bytes(prefix[4:8])==b'ftyp'
        if not valid:raise web.HTTPBadRequest(text='Invalid video container')
        async with request.app['video_conversion']:
            try:info=await asyncio.to_thread(convert_mp4,path,output)
            except (OSError,ValueError) as error:raise web.HTTPInternalServerError(text=str(error))
        sim=request.app['simulation']
        with sim.lock:
            data_file=str(sim.session_path) if sim.session_path else None
            if request.query.get('entry')!='piper':sim.record('video_saved',path=str(output),bytes=output.stat().st_size)
        if request.query.get('entry')=='piper':
            robot=request.app['robot']
            with robot.lock:
                data_file=str(robot.log.path) if robot.log else None
                if robot.log and not robot.log.closed:robot.log.event('video_saved',path=str(output),bytes=output.stat().st_size)
        output.with_suffix('.json').write_text(json.dumps(dict(video=str(output),original_video=str(path),data_file=data_file,started_ms=request.query.get('started_ms'),saved_ms=round(time.time()*1000),bytes=output.stat().st_size,mp4_info=info),ensure_ascii=False,indent=2),encoding='utf-8')
        prefix='piper-recordings' if request.query.get('entry')=='piper' else 'recordings'
        return web.json_response(dict(path=str(output),bytes=output.stat().st_size,data_file=data_file,url=f'/api/{prefix}/{output.name}'))
    except BaseException:
        if not valid:path.unlink(missing_ok=True)
        raise

async def download_recording(request):
    name=request.match_info['name']
    if Path(name).name!=name or not name.startswith('interface_') or not name.endswith('.mp4'):raise web.HTTPNotFound()
    path=ROOT/('results/piper_videos' if '/piper-recordings/' in request.path else 'results/videos')/name
    if not path.is_file():raise web.HTTPNotFound()
    return web.FileResponse(path,headers={'Content-Disposition':f'attachment; filename="{name}"'})

async def sim_export(request):
    sim=request.app['simulation']
    kind=request.query.get('kind','csv')
    if kind not in ('csv','jsonl','png','zip'):raise web.HTTPBadRequest(text='Invalid export format')
    with sim.lock:
        path=sim.plot_path if kind in ('png','zip') else sim.tracking_path if kind=='csv' else sim.session_path
        if not path or not path.exists():raise web.HTTPNotFound(text='Start the simulation and record data first')
        if sim.tracking:sim.tracking.file.flush()
        if sim.session_log:sim.session_log.flush()
        # Snapshot ensures a consistent file even while the simulation continues logging.
        if kind=='zip':
            archive=io.BytesIO()
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
                for item in path.parent.iterdir():
                    if item.is_file() and item.suffix in ('.png','.pdf','.json','.html'):bundle.write(item,item.name)
            body=archive.getvalue();filename=path.parent.name+'.zip'
        else:body=path.read_bytes();filename=path.name
    return web.Response(body=body,content_type={'csv':'text/csv','jsonl':'application/x-ndjson','png':'image/png','zip':'application/zip'}[kind],headers={'Content-Disposition':f'attachment; filename="{filename}"','Cache-Control':'no-store'})

async def sim_frame(request):
    sim=request.app['simulation']
    if not sim.twin:raise web.HTTPServiceUnavailable(text='Simulation not started')
    try:
        async with request.app['http'].get(f'http://127.0.0.1:{sim.stream_port}/frame.jpg') as response:
            if response.status!=200:raise web.HTTPServiceUnavailable(text='Waiting for the MuJoCo view')
            return web.Response(body=await response.read(),content_type='image/jpeg',headers={'Cache-Control':'no-store'})
    except (OSError,asyncio.TimeoutError):raise web.HTTPServiceUnavailable(text='Waiting for the MuJoCo view')

async def resources(app):
    app['simulation']=Simulation()
    def mirror(q):
        sim=app['simulation']
        robot=app.get('robot')
        # Live hair coverage in the MuJoCo window during real-arm phone following (not homing/paused)
        sim.robot_following=bool(robot is not None and not robot.paused and not robot.preparing)
        with sim.lock:
            if sim.twin and sim.paused:sim.twin.show_joints(q)
    app['robot']=RobotExperiment(mirror=mirror)
    # Head placement used to be session-local. User 2026-10-07: reuse the last Align Head (the head has not moved)
    # so it need not be repeated after a restart. Only a clipper-tip record is reused while the clipper is mounted;
    # pressing Align Head again replaces it.
    try:
        from head_alignment import TOOL_MOUNTED
        saved=json.loads((Path(__file__).resolve().parent/'results/calibration/head_alignment.json').read_text(encoding='utf-8'))
        if saved.get('reference')==('tool_tip' if TOOL_MOUNTED else 'flange_center'):
            app['robot'].head_alignment=saved;app['simulation'].head_alignment=saved
            print(f"Reusing saved head alignment {saved['revision']} ({saved['reference']})",flush=True)
    except (OSError,ValueError,KeyError):
        pass
    app['observer']=ObserverCamera()
    app['video_conversion']=asyncio.Semaphore(1)
    app['http']=ClientSession(timeout=ClientTimeout(total=2))
    yield
    await app['http'].close()
    await asyncio.to_thread(app['observer'].stop)
    await asyncio.to_thread(app['robot'].close)
    await asyncio.to_thread(app['simulation'].close)

async def robot_action(request):
    data=await request.json()
    if data.get('action') in ('clutch','prepare','return_initial') and not request.app['simulation'].paused:raise web.HTTPBadRequest(text='Pause simulation control first; MuJoCo mirrors the real arm.')
    try:
        if data.get('action')=='align_head':
            result=await asyncio.to_thread(request.app['robot'].align_head,data,request.app['simulation'])
            return web.json_response(result)
        state=await asyncio.to_thread(request.app['robot'].action,data)
        sim=request.app['simulation']
        sim.mirror_only=state['connected']
        if data.get('action')=='connect' and state['connected']:
            def transfer_mirror():
                with sim.lock:
                    if sim.paused:sim.owner=data['client_id']
            await asyncio.to_thread(transfer_mirror)
        return web.json_response(state)
    except (KeyError,ValueError,RuntimeError,ConnectionError,OSError) as e:raise web.HTTPBadRequest(text=str(e))

async def robot_state(request):return web.json_response(await asyncio.to_thread(request.app['robot'].state))

async def observer_action(request):
    data=await request.json();camera=request.app['observer']
    try:
        if data.get('action')=='start':await asyncio.to_thread(camera.start)
        elif data.get('action')=='stop':await asyncio.to_thread(camera.stop)
        else:raise ValueError('Unknown camera action')
        return web.json_response(camera.state())
    except (OSError,ValueError) as e:raise web.HTTPBadRequest(text=str(e))

async def observer_frame(request):
    camera=request.app['observer']
    with camera.lock:
        if not camera.frame or time.monotonic()-camera.last>1:raise web.HTTPServiceUnavailable(text=camera.error or 'Waiting for external camera frames')
        body=camera.frame
    return web.Response(body=body,content_type='image/jpeg',headers={'Cache-Control':'no-store'})

async def robot_export(request):
    robot=request.app['robot'];kind=request.query.get('kind','csv')
    with robot.lock:
        if not robot.log:raise web.HTTPNotFound(text='Start an experiment and record data first')
        if not robot.log.closed:robot.log.file.flush()
        root=robot.log.root
        if kind=='csv':body=robot.log.path.read_bytes();name=root.name+'.csv';mime='text/csv'
        elif kind=='overview':
            if not robot.plots:raise web.HTTPNotFound(text='The overview figure is generated after the experiment stops')
            body=robot.plots.read_bytes();name=root.name+'_overview.png';mime='image/png'
        elif kind=='zip':
            if robot.plotting:raise web.HTTPBadRequest(text='Plotting not finished yet')
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as bundle:
                for p in root.rglob('*'):
                    if p.is_file():bundle.write(p,p.relative_to(root))
            body=buffer.getvalue();name=root.name+'.zip';mime='application/zip'
        else:raise web.HTTPBadRequest(text='Unknown export type')
    return web.Response(body=body,content_type=mime,headers={'Content-Disposition':f'attachment; filename="{name}"','Cache-Control':'no-store'})

def create_app():
    app=web.Application(client_max_size=12*1024*1024);app['boards']={};app['pose_history']={}
    app.cleanup_ctx.append(resources)
    app.router.add_post('/api/robot',robot_action);app.router.add_get('/api/robot/state',robot_state);app.router.add_get('/api/robot/export',robot_export)
    app.router.add_post('/api/observer',observer_action);app.router.add_get('/api/observer/frame.jpg',observer_frame)
    app.router.add_post('/api/sim',sim_action);app.router.add_get('/api/sim/state',sim_state);app.router.add_get('/api/sim/frame.jpg',sim_frame)
    app.router.add_post('/api/estimate-focal',estimate_tag_intrinsics)
    app.router.add_post('/api/tag-layout',tag_layout)
    app.router.add_get('/api/sim/export',sim_export)
    app.router.add_post('/api/recordings',save_recording)
    app.router.add_get('/api/recordings/{name}',download_recording)
    app.router.add_get('/api/piper-recordings/{name}',download_recording)
    app.router.add_get('/',index);app.router.add_get('/piper/',index);app.router.add_get('/piper',index);app.router.add_post('/api/pose',pose);app.router.add_post('/api/metrics',analyze)
    app.router.add_post('/api/board',board);app.router.add_post('/api/calibrate',calibrate);app.router.add_static('/static/',ROOT/'web')
    return app
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8443);args=ap.parse_args()
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(ROOT/'certs/cert.pem',ROOT/'certs/key.pem')
    ips=sorted({i[4][0] for i in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET)})
    for ip in ips:print(f'Phone URL: https://{ip}:{args.port}/',flush=True)
    web.run_app(create_app(),host='0.0.0.0',port=args.port,ssl_context=context)
