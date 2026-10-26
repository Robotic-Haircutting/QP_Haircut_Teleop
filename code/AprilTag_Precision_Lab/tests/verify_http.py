import asyncio,ssl,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiohttp import web,ClientSession,TCPConnector
from server import create_app,ROOT
async def main():
 context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(ROOT/'certs/cert.pem',ROOT/'certs/key.pem')
 runner=web.AppRunner(create_app());await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0,ssl_context=context);await site.start()
 url=f'https://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
 try:
  async with ClientSession(connector=TCPConnector(ssl=False)) as client:
   for path in ['/','/static/app.js','/static/vendor/apriltag/apriltag_wasm.wasm']:
    async with client.get(url+path) as r:assert r.status==200,(path,r.status)
   det=json.loads((ROOT/'tests/wasm_detection.json').read_text())
   data={'corners':[[p['x'],p['y']] for p in det['corners']],'tag_mm':80,'width':640,'height':480,'calibration':{'width':640,'height':480,'fx':1000,'fy':1000,'cx':320,'cy':240,'calibrated':True}}
   async with client.post(url+'/api/pose',json=data) as r:assert r.status==200;pose=await r.json();assert abs(pose['t_mm'][2]-500)<2
   async with client.post(url+'/api/metrics',json={'samples':[pose]*10}) as r:assert r.status==200;result=await r.json();assert result['position_p95_radius_mm']<1e-6
   data['height']=640
   async with client.post(url+'/api/pose',json=data) as r:assert r.status==400
   print('HTTPS PASS: page, JS, WASM, actual detected corners -> pose, metrics, mismatch rejection')
 finally:await runner.cleanup()
asyncio.run(main())
