import asyncio,ssl,subprocess,sys
from pathlib import Path
from server import create_app,ROOT
async def main():
 context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(ROOT/'certs/cert.pem',ROOT/'certs/key.pem')
 from aiohttp import web
 runner=web.AppRunner(create_app());await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0,ssl_context=context);await site.start()
 url=f'https://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
 try:
  cmd=[r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless','--disable-gpu','--no-first-run','--ignore-certificate-errors','--disable-background-networking','--no-default-browser-check',f'--user-data-dir={ROOT/"tests/browser-profile"}',f'--screenshot={ROOT/"tests/phone_preview.png"}','--window-size=430,1050','--virtual-time-budget=2500',url]
  p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
  try:out,err=await asyncio.wait_for(asyncio.to_thread(p.communicate),20)
  except asyncio.TimeoutError:p.kill();out,err=await asyncio.to_thread(p.communicate)
  print('BROWSER_PREVIEW', (ROOT/'tests/phone_preview.png').exists())
 finally:await runner.cleanup()
asyncio.run(main())
