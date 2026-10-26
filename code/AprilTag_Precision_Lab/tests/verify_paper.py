import asyncio,io,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiohttp import web,ClientSession
from PIL import Image
from server import create_app

async def main():
    folder=Path.home()/'Desktop/HHE-robot/AprilTag_Precision_Lab/results/plots/session_1790966176147888300_figures'
    assert len(list(folder.glob('*.png')))==8 and len(list(folder.glob('*.pdf')))==8
    with Image.open(folder/'tracking_xyz.png') as img:
        assert img.size==(2100,2160)
        assert abs(img.info['dpi'][0]-600)<1
    runner=web.AppRunner(create_app());await runner.setup()
    runner.app['simulation'].plot_path=folder/'tracking_xyz.png'
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    url=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    try:
        async with ClientSession() as client:
            async with client.get(url+'/api/sim/export?kind=zip') as response:
                assert response.status==200
                with zipfile.ZipFile(io.BytesIO(await response.read())) as archive:
                    assert len([n for n in archive.namelist() if n.endswith('.pdf')])==8
                    assert len([n for n in archive.namelist() if n.endswith('.png')])==8
                    assert 'index.html' in archive.namelist()
        print('PAPER TEMPLATE PASS: eight PNG/PDF pairs, 600 dpi, exact single-column size, ZIP export')
    finally:await runner.cleanup()

asyncio.run(main())
