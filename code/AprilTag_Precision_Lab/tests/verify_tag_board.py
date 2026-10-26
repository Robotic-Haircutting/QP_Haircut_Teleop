import sys,itertools,json
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tag_board import solve_board,board_points
from pose_core import solve_pose
geometry=dict(tag_mm=50,gap_x_mm=100,gap_y_mm=100)
cal=dict(width=960,height=1280,fx=1160,fy=1160,cx=480,cy=640,calibrated=True)
k=np.array([[1160.,0,480],[0,1160,640],[0,0,1]])
rv=np.array([.16,-.11,.03]);tv=np.array([.015,.02,.65]);r=cv2.Rodrigues(rv)[0];expected=-r.T@tv*1000
items=[dict(id=i,corners=cv2.projectPoints(board_points(i,geometry),rv,tv,k,None)[0].reshape(4,2).tolist()) for i in range(4)]
for n in range(1,5):
 for subset in itertools.combinations(items,n):
  pose=solve_board(subset,geometry,cal,960,1280)
  assert np.linalg.norm(np.array(pose['camera_t_mm'])-expected)<1e-4,(n,pose)
print('PARTIAL VISIBILITY PASS: all 15 visible subsets use the same board origin, including individual tags')
bad=json.loads(json.dumps(items));bad[1]['corners']=(np.array(bad[1]['corners'])+[22,5]).tolist()
pose=solve_board(bad,geometry,cal,960,1280);assert pose['rejected_ids']==[1] and len(pose['board_ids'])==3,pose
assert np.linalg.norm(np.array(pose['camera_t_mm'])-expected)<1e-3
assert solve_board([items[0],items[0]],geometry,cal,960,1280)['board_ids']==[0]
duplicate=dict(id=0,corners=(np.array(items[0]['corners'])+[90,0]).tolist())
try:solve_board([items[0],duplicate],geometry,cal,960,1280);raise AssertionError('spatial duplicate accepted')
except ValueError:pass
print('OUTLIER PASS: bad tag rejected; same quad duplicates merged; distinct duplicate IDs rejected')
rng=np.random.default_rng(45);board=[];single=[]
for _ in range(80):
 noisy=[dict(id=i['id'],corners=(np.array(i['corners'])+rng.normal(0,.35,(4,2))).tolist()) for i in items]
 board.append(solve_board(noisy,geometry,cal,960,1280)['camera_t_mm'])
 p=solve_pose(noisy[0]['corners'],50,cal,960,1280)
 single.append(np.array(p['camera_t_mm'])+np.mean(board_points(0,geometry),axis=0)*1000)
bs=np.linalg.norm(np.std(board,axis=0));ss=np.linalg.norm(np.std(single,axis=0));assert bs<ss*.6,(bs,ss)
print('NOISE SIMULATION PASS: identical .35px corner noise, board position std',round(bs,3),'mm; single',round(ss,3),'mm (synthetic, not phone accuracy)')
Path(__file__).with_name('board_pose_fixtures.json').write_text(json.dumps(dict(geometry=geometry,calibration=cal,items=items)),encoding='utf-8')
