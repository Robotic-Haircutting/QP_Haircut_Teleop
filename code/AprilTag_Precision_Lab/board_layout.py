"""Estimate a fixed coplanar board layout from calibrated views and known tag size."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from tag_board import unique_detections,candidates
from pose_core import camera_matrix,object_corners

def calibrate_layout(data):
    size=float(data['tag_mm'])/1000
    if not .005<=size<=.3:raise ValueError('Enter the real black-border side length')
    if not data['calibration'].get('calibrated'):raise ValueError('Calibrate the camera first')
    k,dist=camera_matrix(data['calibration'],data['width'],data['height'])
    images=[];layouts=[];square=object_corners(size)
    for frame in data['frames'][:24]:
        tags=unique_detections(frame)
        if [t['id'] for t in tags]!=[0,1,2,3]:continue
        image=np.asarray([t['corners'] for t in tags],float)
        undistorted=cv2.undistortPoints(image.reshape(-1,1,2),k,dist).reshape(4,4,2)
        h=cv2.getPerspectiveTransform(undistorted[0].astype(np.float32),square[:,:2].astype(np.float32))
        layout=cv2.perspectiveTransform(undistorted.reshape(-1,1,2),h).reshape(4,4,2)
        images.append(image);layouts.append(layout)
    if len(images)<12:raise ValueError('At least 12 frames with all of ID 0–3 are required; remove other Tags with the same IDs and keep all four black borders visible')
    layout=np.median(layouts,axis=0)
    def parameterize(points):
        params=[]
        for p in points[1:]:
            delta=p[1]-p[0];params.extend([*np.mean(p,axis=0),np.arctan2(delta[1],delta[0])])
        return params
    def objects(params):
        result=[square]
        for x,y,angle in np.asarray(params[:9]).reshape(3,3):
            c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s],[s,c]])
            result.append(np.column_stack([square[:,:2]@r.T+[x,y],np.zeros(4)]))
        return np.concatenate(result)
    initial=parameterize(layout);obj=objects(initial)
    for image in images:
        poses=candidates(obj,image.reshape(-1,2),k,dist)
        if not poses:raise ValueError('Initial layout estimate failed; retry from a slightly tilted view')
        initial.extend([*poses[0][1].ravel(),*poses[0][2].ravel()])
    def residual(params):
        obj=objects(params);parts=[]
        for n,image in enumerate(images):
            p=params[9+6*n:15+6*n]
            parts.append((cv2.projectPoints(obj,p[:3],p[3:],k,dist)[0].reshape(4,4,2)-image).ravel())
        return np.concatenate(parts)
    sparsity=lil_matrix((len(images)*32,len(initial)),dtype=int)
    for n in range(len(images)):
        sparsity[n*32:(n+1)*32,:9]=1;sparsity[n*32:(n+1)*32,9+6*n:15+6*n]=1
    fit=least_squares(residual,np.asarray(initial),jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=1,max_nfev=120,xtol=1e-7,ftol=1e-7)
    errors=np.sqrt(np.mean(residual(fit.x).reshape(len(images),16,2)**2,axis=(1,2))*2)
    if not fit.success or np.median(errors)>2 or np.max(errors)>3:raise ValueError('Board layout fit is unstable: keep the board flat and all black borders the same size, then resample with slightly different phone angles')
    obj=objects(fit.x).reshape(4,4,3);centers=np.mean(obj,axis=1)
    for i in range(4):
        for j in range(i):
            if np.linalg.norm(centers[i]-centers[j])<size*.9:raise ValueError('Two Tags overlap or were misdetected; layout not saved')
    obj-=np.mean(centers,axis=0)
    return dict(geometry=dict(tag_mm=size*1000,corners_mm={str(i):(p*1000).tolist() for i,p in enumerate(obj)}),frames=len(images),reprojection_px=float(np.median(errors)),center_distances_mm={f'{i}-{j}':float(np.linalg.norm(centers[i]-centers[j])*1000) for i,j in [(0,1),(2,3),(0,2),(1,3)]})
