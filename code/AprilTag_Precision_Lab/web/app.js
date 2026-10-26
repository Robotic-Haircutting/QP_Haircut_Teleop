'use strict';
const $=id=>document.getElementById(id), video=$('video'),overlay=$('overlay'),ctx=overlay.getContext('2d');
const frame=document.createElement('canvas'),fc=frame.getContext('2d',{willReadFrequently:true});
const detector=new ApriltagDetector();let boardLayout=null,layoutCapture=null;try{boardLayout=JSON.parse(localStorage.getItem('apriltag_board_layout_v1'))}catch(e){}let stream=null,calibration=null,session=null,calibrationSize=null,latest=null,record=null,tests=[],points={},busy=false;
let focalSamples=[],focalBusy=false,focalAttemptCount=0,focalCollected=0;
const clientId=sessionStorage.getItem('simulation_client_id')||crypto.randomUUID();sessionStorage.setItem('simulation_client_id',clientId);
const num=id=>Number($(id).value);
async function api(path,data){const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),15000);try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:controller.signal});if(!r.ok)throw Error(await r.text());return await r.json()}finally{clearTimeout(timeout)}}
function message(text){$('status').textContent=text}
function download(name,data,type='application/json'){const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function intrinsicKey(){if(!stream)return null;return 'apriltag_intrinsics_v2_'+stream.getVideoTracks()[0].getSettings().deviceId+'_'+video.videoWidth+'x'+video.videoHeight;}
function setCalibration(c){if(!['fx','fy','cx','cy','width','height'].every(k=>Number.isFinite(c[k])))throw Error('Invalid calibration file');calibration=c;const key=intrinsicKey();if(key&&c.width===video.videoWidth&&c.height===video.videoHeight)localStorage.setItem(key,JSON.stringify(c));for(const k of ['fx','fy','cx','cy'])$(k).value=c[k];$('quality').textContent=c.calibrated?`Calibrated: ${c.width}×${c.height} · RMS ${Number(c.rms_px||0).toFixed(3)} px; validate accuracy with measured physical motion.`:c.estimated?`Tag-assisted intrinsics estimate: focal length ${c.fx.toFixed(1)} · validation error ${c.validation_median_px.toFixed(2)} px; simulation only, calibrate with a checkerboard for real accuracy.`:'Provisional intrinsics: not checkerboard-calibrated; phone mode tries to estimate the focal length from different Tag views.'}

const phoneBrowser=/Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
document.documentElement.classList.toggle('phone-ui',phoneBrowser);
const detectionRole=phoneBrowser?'phone-detection-v1':'laptop-detection-v1';
if(localStorage.getItem('apriltag_camera_role')!==detectionRole){
  localStorage.setItem('apriltag_camera_id',phoneBrowser?'rear':'builtin');localStorage.setItem('apriltag_camera_role',detectionRole);
}
async function resolveDetectionCamera(selection){
  if(selection!=='builtin')return selection;
  let devices=await navigator.mediaDevices.enumerateDevices();
  if(!devices.some(d=>d.kind==='videoinput'&&d.label)){
    const permission=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    try{devices=await navigator.mediaDevices.enumerateDevices()}finally{permission.getTracks().forEach(t=>t.stop())}
  }
  const builtin=devices.find(d=>d.kind==='videoinput'&&/^Integrated Camera(?:\s|\(|$)/i.test(d.label));
  if(!builtin)throw Error('Built-in Integrated Camera not found; refresh the list and select a camera explicitly (an external camera is never chosen automatically).');
  return builtin.deviceId;
}

async function refreshCameras(){
  const devices=await navigator.mediaDevices.enumerateDevices(),select=$('camera'),selected=['default','builtin'].includes(select.value)?(localStorage.getItem('apriltag_camera_id')||'builtin'):select.value;
  for(const option of [...select.options])if(option.dataset.device)option.remove();
  let index=0;for(const device of devices.filter(d=>d.kind==='videoinput'&&d.deviceId)){
    const option=document.createElement('option');option.value=device.deviceId;option.dataset.device='true';
    option.textContent=device.label||`Camera ${++index}`;select.append(option);
  }
  if([...select.options].some(o=>o.value===selected))select.value=selected;
  else if(selected!=='default'){
    const option=document.createElement('option');option.value=selected;option.dataset.device='true';option.textContent='Selected camera is unavailable; choose again';select.append(option);select.value=selected;
  }
}
$('camera').onchange=()=>localStorage.setItem('apriltag_camera_id',$('camera').value);
$('changeCamera').onclick=async()=>{
  try{if(typeof simRunning!=='undefined'&&simRunning)await api('/api/sim',{client_id:clientId,action:'pause'})}
  catch(e){message('Cannot pause simulation: '+e.message);return}
  stream?.getTracks().forEach(t=>t.stop());location.reload();
};
$('refreshCameras').onclick=()=>refreshCameras().catch(e=>message(e.message));
refreshCameras().catch(()=>{});
function imageData(){frame.width=video.videoWidth;frame.height=video.videoHeight;fc.drawImage(video,0,0);return fc.getImageData(0,0,frame.width,frame.height)}
function draw(d,pose,captured=false){const ratio=devicePixelRatio||1,w=overlay.clientWidth,h=overlay.clientHeight;overlay.width=w*ratio;overlay.height=h*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const scale=($('previewFit').value==='cover'?Math.max:Math.min)(w/video.videoWidth,h/video.videoHeight),ox=(w-video.videoWidth*scale)/2,oy=(h-video.videoHeight*scale)/2;const xy=p=>[ox+p[0]*scale,oy+p[1]*scale];if(captured){ctx.fillStyle='#000';ctx.fillRect(0,0,w,h);ctx.drawImage(frame,ox,oy,video.videoWidth*scale,video.videoHeight*scale)}ctx.lineWidth=2;if(d){for(const tag of d.allTags||[d]){ctx.strokeStyle='#4eeaa1';ctx.beginPath();tag.corners.forEach((p,i)=>ctx[i?'lineTo':'moveTo'](...xy(p)));ctx.closePath();ctx.stroke();if(tag.id!==undefined){ctx.fillStyle='#4eeaa1';ctx.fillText('ID '+tag.id,...xy(tag.corners[0]))}}}if(pose){if($('showCube').checked&&pose.cube_pixels){
const vertices=pose.cube_pixels.map(xy),faces=[[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]];
faces.sort((a,b)=>b.reduce((v,i)=>v+pose.cube_depths[i],0)-a.reduce((v,i)=>v+pose.cube_depths[i],0));
for(const face of faces){ctx.beginPath();face.forEach((i,j)=>ctx[j?'lineTo':'moveTo'](...vertices[i]));ctx.closePath();ctx.fillStyle=face.every(i=>i>=4)?'rgba(90,170,255,.3)':'rgba(80,140,240,.15)';ctx.fill();ctx.strokeStyle='#76caff';ctx.lineWidth=2;ctx.stroke()}
}const axes=pose.axes_pixels;['#ff5252','#53df76','#528bff'].forEach((c,i)=>{ctx.strokeStyle=c;ctx.beginPath();ctx.moveTo(...xy(axes[0]));ctx.lineTo(...xy(axes[i+1]));ctx.stroke();ctx.fillStyle=c;ctx.fillText(['X','Y','Z'][i],...xy(axes[i+1]))})}}
let poseRequestBusy=false,previewPose=null,cameraLoopRunning=false;
function alignedPreviewPose(d,boardMode){
 if(!d||!previewPose||previewPose.mode!==boardMode)return null;
 const a=previewPose.corners,b=d.corners;
 return a.every((p,i)=>Math.hypot(p[0]-b[i][0],p[1]-b[i][1])<=3)?previewPose.pose:null;
}
async function submitPose(d,im,started,boardMode,boardTags,capturedAt){
 poseRequestBusy=true;
 try{const p=await api('/api/pose',{client_id:clientId,moving:$('movingMode').value,camera_id:video.srcObject.getVideoTracks()[0].getSettings().deviceId,tracking_mode:boardMode?'board':'single',board:boardMode?currentBoardGeometry():undefined,detections:boardMode?boardTags.map(t=>({id:t.id,corners:t.corners})):undefined,tag_id:boardMode?0:num('tagId'),corners:d.corners,tag_mm:boardMode?num('boardTagSize'):num('tagSize'),calibration,width:im.width,height:im.height});latest={...p,capture_ms:capturedAt,time_ms:Date.now(),processing_ms:performance.now()-started,corners:d.corners,decision_margin:d.decision_margin,hamming:d.hamming,frame_width:im.width,frame_height:im.height,tag_mm:boardMode?num('boardTagSize'):num('tagSize')};if(record)record.samples.push(latest);previewPose={pose:p,corners:d.corners,mode:boardMode};$('poseLiveStatus').textContent='● Live tracking · '+new Date().toLocaleTimeString()+(p.ambiguous?' · Ambiguous planar pose · Temporal selection':'');$('position').textContent=p.camera_t_mm.map((x,i)=>['X','Y','Z'][i]+': '+x.toFixed(2)).join('\n');$('rotation').textContent=p.rpy_deg.map((x,i)=>['Roll','Pitch','Yaw'][i]+': '+x.toFixed(2)).join('\n');$('error').textContent=`${p.reprojection_px.toFixed(3)} px\n${p.ambiguous?'Ambiguous pose · Temporal selection; ambiguity remains':(p.pose_selection==='temporal_continuity'?'Temporal pose selection':'Minimum reprojection solution')}\nCandidates: ${p.candidate_errors_px.map(x=>x.toFixed(3)).join(' / ')}`;message(`${boardMode?'Board ID '+p.board_ids.join(' / ')+(p.board_ids.length===1?' (one Tag visible)':'')+(p.rejected_ids?.length?' · Rejected ID '+p.rejected_ids.join(' / '):''):'Tag '+num('tagId')} · ${im.width}×${im.height}${p.ambiguous?' · Pose ambiguity':''}`)}
 catch(e){latest=null;previewPose=null;message(e.message);$('poseLiveStatus').textContent='Pose unavailable: '+e.message;for(const id of ['position','rotation','error'])$(id).textContent='—'}
 finally{poseRequestBusy=false}
}
async function loop(){
 if(cameraLoopRunning)return;cameraLoopRunning=true;
 try{while(stream){
 const started=performance.now(),capturedAt=Date.now();let d=null;
 try{
 const im=imageData(),gray=new Uint8Array(im.width*im.height);
 for(let i=0,j=0;i<gray.length;i++,j+=4)gray[i]=(77*im.data[j]+150*im.data[j+1]+29*im.data[j+2])>>8;
 const boardMode=$('trackingMode').value==='board',detections=detector.detect(gray,im.width,im.height).map(x=>({...x,id:Number(x.id??x.tag_id),corners:x.corners.map(p=>Array.isArray(p)?p:[p.x,p.y])})),boardTags=detections.filter(x=>x.id>=0&&x.id<=3);
 d=boardMode?boardTags[0]:detections.find(x=>x.id===num('tagId'));
 if(d&&boardMode)d={...d,allTags:boardTags};
 if(layoutCapture&&boardMode&&boardTags.length>=4&&performance.now()-layoutCapture.last>180){layoutCapture.frames.push(boardTags.map(t=>({id:t.id,corners:t.corners,decision_margin:t.decision_margin})));layoutCapture.last=performance.now();$('boardLayoutStatus').textContent='Samples '+layoutCapture.frames.length+' / 16: keep all four Tags visible and slowly tilt the phone'}
 // Display this captured image and its corners together, before any network request.
 draw(d,alignedPreviewPose(d,boardMode),true);
 if(d&&calibration&&!poseRequestBusy){if(record)record.attempts++;collectFocalView(d.corners);void submitPose(d,im,started,boardMode,boardTags,capturedAt)}
 else if(!d){latest=null;previewPose=null;$('poseLiveStatus').textContent='No Tags detected · Pose unavailable';message('No target tag');for(const id of ['position','rotation','error'])$(id).textContent='—'}
 }catch(e){previewPose=null;draw(null,null,true);message(e.message)}
 const elapsed=performance.now()-started;$('speed').textContent=`${(1000/Math.max(elapsed+5,40)).toFixed(1)} frames/s\nLocal detection ${elapsed.toFixed(1)} ms`;
 await new Promise(r=>setTimeout(r,Math.max(5,40-elapsed)));
 }}finally{cameraLoopRunning=false}
}
$('open').onclick=async()=>{try{if(stream){message('Refresh the page before changing the resolution or lens');return}const [width,height]=$('resolution').value.split(',').map(Number);const selection=await resolveDetectionCamera($('camera').value),constraints={width:{ideal:width},height:{ideal:height},aspectRatio:{ideal:width/height},frameRate:{ideal:30}};if(selection==='rear'||selection==='front')constraints.facingMode={ideal:selection==='rear'?'environment':'user'};else if(selection!=='default')constraints.deviceId={exact:selection};stream=await navigator.mediaDevices.getUserMedia({audio:false,video:constraints});const captureTrack=stream.getVideoTracks()[0],caps=captureTrack.getCapabilities?.()||{};if(caps.zoom){try{await captureTrack.applyConstraints({advanced:[{zoom:caps.zoom.min}]})}catch(e){}}video.srcObject=stream;await video.play();await detector.init();detector.setDetectorOptionsFn(2,0,1,1,0,0,0);const settings=stream.getVideoTracks()[0].getSettings();$('cameraInfo').textContent='AprilTag detection camera: '+stream.getVideoTracks()[0].label;$('open').disabled=true;$('resolution').disabled=true;$('camera').disabled=true;await refreshCameras();$('fx').value=$('fy').value=Math.round(Math.max(video.videoWidth,video.videoHeight)*.85);$('cx').value=video.videoWidth/2;$('cy').value=video.videoHeight/2;message(`Camera ${settings.width}×${settings.height}`);if(!calibration){const saved=localStorage.getItem(intrinsicKey());if(saved){setCalibration(JSON.parse(saved));$('lensStatus').textContent='Restored intrinsics for the current lens and resolution';}else{setCalibration({width:video.videoWidth,height:video.videoHeight,fx:num('fx'),fy:num('fy'),cx:num('cx'),cy:num('cy'),distortion:[0,0,0,0,0],calibrated:false,auto_estimate:true});$('quality').textContent='Virtual cube demo: uses estimated intrinsics to show tracking; calibrate with a checkerboard before real accuracy tests.'}}loop()}catch(e){if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;message(e.message)}};
$('manual').onclick=()=>{if(!stream)return message('Open the camera first.');try{setCalibration({width:video.videoWidth,height:video.videoHeight,fx:num('fx'),fy:num('fy'),cx:num('cx'),cy:num('cy'),distortion:[0,0,0,0,0],calibrated:false})}catch(e){message(e.message)}};
$('loadCalibration').onchange=async e=>{try{setCalibration(JSON.parse(await e.target.files[0].text()))}catch(error){message(error.message)}};
$('saveCalibration').onclick=()=>calibration&&download('camera_calibration.json',JSON.stringify(calibration,null,2));
$('capture').onclick=async()=>{if(!stream||busy)return;busy=true;try{imageData();const data=await api('/api/board',{image:frame.toDataURL('image/jpeg',.9).split(',')[1],cols:num('cols'),rows:num('rows'),square_mm:num('square'),session});session=data.session;calibrationSize=data.size;$('boardStatus').textContent=`Samples: ${data.count} / 15 · ${data.size.join('×')}; vary checkerboard angle and position`}catch(e){$('boardStatus').textContent=e.message}finally{busy=false}};
$('clearBoard').onclick=()=>{session=null;calibrationSize=null;$('boardStatus').textContent='Samples:0 / 15'};
$('calibrate').onclick=async()=>{if(busy)return;busy=true;try{setCalibration(await api('/api/calibrate',{session}));$('boardStatus').textContent=`Calibration complete, ${calibration.views} views · Per-view errors: ${calibration.view_errors_px.map(x=>x.toFixed(2)).join(', ')} px`}catch(e){message(e.message)}finally{busy=false}};
async function collect(kind,seconds){if(record||!stream||!calibration)return message('Open the camera, set the intrinsics, and wait for the current recording to finish');const config=JSON.parse(JSON.stringify(calibration)),tagSize=num('tagSize');record={kind,samples:[],attempts:0,started:Date.now(),calibration:config,tag_mm:tagSize};for(const id of ['staticTest','startPoint','endPoint','manual','capture','calibrate','loadCalibration','tagSize','tagId'])$(id).disabled=true;$('report').textContent=`Recording ${kind} ${seconds} s: keep the phone and Tag still`;setTimeout(async()=>{const finished=record;record=null;for(const id of ['staticTest','startPoint','endPoint','manual','capture','calibrate','loadCalibration','tagSize','tagId'])$(id).disabled=false;try{finished.metrics=await api('/api/metrics',{samples:finished.samples});finished.detection_fraction=finished.samples.length/finished.attempts;tests.push(finished);if(kind!=='static')points[kind]={...finished.metrics,calibration:finished.calibration,tag_mm:finished.tag_mm};$('report').textContent=JSON.stringify({kind,valid_fraction:finished.detection_fraction,...finished.metrics},null,2)}catch(e){tests.push(finished);$('report').textContent=e.message}},seconds*1000)}
$('staticTest').onclick=()=>collect('static',10);$('startPoint').onclick=()=>collect('A',2);$('endPoint').onclick=()=>collect('B',2);
$('compare').onclick=()=>{if(!points.A||!points.B)return message('Sample the start and end points first');if(points.A.tag_mm!==points.B.tag_mm||JSON.stringify(points.A.calibration)!==JSON.stringify(points.B.calibration))return message('Start and end points used different calibrations or Tag sizes; sample again');const delta=points.B.mean_camera_t_mm.map((x,i)=>x-points.A.mean_camera_t_mm[i]),distance=Math.hypot(...delta);let trace=0;for(let i=0;i<3;i++)for(let j=0;j<3;j++)trace+=points.B.mean_camera_R[i][j]*points.A.mean_camera_R[i][j];const angle=Math.acos(Math.max(-1,Math.min(1,(trace-1)/2)))*180/Math.PI;const comparison={delta_mm:delta,measured_distance_mm:distance,known_distance_mm:num('knownDistance'),distance_error_mm:distance-num('knownDistance'),measured_angle_deg:angle,known_angle_deg:num('knownAngle'),angle_error_deg:angle-num('knownAngle')};tests.push({kind:'comparison',comparison});$('report').textContent=JSON.stringify(comparison,null,2)};
$('downloadJson').onclick=()=>download('apriltag_tests.json',JSON.stringify({created:new Date().toISOString(),tag_family:'tag36h11',calibration,tests},null,2));
$('downloadCsv').onclick=()=>{const rows=['test,time_ms,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg,reprojection_px,ambiguous,processing_ms'];for(const test of tests)for(const s of test.samples||[])rows.push([test.kind,s.time_ms,...s.camera_t_mm,...s.rpy_deg,s.reprojection_px,s.ambiguous,s.processing_ms].join(','));download('apriltag_raw_poses.csv',rows.join('\n'),'text/csv')};

function resizeCameraPreview(){const value=Number($('cameraPreviewSize').value);document.documentElement.style.setProperty('--camera-preview-height',value+'dvh');$('cameraPreviewValue').textContent=value+'%';localStorage.setItem('camera_preview_size_v4',String(value))};$('cameraPreviewSize').value=localStorage.getItem('camera_preview_size_v4')||(localStorage.getItem('camera_preview_size_v3')&&localStorage.getItem('camera_preview_size_v3')!=='30'?localStorage.getItem('camera_preview_size_v3'):'38');$('cameraPreviewSize').oninput=resizeCameraPreview;resizeCameraPreview();

$('movingMode').value=localStorage.getItem('apriltag_moving_mode')||(phoneBrowser?'phone':'tag');
function showMovingMode(){const phone=$('movingMode').value==='phone';localStorage.setItem('apriltag_moving_mode',$('movingMode').value);$('modeHint').textContent=phone?'Tag fixed; phone translation and rotation both drive the end effector. Translation scale is adjustable (default 10:3.5), orientation 1:1; Resume resets both the position and orientation reference. Pose-ambiguity warnings are kept, and control picks the temporally continuous candidate. Current solver: DLS.':'Computer camera fixed, handheld Tag translates in three axes; toward the computer = forward. Current solver: DLS.';if(phoneBrowser)$('cameraInfo').textContent='Phone mode uses the rear camera; resizing the preview does not change the detection resolution.'}
$('movingMode').onchange=showMovingMode;showMovingMode();
$('movingMode').addEventListener('change',()=>{$('axisAlignment').hidden=$('movingMode').value==='phone'});
$('axisAlignment').hidden=$('movingMode').value==='phone';

function collectFocalView(corners){
 if($('movingMode').value!=='phone'||!calibration?.auto_estimate||calibration.estimated||calibration.calibrated)return;
 if(corners.some(p=>p[0]<0||p[1]<0||p[0]>=video.videoWidth||p[1]>=video.videoHeight))return;
 if(focalSamples.some(f=>f.reduce((a,p,i)=>a+Math.hypot(p[0]-corners[i][0],p[1]-corners[i][1]),0)/4<8))return;
 focalCollected++;focalSamples.push(corners.map(p=>p.slice()));if(focalSamples.length>40)focalSamples.shift();
 $('lensStatus').textContent=`Intrinsics samples: ${focalSamples.length}/12; keep the Tag fixed and slowly change the phone angle and distance`;
 if(focalSamples.length>=12&&focalCollected>=focalAttemptCount+3)estimateLens();
}
async function estimateLens(){
 if(focalBusy)return;if(!stream||!calibration)return message('Open the camera first.');
 if(calibration.calibrated)return message('Existing calibration retained.');
 if(focalSamples.length<12){$('lensStatus').textContent='Keep the Tag fully visible and slowly change the phone angle and distance to collect at least 12 different views';return;}
 if(typeof simPaused!=='undefined'&&!simPaused&&!(typeof simTrackingHold!=='undefined'&&simTrackingHold)){$('lensStatus').textContent='Pause before estimating intrinsics.';return;}
 focalBusy=true;focalAttemptCount=focalCollected;$('lensStatus').textContent='Estimating focal length and validating held-out views…';
 const original=calibration,key=intrinsicKey();
 try{const fitted=await api('/api/estimate-focal',{frames:focalSamples.slice(),calibration:original,tag_mm:$('trackingMode').value==='board'?num('boardTagSize'):num('tagSize'),width:video.videoWidth,height:video.videoHeight});
   if(intrinsicKey()!==key||calibration!==original)return;
   setCalibration(fitted);$('lensStatus').textContent=`Estimate complete: focal length ${fitted.fx.toFixed(1)}; validation error ${fitted.validation_median_px.toFixed(2)} px; camera profile saved`;
 }catch(e){$('lensStatus').textContent=e.message+'; collect more varied views and retry.';}
 finally{focalBusy=false;}
}
$('estimateLens').onclick=estimateLens;

$('resolution').value=phoneBrowser?'1280,960':'1280,720';

function applyPreviewFit(){const fit=$('previewFit').value;video.style.objectFit=fit;localStorage.setItem('camera_preview_fit_v1',fit);draw(null,null)}
$('previewFit').value=localStorage.getItem('camera_preview_fit_v1')||(phoneBrowser?'cover':'contain');$('previewFit').onchange=applyPreviewFit;applyPreviewFit();

function showTrackingMode(){const board=$('trackingMode').value==='board';localStorage.setItem('apriltag_tracking_mode_v1',$('trackingMode').value);$('tagBoardSettings').hidden=!board;$('tagId').disabled=board;$('tagSize').disabled=board;$('trackingOrigin').textContent=board?'World origin: center of the four-Tag board; top row IDs 0 / 1, bottom row IDs 2 / 3, same orientation.':'World origin: Tag center; axes: X / Y / Z on the Tag.';focalSamples=[];focalCollected=0;focalAttemptCount=0}
$('trackingMode').value=localStorage.getItem('apriltag_tracking_mode_v1')||(phoneBrowser?'board':'single');$('trackingMode').onchange=showTrackingMode;showTrackingMode();
for(const id of ['boardTagSize','boardGapX','boardGapY'])$(id).oninput=()=>{clearBoardLayout();$('boardGeometryHint').textContent='Top row IDs 0 / 1, bottom row IDs 2 / 3; black-border center spacing: horizontal '+(num('boardTagSize')+num('boardGapX'))+' mm, vertical '+(num('boardTagSize')+num('boardGapY'))+' mm; world origin at the board center'};
$('boardTagSize').oninput();

function currentBoardGeometry(){return boardLayout&&Number(boardLayout.tag_mm)===num('boardTagSize')?boardLayout:{tag_mm:num('boardTagSize'),gap_x_mm:num('boardGapX'),gap_y_mm:num('boardGapY')}}
function clearBoardLayout(){boardLayout=null;localStorage.removeItem('apriltag_board_layout_v1');$('boardLayoutStatus').textContent='Using manual spacing. Calibrate Board Layout to measure actual spacing.'}
$('resetBoardLayout').onclick=clearBoardLayout;
if(boardLayout){$('boardTagSize').value=boardLayout.tag_mm;$('boardLayoutStatus').textContent='Saved board layout restored. Recalibrate after changing the board.'}
$('calibrateTagLayout').onclick=async()=>{
 const button=$('calibrateTagLayout');
 if(!stream||!calibration?.calibrated){message('Open the camera and complete camera calibration');return}
 if(simRunning&&!simPaused){message('Pause the simulation before calibrating the board layout');return}
 button.disabled=true;layoutCapture={frames:[],last:0};
 try{
  const deadline=performance.now()+20000;
  while(layoutCapture.frames.length<16&&performance.now()<deadline)await new Promise(r=>setTimeout(r,100));
  const frames=layoutCapture.frames;layoutCapture=null;
  $('boardLayoutStatus').textContent='Optimizing board layout…';
  const im=imageData(),result=await api('/api/tag-layout',{frames,tag_mm:num('boardTagSize'),calibration,width:im.width,height:im.height});
  boardLayout=result.geometry;localStorage.setItem('apriltag_board_layout_v1',JSON.stringify(boardLayout));
  $('boardLayoutStatus').textContent='Layout saved · '+result.frames+' frames · '+result.reprojection_px.toFixed(2)+' px; center distances '+Object.entries(result.center_distances_mm).map(([ids,mm])=>'ID '+ids+': '+mm.toFixed(1)+' mm').join(';');
  message('Calibration done; press Resume to set a new reference, then move the phone');
 }catch(e){$('boardLayoutStatus').textContent='Layout not saved: '+e.message;message(e.message)}finally{layoutCapture=null;button.disabled=false}
};
