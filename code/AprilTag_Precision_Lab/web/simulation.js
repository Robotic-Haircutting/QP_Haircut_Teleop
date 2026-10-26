'use strict';
let simRunning=false,simPolling=false,simPaused=true,simBusy=false,simActionError='',simTrackingHold=false;
function renderSim(s){
  simRunning=s.running;simTrackingHold=!!s.tracking_hold;
  if(document.activeElement?.id!=='sensitivity')$('sensitivity').value=Math.round((s.scale??.35)*100);$('sensitivityValue').textContent='10:'+Number($('sensitivity').value)/10;

  $('dataLogStatus').textContent=s.data_log?(s.data_recording?'● Recording data: ':'Data saved: ')+s.data_log+(s.plot_file?'; plots: '+s.plot_file:'; plots are generated after the simulation stops'):'Data is recorded automatically once the simulation starts; plots are generated when it stops.';
  for(const [id,available] of [['exportFigures',s.plot_file],['exportTracking',s.tracking_log],['exportEvents',s.data_log],['exportPlot',s.plot_file]]){$(id).style.pointerEvents=available?'auto':'none';$(id).style.opacity=available?'1':'.4';$(id).setAttribute('aria-disabled',available?'false':'true');}

  if(s.coverage_file)$('dataLogStatus').textContent+='; haircut log for this run: '+s.coverage_file;
  if(s.plotting)$('dataLogStatus').textContent+='; generating the full set of PNG/PDF…'; if(s.plot_error)$('dataLogStatus').textContent+='; plotting failed: '+s.plot_error; simPaused=s.paused;
  $('simStart').disabled=s.running;
  $('simPause').disabled=simBusy;
  $('simPause').textContent=!s.running?'Start and set reference':s.returning?'Stop arm homing':s.resume_pending?'Cancel wait / Pause':s.paused?(s.return_blocked?'Resume (move outward first)':s.needs_return?'Home arm and resume':'Resume (homing done)'):'Pause / Home';
  $('simPause').style.background=s.returning?'#8b412d':s.resume_pending?'#7d6429':s.paused?'#226b50':'#8b412d';
  $('simPause').title=!s.running?'Click once a Tag is detected to start the simulation and set the current phone position and orientation as the control reference':s.returning?'Click to stop arm homing immediately':s.paused?(s.needs_return?'Joints near their limits: first return slowly to the initial working pose, then set a new phone reference':'Re-center the phone, then click; continues from where it stopped'):'Press Pause, then move the Tag back to a comfortable position';
  $('simStop').disabled=!s.running;
  const other=s.running&&s.controller_id&&s.controller_id!==clientId;
  const near=(s.joint_margins_deg||[]).map((m,i)=>({margin:m,index:i})).filter(x=>x.margin<2);
  $('robotLimitStatus').textContent=!s.running?'Joint margin: shown after the simulation starts':near.length?'Near joint limits: '+near.map(x=>'J'+(x.index+1)+'='+s.joints_deg[x.index].toFixed(1)+'° (margin '+x.margin.toFixed(1)+'°)').join(' / ')+(s.paused?'; press Home arm and resume to restore margin':'; press Pause, then Home arm and resume to restore margin'):'Joint margins OK';
  $('poseReference').textContent='Control reference: '+(!s.running?'not set; keep the Tag visible and press Start and set reference':s.returning?'The arm slowly returns to the initial working pose, then sets a new reference from the current phone pose':other?'Controlled by another page; press Pause, then press Resume on this page to take over':s.resume_pending?'Waiting for a valid pose; keep the Tag fully visible and the reference will be set automatically':s.paused?'Paused; move the phone back to a comfortable position and press Resume to reset the position + orientation reference':s.anchored?'set; the end effector follows from where it stopped at this reference':'not set');
  $('simStatus').textContent=(!s.running?'Not running · ':s.returning?'Arm homing · ':s.resume_pending?'Waiting for pose · ':s.paused?'Paused · ':s.tracking_hold?'Waiting for pose · ':s.motion_limit?'Motion limited · ':'Following · ')+(s.returning?'returning to the initial working pose at up to 15°/s; the phone is not driving it; homing can be stopped at any time':s.reason)+(s.running&&s.paused&&!s.visible&&!s.returning&&!s.resume_pending?'; bring the Tag back into view, then press Resume':'')+(s.position_mm?'; end effector mm: '+s.position_mm.map(v=>v.toFixed(1)).join(' / '):'');
  if(s.head_safety)$('simStatus').textContent+='; head guard: closest '+s.head_safety.distance_mm.toFixed(1)+' mm (min 10 mm)'+(s.head_safety.blocked?', approach blocked':'');
  if(simActionError)$('simStatus').textContent+='; action failed: '+simActionError;
  if(s.paused&&s.tracking_quality){const q=s.tracking_quality;$('simStatus').textContent+='; latest detection '+q.reprojection_px.toFixed(2)+' px'+(q.age_seconds>.6?' (stale)':'')+(q.calibrated?'':'; current intrinsics are uncalibrated');}
  $('simAxes').textContent='Sampled: '+(s.alignment_samples.join(' / ')||'none')+'\nCamera frame → arm base:\n'+s.axis_map.map(row=>row.map(v=>v.toFixed(3)).join('  ')).join('\n');
}
async function simAction(action,step){
  if(simBusy)return;simBusy=true;simActionError='';$('simPause').disabled=true;
  $('simStatus').textContent='Request received: '+(action==='clutch'?(simRunning&&!simPaused?'pausing':'setting position/orientation reference'):'processing')+'…';
  try{const state=await api('/api/sim',{client_id:clientId,action,step,observer:location.pathname.startsWith('/piper'),solver:location.pathname.startsWith('/piper')?'PND':'DLS',prepare_robot:action==='start'&&typeof robotConnected!=='undefined'&&robotConnected,bench_clear:$('benchClear')?.checked===true});simBusy=false;renderSim(state)}
  catch(e){simActionError=e.message;$('simStatus').textContent='Action failed: '+e.message}
  finally{simBusy=false;$('simPause').disabled=false}
}
$('simStart').onclick=()=>simAction('start');
$('simPause').onclick=()=>simAction('clutch');
$('simStop').onclick=()=>simAction('stop');
$('simDefault').onclick=()=>simAction('default');
document.querySelectorAll('[data-align]').forEach(b=>b.onclick=()=>simAction('align',b.dataset.align));
let framePending=false,frameUrl=null;
function drawPhoneSimulation(){
  if(!phoneBrowser)return;
  const source=$('simImage'),canvas=$('mobileSimulation');
  if(!source.naturalWidth)return;
  const context=canvas.getContext('2d'),panelWidth=source.naturalWidth/3,panelHeight=source.naturalHeight;
  context.drawImage(source,0,0,panelWidth,panelHeight,0,0,512,384);
  for(const [panel,y,label] of [[2,172,'Side view / Clearance'],[1,272,'Tool close-up']]){
    context.drawImage(source,panel*panelWidth,0,panelWidth,panelHeight,376,y,128,96);
    context.fillStyle='#111';context.fillRect(376,y,128,18);
    context.fillStyle='#fff';context.font='12px system-ui';context.fillText(label,382,y+13);
    context.strokeStyle='#7f9bad';context.strokeRect(376,y,128,96);
  }
}
$('simImage').onload=drawPhoneSimulation;
async function refreshSim(){
  if(simPolling)return;simPolling=true;
  try{const r=await fetch('/api/sim/state',{cache:'no-store'});if(!r.ok)throw Error('Cannot read simulation state');renderSim(await r.json());}
  catch(e){$('simStatus').textContent=e.message;simRunning=false}
  finally{simPolling=false}
}
async function refreshFrame(){
  if(!simRunning||framePending)return;framePending=true;
  try{const r=await fetch('/api/sim/frame.jpg',{cache:'no-store'});if(!r.ok)return;
    const next=URL.createObjectURL(await r.blob());$('simImage').src=next;
    if(frameUrl)URL.revokeObjectURL(frameUrl);frameUrl=next;
  }catch(e){}finally{framePending=false}
}
setInterval(refreshSim,300);setInterval(refreshFrame,150);refreshSim();
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName)&&simRunning){e.preventDefault();simAction('pause')}});
// Hidden pages may stop camera updates; the server holds position and resumes on valid input.

$('simFullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.querySelector('.sim-view').requestFullscreen()}catch(e){$('simStatus').textContent='Browser cannot enter fullscreen: '+e.message}};

$('sensitivity').oninput=()=>{$('sensitivityValue').textContent='10:'+Number($('sensitivity').value)/10};$('sensitivity').onchange=()=>simAction('sensitivity',Number($('sensitivity').value)/100);
