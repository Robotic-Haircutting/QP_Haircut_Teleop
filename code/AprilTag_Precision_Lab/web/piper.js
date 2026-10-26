'use strict';
let robotBusy=false,observerBusy=false,observerOn=false,observerURL=null,robotActionError='',robotConnected=false,robotState=null;
let startupActive=false,startupReady=false,startupToken=0;
// The operator checks the clear workspace and E-stop before pressing Start; no extra startup dialog.
let benchConfirmed=false;
function benchOk(){return $('robotBackend').value==='mock'||benchConfirmed}
function confirmBench(what){benchConfirmed=true;return true;}

const WORKFLOW_TEXT={idle:'Step 0/5 · Not started',connected:'Step 1/5 · Connected — press Start',homing:'Step 1/5 · Returning to the start pose…',
 head_calibration:'Step 2/5 · Head calibration: Resume, guide the clipper tip 10 mm above the crown center, then Align Head',
 haircut:'Step 3/5 · Haircut: Resume to follow, Pause to clutch. Finish & Home when done',
 resting:'Step 4/5 · Finishing: returning to the initial pose…',rested:'Step 5/5 · At initial pose, motors holding. Session saved',
 rest_failed:'⚠ Return not completed — motors kept enabled. See status below'};
const startupSteps={phone:'Off',camera:'Off',simulation:'Off',home:'Not ready'};
function startupProgress(){ $('startupStatus').textContent='Phone Camera: '+startupSteps.phone+' · Real Camera: '+startupSteps.camera+' · MuJoCo: '+startupSteps.simulation+' · Arm: '+startupSteps.home; }
function renderRobot(s){
 $('alignHead').disabled=robotBusy||s.preparing;
 robotState=s;robotConnected=s.connected;
 if(s.head_alignment&&!$('headAlignmentStatus').dataset.revision){
  $('headAlignmentStatus').dataset.revision=s.head_alignment.revision;
  $('headAlignmentStatus').textContent='Retained head alignment · Crown / world mm: '+s.head_alignment.crown_world_m.map(v=>(v*1000).toFixed(1)).join(' / ')+' · Translation only; not validated for physical cutting.';
 }

 $('workflowStatus').textContent=WORKFLOW_TEXT[s.workflow]||'';
 $('robotRest').disabled=robotBusy||!s.connected||!s.enabled||(s.preparing&&s.workflow!=='homing');$('robotPrepare').disabled=robotBusy||!s.connected||s.preparing;
 $('robotStatus').textContent=(robotActionError?'Action failed: '+robotActionError+'; ':'')+(s.backend==='hardware'?'Piper':'Mock')+' · '+s.reason;
 if(s.resume_pending)$('robotStatus').textContent+=' · Stable frames: '+Math.min(s.stable_frames,3)+'/3';
 if(!s.paused&&s.input_filter&&!s.input_filter.accepted)$('robotStatus').textContent+=' · Outlier ignored; following last valid target';
 $('robotClutch').textContent=s.preparing||s.resume_pending?'Pause':!startupReady||s.fault||!s.connected||!s.enabled||!s.initial_ready?'Start':s.paused?'Resume':'Pause';$('robotClutch').style.background=s.preparing||s.resume_pending||!s.paused?'#8b412d':'#226b50';
 for(const input of document.querySelectorAll('.camera-settings input,.camera-settings select,.camera-settings button')){
  if(s.preparing&&!input.disabled){input.dataset.homeLocked='1';input.disabled=true}
  else if(!s.preparing&&input.dataset.homeLocked){delete input.dataset.homeLocked;input.disabled=false}
 }
 $('robotSensitivity').disabled=s.preparing;
 $('robotConnect').disabled=robotBusy||s.connected;$('robotEnable').disabled=robotBusy||!s.connected||(s.enabled&&!s.fault);
 $('robotClutch').disabled=robotBusy;$('robotEmergency').disabled=robotBusy||!s.connected;
 $('robotBackend').disabled=s.connected;
 if(document.activeElement?.id!=='robotSensitivity')$('robotSensitivity').value=Math.round(s.scale*100);
 $('robotSensitivityValue').textContent='10:'+Number($('robotSensitivity').value)/10;
 $('robotFeedback').textContent=s.joints_deg?'Joints / deg: '+s.joints_deg.map(v=>v.toFixed(2)).join(' / ')+'\nInitial / deg: '+(s.initial_joints_deg||s.rest_joints_deg).map(v=>v.toFixed(2)).join(' / ')+'\nMeasured end / mm: '+s.position_mm.map(v=>v.toFixed(2)).join(' / ')+(s.solver_info?'\nPND iterations '+s.solver_info.qp_iters+' · '+s.solver_info.solve_ms.toFixed(2)+' ms · residual '+s.solver_info.projected_residual_rad_s.toExponential(2):''):'Waiting for CAN feedback';
 if(s.input_filter)$('robotFeedback').textContent+='\nAdaptive input filter · '+(s.input_filter.accepted?'Frame accepted':'Outlier rejected');
 $('robotResultStatus').textContent=s.session_dir?'Results: '+s.session_dir+(s.plotting?' · Generating figures':s.plot_error?' · Figure error: '+s.plot_error:s.plots?' · Overview and signal figures saved':' · Recording; Stop & Save to generate figures'):'';
}
async function robotAction(action,step){
 if(robotBusy)return;robotBusy=true;robotActionError='';
 try{
  if(robotConnected&&robotState?.controller_id!==clientId&&!['stop','emergency','connect'].includes(action))renderRobot(await api('/api/robot',{action:'connect',client_id:clientId,backend:$('robotBackend').value}));
  renderRobot(await api('/api/robot',{client_id:clientId,action,step,backend:$('robotBackend').value,bench_clear:benchOk()}));
  if(['stop','disconnect','emergency'].includes(action)){startupReady=false;startupActive=false;startupToken++}
  if(['disconnect','emergency'].includes(action))benchConfirmed=false;
 }
 catch(e){robotActionError=e.message;$('robotStatus').textContent='Action failed: '+e.message}
 finally{robotBusy=false}
}
$('robotRest').onclick=()=>{
 if(robotBusy)return;
 if($('robotBackend').value!=='mock'&&!window.confirm('Finish & Home: the arm returns to the same initial pose used by Start. Motors remain enabled to hold it.\n\nScissors removed and nobody in the workspace?'))return;
 benchConfirmed=benchConfirmed||$('robotBackend').value==='mock'||true;
 startupReady=false;robotAction('rest');
};
for(const [id,action] of [['robotConnect','connect'],['robotEnable','enable'],['robotPrepare','prepare'],['robotClutch','clutch'],['robotStop','stop'],['robotEmergency','emergency'],['robotDisconnect','disconnect']])$(id).onclick=()=>robotAction(action);
$('robotClutch').onclick=async()=>{
 if(robotBusy)return;
 if(startupActive&&robotState?.preparing){startupActive=false;startupToken++;startupReady=false;startupSteps.home='Paused';startupProgress();return robotAction('clutch')}
 if(robotState?.preparing||startupReady&&robotState?.enabled&&robotState?.initial_ready&&!robotState?.fault)return robotAction('clutch');
 return startSystem();
};
async function startSystem(){
 if(robotBusy||startupActive)return;
 if(!confirmBench('Start: return to the start pose'))return;
 const token=++startupToken;startupActive=true;startupReady=false;robotBusy=true;robotActionError='';
 const cancelled=()=>token!==startupToken;
 const wait=ms=>new Promise(r=>setTimeout(r,ms));
 async function readyFrame(url,label){
  let reason='Waiting for first frame';const until=performance.now()+12000;
  while(performance.now()<until){
   if(cancelled())throw Error('Startup cancelled');
   const r=await fetch(url,{cache:'no-store'});
   if(r.ok){const blob=await r.blob();if(blob.size)return}
   else reason=await r.text();
   await wait(200);
  }
  throw Error(label+' failed to start: '+reason);
 }
 try{
  startupSteps.phone='Opening';startupSteps.camera='Off';startupSteps.simulation='Off';startupSteps.home='Waiting';startupProgress();
  if(stream&&!stream.getVideoTracks().some(t=>t.readyState==='live'))stream=null;
  if(!stream)await $('open').onclick();
  if(!stream)throw Error('Allow the phone camera first.');
  startupSteps.phone='Ready';startupSteps.camera='Opening';startupProgress();
  const camera=await api('/api/observer',{action:'start'});observerOn=camera.running;
  if(!observerOn)throw Error(camera.error||'Real camera did not start');
  await readyFrame('/api/observer/frame.jpg','Real camera');
  startupSteps.camera='Ready';startupSteps.simulation='Opening';startupProgress();
  renderRobot(await api('/api/robot',{action:'connect',client_id:clientId,backend:$('robotBackend').value}));
  const sim=await (await fetch('/api/sim/state')).json();
  if(sim.running&&!sim.paused)await api('/api/sim',{action:'pause',client_id:clientId});
  renderSim(await api('/api/sim',{action:'start',client_id:clientId,observer:true,solver:'PND'}));
  await readyFrame('/api/sim/frame.jpg','MuJoCo');
  startupSteps.simulation='Ready';startupSteps.home='Returning Home';startupProgress();
  renderRobot(await api('/api/robot',{action:'prepare',client_id:clientId,bench_clear:benchOk()}));
  robotBusy=false;
  const until=performance.now()+65000;
  while(performance.now()<until){
   if(cancelled())throw Error('Startup cancelled');
   const state=await (await fetch('/api/robot/state',{cache:'no-store'})).json();renderRobot(state);
   if(state.fault)throw Error(state.reason);
   if(!state.preparing){if(!state.initial_ready)throw Error(state.reason);startupReady=true;startupSteps.home='Ready';startupProgress();renderRobot(state);return}
   await wait(200);
  }
  throw Error('Arm did not reach home in time');
 }catch(e){if(!cancelled()){robotActionError=e.message;$('robotStatus').textContent='Action failed: '+e.message;startupSteps.home='Not ready';startupProgress()}}
 finally{if(!cancelled())startupActive=false;robotBusy=false}
}
$('robotSensitivity').oninput=()=>{$('robotSensitivityValue').textContent='10:'+Number($('robotSensitivity').value)/10};$('robotSensitivity').onchange=()=>robotAction('sensitivity',Number($('robotSensitivity').value)/100);
async function observerAction(action){try{const s=await api('/api/observer',{action});observerOn=s.running;$('observerStatus').textContent=s.name+(s.error?' · '+s.error:s.running?' · Waiting for frames':' · Off')}catch(e){$('observerStatus').textContent=e.message}}
$('observerStart').onclick=()=>observerAction('start');$('observerStop').onclick=()=>observerAction('stop');
async function observerFrame(){
 if(!observerOn||observerBusy)return;observerBusy=true;
 try{const r=await fetch('/api/observer/frame.jpg',{cache:'no-store'});if(!r.ok){$('observerStatus').textContent=await r.text();return}
 const next=URL.createObjectURL(await r.blob());$('observerImage').src=next;if(observerURL)URL.revokeObjectURL(observerURL);observerURL=next;$('observerStatus').textContent='Real camera · '+new Date().toLocaleTimeString();
 }catch(e){$('observerStatus').textContent='Camera connection lost: '+e.message}finally{observerBusy=false}
}
setInterval(observerFrame,150);
setInterval(async()=>{if(robotBusy)return;try{const r=await fetch('/api/robot/state',{cache:'no-store'});if(r.ok)renderRobot(await r.json())}catch(e){}},400);

$('robotPrepare').onclick=async()=>{
 if(robotBusy||robotState?.preparing)return;
 confirmBench('Return to Initial');
 return robotAction('return_initial');
};
$('alignHead').onclick=async()=>{
 if(robotBusy)return;robotBusy=true;
 $('headAlignmentStatus').textContent='Reading flange feedback (clipper tip from the tool model) and aligning virtual head…';
 try{
  renderRobot(await api('/api/robot',{action:'connect',client_id:clientId,backend:$('robotBackend').value}));
  const sim=await (await fetch('/api/sim/state',{cache:'no-store'})).json();
  if(!sim.running)renderSim(await api('/api/sim',{action:'start',client_id:clientId,observer:true,solver:'PND'}));
  const result=await api('/api/robot',{action:'align_head',client_id:clientId});
  const record=result.alignment;renderRobot(result.robot);
  const until=performance.now()+6000;let applied=false;
  while(performance.now()<until){
   const state=await (await fetch('/api/sim/state',{cache:'no-store'})).json();
   if(state.head_alignment_applied===record.revision){applied=true;break}
   await new Promise(r=>setTimeout(r,200));
  }
  $('headAlignmentStatus').textContent=(applied?'Aligned':'Saved; MuJoCo update not confirmed')+' · Crown / world mm: '+record.crown_world_m.map(x=>(x*1000).toFixed(1)).join(' / ')+' · Clipper-tip gap: 10 mm · Translation only. Saved: results/calibration/head_alignment.json';
 }catch(e){$('headAlignmentStatus').textContent='Alignment failed: '+e.message}
 finally{robotBusy=false;if(robotState)renderRobot(robotState)}
};
startupProgress();
