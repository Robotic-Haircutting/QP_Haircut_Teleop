"use strict";
// Piper-page wording for elements shared with the simulation page; status fields written by piper.js are left alone.
const englishLabels=new Map([...document.querySelectorAll('button[id]')].map(e=>[e.id,e.textContent]));
const piperOwned=new Set(['headAlignmentStatus','observerStatus','robotClutch','robotFeedback','robotResultStatus','robotSensitivityValue','robotStatus','startupStatus','workflowStatus']);
function englishField(e,text){
 const id=e.id;
 if(englishLabels.has(id))return id==='recordInterface'&&text.startsWith('Stop')?'Stop Recording & Save':englishLabels.get(id);
 if(id==='cameraInfo')return 'Detection camera: '+(stream?.getVideoTracks()[0]?.label||'phone rear camera');
 if(id==='modeHint')return 'Fixed Tags / moving phone: 6DOF control with PND box QP.';
 if(id==='boardGeometryHint')return 'Top: IDs 0 / 1; bottom: IDs 2 / 3. All Tags fixed on one flat board; world origin at board center.';
 if(id==='trackingOrigin')return $('trackingMode').value==='board'?'World origin: fixed four-Tag board center.':'World origin: fixed Tag center.';
 if(id==='cameraWideHint')return 'Preview cropping does not change the detection image or lens field of view.';
 if(id==='quality')return calibration?(calibration.calibrated?'Calibrated: ':'Uncalibrated estimate: ')+calibration.width+' × '+calibration.height+' · RMS '+Number(calibration.rms_px||0).toFixed(3)+' px. Validate accuracy with measured physical motion.':'Camera calibration required.';
 if(id==='poseReference')return robotState?.preparing?'Phone control locked · Waiting for home.':robotState?.initial_ready?'Arm ready · Resume sets a new phone reference.':'Home the arm before phone control.';
 if(id==='robotLimitStatus')return 'Joint feedback and limits: see Feedback & Results.';
 if(id==='simStatus')return 'Simulation display · Use Arm Control when Piper is connected.';
 if(id==='dataLogStatus')return 'Simulation data recorded separately. Stop Simulation to generate figures.';
 if(id==='simAxes')return 'Camera to robot axis mapping · Configure in the simulation-only page.';
 return text;
}
function localize(root){
 if(!root||root.nodeType!==1)return;
 const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let node;
 while(node=walker.nextNode()){
  if(piperOwned.has(node.parentElement.id)||node.parentElement.closest('script,style'))continue;
  const next=englishField(node.parentElement,node.nodeValue);if(next!==node.nodeValue)node.nodeValue=next;
 }
}
const englishObserver=new MutationObserver(records=>{for(const record of records)localize(record.target.nodeType===1?record.target:record.target.parentElement)});
englishObserver.observe(document.querySelector('main'),{subtree:true,childList:true,characterData:true});
localize(document.querySelector('main'));
