'use strict';
let screenRecorder=null,recordingStream=null,videoChunks=[],videoUrl=null,recordingStarted=0,recordingTimer=null,requestingRecording=false,savingRecording=false;
const videoStatus=$('videoRecordStatus'),recordButton=$('recordInterface'),videoLink=$('downloadVideo');
function videoClock(){const seconds=Math.floor((Date.now()-recordingStarted)/1000);videoStatus.textContent=`● Recording ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')} · Stop to save video`;}
function stopInterfaceRecording(){if(screenRecorder&&screenRecorder.state!=='inactive')screenRecorder.stop();}
recordButton.onclick=async()=>{
  if(screenRecorder&&screenRecorder.state!=='inactive'){stopInterfaceRecording();return;}
  if(requestingRecording)return;
  if(!navigator.mediaDevices?.getDisplayMedia||!window.MediaRecorder){videoStatus.textContent='Screen recording is unavailable in this browser. Use desktop Chrome or Edge.';return;}
  requestingRecording=true;recordButton.disabled=true;
  try{
    recordingStream=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:{ideal:20,max:30}},audio:false,preferCurrentTab:true});
    const mime=['video/mp4;codecs=avc1.42001E','video/mp4','video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(m=>MediaRecorder.isTypeSupported(m));
    const recorder=new MediaRecorder(recordingStream,mime?{mimeType:mime}:{});screenRecorder=recorder;videoChunks=[];
    recorder.ondataavailable=e=>{if(e.data.size)videoChunks.push(e.data);};
    recorder.onerror=e=>{videoStatus.textContent='Recording error: '+(e.error?.message||'Unknown error');stopInterfaceRecording();};
    recorder.onstop=async()=>{
      clearInterval(recordingTimer);recordingStream?.getTracks().forEach(t=>t.stop());recordingStream=null;
      recordButton.textContent='Record Video';screenRecorder=null;
      const type=recorder.mimeType||mime||'video/webm';const blob=new Blob(videoChunks,{type});videoChunks=[];
      if(!blob.size){videoStatus.textContent='No video recorded. Try again.';return;}
      if(videoUrl)URL.revokeObjectURL(videoUrl);videoUrl=URL.createObjectURL(blob);
      const name='Piper_interface_'+new Date(recordingStarted).toISOString().replace(/[:.]/g,'-')+(type.includes('mp4')?'.mp4':'.webm');
      videoLink.href=videoUrl;videoLink.download=name;videoLink.hidden=false;
      videoStatus.textContent='Saving a seekable MP4; keep this page open. Longer recordings take a while to convert…';
      savingRecording=true;
      recordButton.disabled=true;
      try{
        const ext=type.includes('mp4')?'mp4':'webm';
        const response=await fetch(`/api/recordings?ext=${ext}&started_ms=${recordingStarted}&entry=${location.pathname.startsWith('/piper')?'piper':'simulation'}`,{method:'POST',headers:{'Content-Type':type},body:blob});
        if(!response.ok)throw Error(await response.text());
        const saved=await response.json();videoStatus.textContent='MP4 saved: '+saved.path;
        if(saved.url){videoLink.href=saved.url;videoLink.download=saved.path.split(/[\\/]/).pop();videoLink.textContent='Download MP4 Copy';}
      }catch(e){videoStatus.textContent='Save failed: '+e.message+'; download a copy to retain the recording.';}
      finally{savingRecording=false;recordButton.disabled=false;}
    };
    recordingStream.getVideoTracks()[0].addEventListener('ended',stopInterfaceRecording,{once:true});
    recordingStarted=Date.now();recorder.start(1000);recordButton.textContent='Stop Recording & Save';videoClock();recordingTimer=setInterval(videoClock,1000);
  }catch(e){recordingStream?.getTracks().forEach(t=>t.stop());recordingStream=null;screenRecorder=null;videoStatus.textContent=e.name==='NotAllowedError'?'Screen capture cancelled or denied.':e.message;}
  finally{requestingRecording=false;recordButton.disabled=false;}
};
window.addEventListener('beforeunload',e=>{if(savingRecording||(screenRecorder&&screenRecorder.state!=='inactive')){e.preventDefault();e.returnValue='';}});
