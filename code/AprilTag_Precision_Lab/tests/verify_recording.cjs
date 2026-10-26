const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes={};const $=id=>nodes[id]??=( {textContent:'',disabled:false,hidden:true,click(){this.clicked=true}} );
let pickerOptions,uploaded=null,trackStopped=false,unload;
const track={addEventListener(){},stop(){trackStopped=true}};
class Recorder{
 static isTypeSupported(m){return m.includes('vp9')}
 constructor(stream,opts){this.state='inactive';this.mimeType=opts.mimeType;Recorder.last=this}
 start(){this.state='recording'}
 stop(){this.state='inactive';this.ondataavailable({data:new Blob([new Uint8Array([26,69,223,163,0])])});this.onstop()}
}
const context={$,navigator:{mediaDevices:{async getDisplayMedia(options){pickerOptions=options;return {getTracks:()=>[track],getVideoTracks:()=>[track]}}}},window:{MediaRecorder:Recorder,addEventListener(name,fn){unload=fn}},MediaRecorder:Recorder,Blob,URL,Date,setInterval:()=>1,clearInterval:()=>{},fetch:async(url,options)=>{uploaded={url,bytes:await options.body.arrayBuffer()};return {ok:true,json:async()=>({path:'project/results/videos/test.webm'})}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(require('path').join(__dirname,'../web/recording.js'),'utf8'),context);
(async()=>{
 await $('recordInterface').onclick();assert.equal(Recorder.last.state,'recording');assert.equal(pickerOptions.audio,false);assert(pickerOptions.preferCurrentTab);
 let prevented=false;unload({preventDefault(){prevented=true}});assert(prevented);
 await $('recordInterface').onclick();
 for(let i=0;i<30&&!uploaded;i++)await new Promise(r=>setTimeout(r,5));
 assert(uploaded&&uploaded.url.startsWith('/api/recordings?ext=webm'));assert.equal(uploaded.bytes.byteLength,5);
 for(let i=0;i<10&&$('recordInterface').disabled;i++)await new Promise(r=>setTimeout(r,5));
 assert(trackStopped);assert.equal($('recordInterface').disabled,false);assert($('videoRecordStatus').textContent.includes('project/results/videos'));
 assert.equal($('downloadVideo').hidden,false);assert(!$('downloadVideo').clicked,'should save to project instead of automatic Downloads');
 context.navigator.mediaDevices.getDisplayMedia=async()=>{throw Object.assign(new Error('cancel'),{name:'NotAllowedError'})};
 await $('recordInterface').onclick();assert.equal($('recordInterface').disabled,false);assert($('videoRecordStatus').textContent.includes('cancelled'));
 console.log('RECORDING CONTROLS PASS: share picker, start/stop, upload, unload guard, fallback download, cancellation (MediaRecorder mocked)');
})().catch(e=>{console.error(e);process.exitCode=1});
