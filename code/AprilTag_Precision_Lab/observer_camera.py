"""Named external DirectShow camera -> bounded latest JPEG cache."""
import subprocess,time,threading
from video_mp4 import executable

class ObserverCamera:
    name='WN CAMERA 1080'
    def __init__(self):self.lock=threading.RLock();self.process=None;self.frame=None;self.last=0;self.thread=None;self.error='';self.log=None
    def start(self):
        with self.lock:
            if self.process and self.process.poll() is None:return
            self.stop();self.frame=None;self.last=0;self.error=''
            import tempfile
            self.log=tempfile.TemporaryFile()
            self.process=subprocess.Popen([executable('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-f','dshow','-i','video='+self.name,'-an','-vf','scale=640:-2','-r','10','-c:v','mjpeg','-q:v','5','-threads','1','-f','image2pipe','pipe:1'],stdout=subprocess.PIPE,stderr=self.log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            process=self.process
            def read():
                buffer=b''
                while process.poll() is None:
                    chunk=process.stdout.read1(8192)
                    if not chunk:break
                    buffer+=chunk
                    while True:
                        begin=buffer.find(b'\xff\xd8');end=buffer.find(b'\xff\xd9',max(0,begin+2))
                        if begin<0 or end<0:break
                        frame=buffer[begin:end+2];buffer=buffer[end+2:]
                        with self.lock:
                            if self.process is process:self.frame=frame;self.last=time.monotonic()
                    if len(buffer)>2*1024*1024:buffer=b''
                with self.lock:
                    if self.process is process and process.poll() not in (None,0):
                        self.log.seek(0);self.error=self.log.read().decode('utf-8',errors='replace')[-500:]
            self.thread=threading.Thread(target=read,daemon=True);self.thread.start()
    def stop(self):
        with self.lock:
            process=self.process;self.process=None
            if process:
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
                process.stdout.close()
            if self.log:self.log.close();self.log=None
            self.frame=None;self.last=0
    def state(self):
        with self.lock:return dict(name=self.name,running=bool(self.process and self.process.poll() is None),age_seconds=time.monotonic()-self.last if self.last else None,error=self.error)
