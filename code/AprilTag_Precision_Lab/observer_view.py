"""Shared camera client for native MuJoCo right-side video (no second camera open)."""
import ssl,threading,time,urllib.request
import cv2,numpy as np

class ObserverView:
    def __init__(self,url):
        self.url=url;self.frame=None;self.last=0;self.alive=True
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
    def _loop(self):
        context=ssl._create_unverified_context()
        while self.alive:
            try:
                with urllib.request.urlopen(self.url,context=context,timeout=.5) as response:raw=response.read()
                frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if frame is not None:self.frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);self.last=time.monotonic()
            except Exception:pass
            time.sleep(.08)
    def image(self,width,height):
        canvas=np.zeros((height,width,3),np.uint8);canvas[:]=[20,25,32]
        if self.frame is not None and time.monotonic()-self.last<1:
            frame=self.frame;scale=min(width/frame.shape[1],(height-30)/frame.shape[0]);w=max(1,int(frame.shape[1]*scale));h=max(1,int(frame.shape[0]*scale));x=(width-w)//2;y=30+(height-30-h)//2
            canvas[y:y+h,x:x+w]=cv2.resize(frame,(w,h))
        else:cv2.putText(canvas,'Camera waiting / stopped',(10,height//2),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
        cv2.putText(canvas,'PIPER REAL CAMERA',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1)
        return np.ascontiguousarray(canvas)
    def close(self):self.alive=False;self.thread.join(timeout=1)
