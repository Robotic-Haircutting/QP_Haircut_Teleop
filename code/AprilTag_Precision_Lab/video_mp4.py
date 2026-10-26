"""Finalize browser recordings into seekable H.264 MP4 using local FFmpeg."""
import json,shutil,subprocess,sys
from pathlib import Path

def executable(name):
    found=shutil.which(name)
    if found:return found
    path=Path(sys.prefix)/'Library/bin'/f'{name}.exe'
    if path.exists():return str(path)
    raise OSError(f'{name} not found on this computer')

def inspect_video(path):
    result=subprocess.run([executable('ffprobe'),'-v','error','-show_entries',
        'format=duration:stream=codec_name,width,height,pix_fmt,duration','-of','json',str(path)],
        capture_output=True,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    data=json.loads(result.stdout)
    duration=float(data.get('format',{}).get('duration',0))
    if duration<=0:raise ValueError('Video has no valid duration')
    if not data['streams'] or data['streams'][0]['codec_name']!='h264':raise ValueError('Video is not H.264')
    return data

def convert_mp4(source,output):
    source,output=Path(source),Path(output)
    if output.exists():return inspect_video(output)
    partial=output.with_suffix('.partial.mp4')
    command=[executable('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-y','-fflags','+genpts',
        '-i',str(source),'-map','0:v:0','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2',
        '-movflags','+faststart',str(partial)]
    try:
        result=subprocess.run(command,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise OSError('MP4 conversion failed: '+result.stderr.decode('utf-8',errors='replace')[-1500:])
        info=inspect_video(partial)
        partial.replace(output)
        return info
    finally:partial.unlink(missing_ok=True)

def verify_seek(path):
    info=inspect_video(path);duration=float(info['format']['duration'])
    # Actually decode a frame after a seek, rather than just reading a duration field.
    result=subprocess.run([executable('ffmpeg'),'-v','error','-ss',str(duration*.6),'-i',str(path),
        '-frames:v','1','-f','image2pipe','-c:v','png','-'],capture_output=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode or not result.stdout.startswith(b'\x89PNG'):raise ValueError('MP4 seek-and-decode check failed')
    return info
