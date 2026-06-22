import os
import time
from pathlib import Path
import posixpath
import paramiko

LOCAL_ROOT = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--Menlo--AlphaMaze-v0.2-1.5B'
REMOTE_ROOT = '/root/.cache/huggingface/hub/models--Menlo--AlphaMaze-v0.2-1.5B'
LOG = Path('artifacts/resume_upload_alphamaze_cache.log')
LOG.parent.mkdir(parents=True, exist_ok=True)
CHUNK = 4 * 1024 * 1024

files = [p for p in LOCAL_ROOT.rglob('*') if p.is_file()]
total = sum(p.stat().st_size for p in files)

password = os.environ['GPU_PW']
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('i-1.gpushare.com', port=16496, username='root', password=password, timeout=20, banner_timeout=20, auth_timeout=20)
sftp = client.open_sftp()

for d in sorted({p.parent for p in files}):
    rel = d.relative_to(LOCAL_ROOT).as_posix()
    remote_dir = REMOTE_ROOT if rel == '.' else posixpath.join(REMOTE_ROOT, rel)
    cur = ''
    for part in remote_dir.strip('/').split('/'):
        cur += '/' + part
        try:
            sftp.mkdir(cur)
        except IOError:
            pass

start = time.time()
last = [0.0]
completed_initial = 0
with LOG.open('w', encoding='utf-8') as log:
    def emit(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    emit(f'start resume upload {len(files)} files total={total/1024/1024:.1f} MB')

    # Count already-complete files toward total; partial files count their remote size.
    for local in files:
        rel = local.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(REMOTE_ROOT, rel)
        size = local.stat().st_size
        try:
            rsize = sftp.stat(remote).st_size
        except IOError:
            rsize = 0
        completed_initial += min(size, rsize)

    transferred_base = completed_initial
    emit(f'initial remote bytes={completed_initial/1024/1024:.1f} MB ({completed_initial/total*100:.1f}%)')

    for idx, local in enumerate(files, 1):
        rel = local.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(REMOTE_ROOT, rel)
        size = local.stat().st_size
        try:
            rsize = sftp.stat(remote).st_size
        except IOError:
            rsize = 0
        if rsize == size:
            emit(f'skip {idx}/{len(files)} {rel} {size/1024/1024:.1f} MB complete')
            continue
        if rsize > size:
            emit(f'remote larger than local, overwrite {rel}')
            rsize = 0
            rf = sftp.open(remote, 'wb')
        else:
            emit(f'resume {idx}/{len(files)} {rel} from {rsize/1024/1024:.1f}/{size/1024/1024:.1f} MB')
            rf = sftp.open(remote, 'ab') if rsize else sftp.open(remote, 'wb')
        sent_this_run = 0
        file_start = time.time()
        with local.open('rb') as lf, rf:
            lf.seek(rsize)
            while True:
                chunk = lf.read(CHUNK)
                if not chunk:
                    break
                rf.write(chunk)
                sent_this_run += len(chunk)
                now = time.time()
                current_done = transferred_base + sent_this_run
                if now - last[0] >= 10:
                    speed = sent_this_run / max(1e-6, now - file_start) / 1024 / 1024
                    avg = (current_done - completed_initial) / max(1e-6, now - start) / 1024 / 1024
                    total_effective = completed_initial + sent_this_run
                    emit(f'progress {rel}: remote {(rsize+sent_this_run)/1024/1024:.1f}/{size/1024/1024:.1f} MB, total {total_effective/1024/1024:.1f}/{total/1024/1024:.1f} MB ({total_effective/total*100:.1f}%), file {speed:.2f} MB/s, session {avg:.2f} MB/s')
                    last[0] = now
        transferred_base += sent_this_run
        emit(f'file done {rel}: now {size/1024/1024:.1f} MB')
    emit('resume upload complete')

sftp.close()
stdin, out, err = client.exec_command(f"find {REMOTE_ROOT} -type f -printf '%s %p\\n' | sort -nr | head -20")
print(out.read().decode(errors='replace'))
print(err.read().decode(errors='replace'))
client.close()
