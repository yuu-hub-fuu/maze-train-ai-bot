import os
import time
from pathlib import Path
import posixpath
import paramiko

LOCAL_ROOT = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--Menlo--AlphaMaze-v0.2-1.5B'
REMOTE_ROOT = '/root/.cache/huggingface/hub/models--Menlo--AlphaMaze-v0.2-1.5B'
LOG = Path('artifacts/upload_alphamaze_cache.log')
LOG.parent.mkdir(parents=True, exist_ok=True)

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

uploaded_total = 0
start = time.time()
last_print = [0.0]
with LOG.open('w', encoding='utf-8') as log:
    def emit(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    emit(f'start upload {len(files)} files total={total/1024/1024:.1f} MB')
    for idx, local in enumerate(files, 1):
        rel = local.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(REMOTE_ROOT, rel)
        size = local.stat().st_size
        try:
            st = sftp.stat(remote)
            if st.st_size == size:
                uploaded_total += size
                emit(f'skip {idx}/{len(files)} {rel} {size/1024/1024:.1f} MB already exists')
                continue
        except IOError:
            pass

        file_sent_base = uploaded_total
        file_start = time.time()
        def cb(sent, file_size):
            now = time.time()
            if now - last_print[0] >= 5 or sent == file_size:
                done = file_sent_base + sent
                speed = done / max(1e-6, now - start) / 1024 / 1024
                fspeed = sent / max(1e-6, now - file_start) / 1024 / 1024
                emit(f'upload {idx}/{len(files)} {rel}: file {sent/1024/1024:.1f}/{file_size/1024/1024:.1f} MB, total {done/1024/1024:.1f}/{total/1024/1024:.1f} MB ({done/total*100:.1f}%), avg {speed:.2f} MB/s, file {fspeed:.2f} MB/s')
                last_print[0] = now
        sftp.put(str(local), remote, callback=cb)
        uploaded_total += size
    emit('upload complete')

sftp.close()
stdin, out, err = client.exec_command(f"find {REMOTE_ROOT} -type f -printf '%s %p\\n' | sort -nr | head -20")
print(out.read().decode(errors='replace'))
print(err.read().decode(errors='replace'))
client.close()
