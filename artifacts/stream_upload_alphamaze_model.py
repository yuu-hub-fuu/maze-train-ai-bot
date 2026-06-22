import os
import posixpath
import shlex
import sys
import time

import paramiko


HOST = "i-1.gpushare.com"
PORT = 16496
USER = "root"
LOCAL_MODEL = (
    r"C:\Users\gjy10\.cache\huggingface\hub\models--Menlo--AlphaMaze-v0.2-1.5B"
    r"\snapshots\2a7c08f3614fa672bb1be573b210b3b5e575cd30\model.safetensors"
)
REMOTE_MODEL = (
    "/root/.cache/huggingface/hub/models--Menlo--AlphaMaze-v0.2-1.5B/"
    "snapshots/2a7c08f3614fa672bb1be573b210b3b5e575cd30/model.safetensors"
)
CHUNK = 4 * 1024 * 1024
REPORT_EVERY_SECONDS = 5


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def fmt_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def ensure_remote_parent(ssh: paramiko.SSHClient, remote_path: str) -> None:
    parent = posixpath.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(parent)}"
    stdin, stdout, stderr = ssh.exec_command(cmd)
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(stderr.read().decode("utf-8", "replace"))


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int:
    try:
        return sftp.stat(path).st_size
    except FileNotFoundError:
        return 0


def main() -> int:
    password = os.environ.get("GPU_PW")
    if not password:
        print("GPU_PW is required", file=sys.stderr)
        return 2

    local_size = os.path.getsize(LOCAL_MODEL)
    log(f"local file: {LOCAL_MODEL}")
    log(f"local size: {local_size} bytes ({fmt_bytes(local_size)})")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        HOST,
        port=PORT,
        username=USER,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    try:
        ensure_remote_parent(ssh, REMOTE_MODEL)
        sftp = ssh.open_sftp()
        try:
            offset = remote_size(sftp, REMOTE_MODEL)
        finally:
            sftp.close()

        if offset > local_size:
            raise RuntimeError(f"remote file is larger than local: {offset} > {local_size}")
        if offset == local_size:
            log("remote model already complete; nothing to upload")
            return 0

        remaining = local_size - offset
        log(f"remote path: {REMOTE_MODEL}")
        log(f"resume offset: {offset} bytes ({fmt_bytes(offset)})")
        log(f"remaining: {remaining} bytes ({fmt_bytes(remaining)})")

        cmd = f"cat >> {shlex.quote(REMOTE_MODEL)}"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        channel = stdin.channel
        start = time.time()
        last_report = start
        last_sent = offset
        sent_total = offset

        with open(LOCAL_MODEL, "rb") as handle:
            handle.seek(offset)
            while True:
                data = handle.read(CHUNK)
                if not data:
                    break
                stdin.write(data)
                sent_total += len(data)
                now = time.time()
                if now - last_report >= REPORT_EVERY_SECONDS:
                    delta = sent_total - last_sent
                    speed = delta / max(now - last_report, 0.001)
                    done = sent_total - offset
                    overall_speed = done / max(now - start, 0.001)
                    eta = (local_size - sent_total) / max(overall_speed, 1)
                    pct = sent_total * 100 / local_size
                    log(
                        "progress "
                        f"{pct:.2f}% | {fmt_bytes(sent_total)} / {fmt_bytes(local_size)} | "
                        f"window {fmt_bytes(speed)}/s | avg {fmt_bytes(overall_speed)}/s | "
                        f"eta {fmt_eta(eta)}"
                    )
                    last_report = now
                    last_sent = sent_total

        stdin.flush()
        stdin.channel.shutdown_write()
        rc = channel.recv_exit_status()
        err = stderr.read().decode("utf-8", "replace").strip()
        out = stdout.read().decode("utf-8", "replace").strip()
        if out:
            log(f"remote stdout: {out}")
        if err:
            log(f"remote stderr: {err}")
        if rc != 0:
            raise RuntimeError(f"remote append failed with exit code {rc}")

        sftp = ssh.open_sftp()
        try:
            final_size = remote_size(sftp, REMOTE_MODEL)
        finally:
            sftp.close()
        elapsed = time.time() - start
        log(f"final remote size: {final_size} bytes ({fmt_bytes(final_size)})")
        log(f"elapsed: {fmt_eta(elapsed)}")
        if final_size != local_size:
            raise RuntimeError(f"incomplete upload: {final_size} != {local_size}")
        log("upload complete")
        return 0
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
