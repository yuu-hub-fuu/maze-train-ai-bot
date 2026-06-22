import argparse
import os
import posixpath
import shlex
import time
from pathlib import Path

import paramiko


HOST = "i-1.gpushare.com"
PORT = 16496
USER = "root"
REMOTE_PROJECT = "/hy-tmp/maze-train-ai-bot"
REMOTE_CACHE_MODEL = (
    "/root/.cache/huggingface/hub/models--Menlo--AlphaMaze-v0.2-1.5B/"
    "snapshots/2a7c08f3614fa672bb1be573b210b3b5e575cd30/model.safetensors"
)
LOCAL_TRAIN_SCRIPT = Path("scripts/hf_lora_train_alpha_maze_compat.py")
REMOTE_TRAIN_SCRIPT = f"{REMOTE_PROJECT}/scripts/hf_lora_train_alpha_maze_compat.py"


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def connect() -> paramiko.SSHClient:
    password = os.environ["GPU_PW"]
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
    return ssh


def run(ssh: paramiko.SSHClient, command: str, log_path: Path | None = None) -> int:
    stdin, stdout, stderr = ssh.exec_command(command)
    channel = stdout.channel
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    with (log_path.open("w", encoding="utf-8") if log_path else open(os.devnull, "w", encoding="utf-8")) as log:
        log.write(f"[{stamp()}] COMMAND\n{command}\n\n")
        while not channel.exit_status_ready():
            while channel.recv_ready():
                text = channel.recv(65536).decode("utf-8", "replace")
                out_chunks.append(text)
                print(text, end="", flush=True)
                log.write(text)
                log.flush()
            while channel.recv_stderr_ready():
                text = channel.recv_stderr(65536).decode("utf-8", "replace")
                err_chunks.append(text)
                print(text, end="", flush=True)
                log.write(text)
                log.flush()
            time.sleep(0.2)
        while channel.recv_ready():
            text = channel.recv(65536).decode("utf-8", "replace")
            out_chunks.append(text)
            print(text, end="", flush=True)
            log.write(text)
        while channel.recv_stderr_ready():
            text = channel.recv_stderr(65536).decode("utf-8", "replace")
            err_chunks.append(text)
            print(text, end="", flush=True)
            log.write(text)
        rc = channel.recv_exit_status()
        log.write(f"\n[{stamp()}] EXIT {rc}\n")
    return rc


def upload_train_script(ssh: paramiko.SSHClient) -> None:
    sftp = ssh.open_sftp()
    try:
        parent = posixpath.dirname(REMOTE_TRAIN_SCRIPT)
        run(ssh, f"mkdir -p {shlex.quote(parent)}")
        sftp.put(str(LOCAL_TRAIN_SCRIPT), REMOTE_TRAIN_SCRIPT)
        print(f"uploaded {LOCAL_TRAIN_SCRIPT} -> {REMOTE_TRAIN_SCRIPT}")
    finally:
        sftp.close()


def verify_model(ssh: paramiko.SSHClient) -> int:
    command = f"""bash -lc {shlex.quote(f'''
set -e
cd {REMOTE_PROJECT}
source /hy-tmp/miniconda3/etc/profile.d/conda.sh
conda activate maze-agent
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
stat -c "model_size=%s path=%n" {shlex.quote(REMOTE_CACHE_MODEL)}
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Menlo/AlphaMaze-v0.2-1.5B"
tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
print("tokenizer_ok", tok.__class__.__name__, len(tok))
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
print("model_ok", type(model).__name__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("allocated_gb", round(torch.cuda.memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0)
PY
''')}"""
    return run(ssh, command, Path("artifacts/remote_verify_alphamaze_model.log"))


def start_train(ssh: paramiko.SSHClient) -> int:
    remote_runner = f"{REMOTE_PROJECT}/run_train_alphamaze_lora_cached.sh"
    remote_log = f"{REMOTE_PROJECT}/logs/train_alphamaze_lora_cached.log"
    remote_pid = f"{REMOTE_PROJECT}/logs/train_alphamaze_lora_cached.pid"
    runner = f"""#!/usr/bin/env bash
set -euo pipefail
cd {REMOTE_PROJECT}
source /hy-tmp/miniconda3/etc/profile.d/conda.sh
conda activate maze-agent
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
python scripts/hf_lora_train_alpha_maze_compat.py \\
  --data train/hf_sft_messages.jsonl \\
  --out artifacts/hf_lora/alphamaze-course-lora-main \\
  --max-samples 3986 \\
  --max-steps 300 \\
  --seq-len 1536
"""
    quoted_runner = shlex.quote(runner)
    command = f"""bash -lc {shlex.quote(f'''
set -e
cd {REMOTE_PROJECT}
mkdir -p logs artifacts/hf_lora
cat > {shlex.quote(remote_runner)} <<'SCRIPT'
{runner}
SCRIPT
chmod +x {shlex.quote(remote_runner)}
nohup bash {shlex.quote(remote_runner)} > {shlex.quote(remote_log)} 2>&1 &
echo $! > {shlex.quote(remote_pid)}
echo started_train_pid=$(cat {shlex.quote(remote_pid)})
echo train_log={remote_log}
''')}"""
    return run(ssh, command, Path("artifacts/remote_start_train.log"))


def status(ssh: paramiko.SSHClient) -> int:
    command = f"""bash -lc {shlex.quote(f'''
cd {REMOTE_PROJECT}
echo "--- pid ---"
cat logs/train_alphamaze_lora_cached.pid 2>/dev/null || true
echo "--- ps ---"
pid=$(cat logs/train_alphamaze_lora_cached.pid 2>/dev/null || true)
if [ -n "$pid" ]; then ps -fp "$pid" || true; fi
echo "--- gpu ---"
nvidia-smi || true
echo "--- train log tail ---"
tail -80 logs/train_alphamaze_lora_cached.log 2>/dev/null || true
''')}"""
    return run(ssh, command, Path("artifacts/remote_train_status.log"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upload-train-script", "verify-model", "start-train", "status"])
    args = parser.parse_args()
    ssh = connect()
    try:
        if args.action == "upload-train-script":
            upload_train_script(ssh)
            return 0
        if args.action == "verify-model":
            return verify_model(ssh)
        if args.action == "start-train":
            return start_train(ssh)
        if args.action == "status":
            return status(ssh)
        raise AssertionError(args.action)
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
