import os

import paramiko


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    "i-1.gpushare.com",
    port=16496,
    username="root",
    password=os.environ["GPU_PW"],
    timeout=30,
    banner_timeout=30,
    auth_timeout=30,
)
cmd = r"""
pid=$(cat /hy-tmp/maze-train-ai-bot/logs/train_alphamaze_lora_cached.pid 2>/dev/null || true)
if [ -n "$pid" ]; then
  pkill -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
fi
echo stopped_pid=$pid
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
"""
stdin, out, err = ssh.exec_command(cmd)
print(out.read().decode("utf-8", "replace"))
print(err.read().decode("utf-8", "replace"))
ssh.close()
