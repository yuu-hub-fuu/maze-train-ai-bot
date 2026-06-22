import os
import posixpath

import paramiko


HOST = "i-1.gpushare.com"
PORT = 16496
USER = "root"
LOCAL = "scripts/hf_lora_train_alpha_maze_compat_v2.py"
REMOTE = "/hy-tmp/maze-train-ai-bot/scripts/hf_lora_train_alpha_maze_compat.py"


password = os.environ["GPU_PW"]
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=password, timeout=30, banner_timeout=30, auth_timeout=30)
sftp = ssh.open_sftp()
try:
    parent = posixpath.dirname(REMOTE)
    ssh.exec_command(f"mkdir -p {parent}")[1].channel.recv_exit_status()
    sftp.put(LOCAL, REMOTE)
    print(f"uploaded {LOCAL} -> {REMOTE}")
finally:
    sftp.close()
    ssh.close()
