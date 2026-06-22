# MazeGPT-Agent Run State

## Objective

Complete task 2 as a model-centered Maze Agent project:

- Traditional algorithms are teachers, evaluators, and fallback execution constraints.
- The final product is a post-trained Maze Agent for course-rule mazes with coins, traps, BOSS resource gates, 3x3 local resource pickup, full exploration visualization, remaining resource value, steps, and value/step score.

## Course PDF Constraints Confirmed

The full 3-page PDF was read. Task 2 requires:

- Greedy real-time resource pickup under 3x3 vision.
- Complete maze exploration from start to exit.
- Process visualization.
- Remaining resource value at exit.
- Step count.
- Ratio of remaining resource value to steps.
- Cross-test robustness on maze matrices provided by other groups.

## Local Project State

Workspace:

`C:\Users\gjy10\Desktop\顾秋雨\好玩项目\maze-train-ai-bot`

Implemented modules:

- `maze_gpt_agent/maze_env.py`: course maze environment with coins, one-shot traps, BOSS cost, scoring, prompts.
- `maze_gpt_agent/maze_generator.py`: perfect-maze style generator plus six course scenarios.
- `maze_gpt_agent/expert_solver.py`: state-space expert teacher using position, collected coins, triggered traps, BOSS state, gold, and steps.
- `maze_gpt_agent/dataset_builder.py`: SFT records and HuggingFace chat-message JSONL.
- `maze_gpt_agent/agents.py`: Greedy 3x3, BFS, A* resource, local policy fallback.
- `maze_gpt_agent/evaluator.py`: success, boss clear, gold, steps, score, traps, invalid actions.
- `maze_gpt_agent/visualizer.py`: HTML frame visualizer.

Generated local data:

- `artifacts/train/sft_records.jsonl`: 3986 expert state-action records.
- `artifacts/train/hf_sft_messages.jsonl`: HuggingFace SFT chat format.
- `artifacts/mazes/train_mazes.json`: generated train mazes.

Reference repos downloaded locally:

- `third_party/maze-dataset`
- `third_party/neural-astar`
- `third_party/pymaze`
- `third_party/decision-transformer`

## Actual Model Training State

Local AlphaMaze LoRA smoke adapter exists:

- Path: `artifacts/hf_lora/alphamaze-course-lora-smoke`
- Base model: `Menlo/AlphaMaze-v0.2-1.5B`
- Adapter: `adapter_model.safetensors`
- LoRA config: r=8, alpha=16, target modules q/k/v/o projection.
- This was a 5-step smoke run and is not the final training target.

Do not treat the local MLP/GRU policy as the final model. They are fallback/baseline proof-of-pipeline artifacts only.

## Remote Server State

Remote:

`ssh -p 16496 root@i-1.gpushare.com`

Do not store the password in this file.

Remote workspace:

`/hy-tmp/maze-train-ai-bot`

Remote data disk:

`/hy-tmp`, about 100G total.

Remote GPU:

`NVIDIA GeForce RTX 3090`, 24GB.

Remote project files are unpacked under `/hy-tmp/maze-train-ai-bot`.

Remote data paths after unpacking:

- `/hy-tmp/maze-train-ai-bot/train/hf_sft_messages.jsonl`
- `/hy-tmp/maze-train-ai-bot/train/sft_records.jsonl`
- `/hy-tmp/maze-train-ai-bot/mazes/train_mazes.json`

Remote logs:

- `/hy-tmp/maze-train-ai-bot/logs/env_probe.log`
- `/hy-tmp/maze-train-ai-bot/logs/install_conda_torch_cu118.log`

## Remote Environment Findings

`metion`:

- Python 3.7.12.
- CUDA works.
- Torch currently reports `1.13.1+cu117` after dependency install.
- Transformers is old enough that AlphaMaze/Qwen2 tokenizer/model path is not suitable.
- Use only if training a fallback neural policy, not for AlphaMaze LoRA.

`maze-agent`:

- Python 3.10.
- Modern HF stack installed.
- Torch installed as CUDA 13 build, but server driver is too old for CUDA 13, so CUDA is unavailable.

`maze-agent-cu118`:

- Python 3.10.
- Intended environment for AlphaMaze LoRA.
- Current task: install CUDA 11.8 compatible PyTorch via conda, then install modern HF packages.

## Current Active Remote Action

Installing PyTorch CUDA 11.8 into `maze-agent-cu118` using conda:

```bash
cd /hy-tmp/maze-train-ai-bot
bash install_conda_torch_cu118.sh
```

Background PID file:

`/hy-tmp/maze-train-ai-bot/logs/install_conda_torch_cu118.pid`

Log:

`/hy-tmp/maze-train-ai-bot/logs/install_conda_torch_cu118.log`

Started at:

`2026-06-22 21:15:19`

## Next Steps

1. Tail `/hy-tmp/maze-train-ai-bot/logs/install_conda_torch_cu118.log` until PyTorch CUDA 11.8 install completes.
2. Verify in `maze-agent-cu118`:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

3. Install modern HuggingFace dependencies in `maze-agent-cu118`:

```bash
python -m pip install transformers==4.46.3 datasets peft==0.13.2 trl==0.12.2 accelerate sentencepiece protobuf -i https://pypi.org/simple
```

4. Run AlphaMaze LoRA on the server:

```bash
cd /hy-tmp/maze-train-ai-bot
source /hy-tmp/miniconda3/etc/profile.d/conda.sh
conda activate maze-agent-cu118
export HF_ENDPOINT=https://hf-mirror.com
python scripts/hf_lora_train_alpha_maze.py \
  --data train/hf_sft_messages.jsonl \
  --out artifacts/hf_lora/alphamaze-course-lora-main \
  --max-samples 3986 \
  --max-steps 300
```

5. Download final adapter and evaluation artifacts back to local `artifacts/`.

## Working Rule From User

Do not be conservative. Do not stop at tiny smoke tests as if that were the goal. Use visible logs and progress, but push the full implementation/training pipeline forward.

## Progress Visibility Rule

For any remote GPU, model download, dependency install, or training action:

- State what will run before running it.
- State what may download.
- State the log path.
- Check progress by tailing logs and process/GPU status.
- Do not run black-box long commands.
- Do not apologize and then stop; after a correction, continue with concrete progress.

## 2026-06-22 Local AlphaMaze Cache Upload

User rejected slow remote HuggingFace download. Current approach is local cache upload with visible progress:

- Local model file: `C:\Users\gjy10\.cache\huggingface\hub\models--Menlo--AlphaMaze-v0.2-1.5B\snapshots\2a7c08f3614fa672bb1be573b210b3b5e575cd30\model.safetensors`
- Remote target: `/root/.cache/huggingface/hub/models--Menlo--AlphaMaze-v0.2-1.5B/snapshots/2a7c08f3614fa672bb1be573b210b3b5e575cd30/model.safetensors`
- Upload script: `artifacts/stream_upload_alphamaze_model.py`
- Progress log: `artifacts/upload_alphamaze_tail_stream.log`
- PID file: `artifacts/upload_alphamaze_tail_stream.pid`
- Started by resuming from the remote file size, so rerunning the script continues rather than restarting.

## 2026-06-23 AGRL-Maze Full Implementation

Implemented AGRL route from `D:\下载\AGRL-Maze Documentation.docx` as a separate pipeline in `maze_gpt_agent/agrl_core.py` and scripts:

- `scripts/agrl_generate_dataset.py`
- `scripts/agrl_train_qlearning.py`
- `scripts/agrl_evaluate.py`
- `docs/AGRL_IMPLEMENTATION.md`

Important rule from user: final AI player sees only current 3x3 view, but has memory of previously observed cells. Unknown cells are hidden and cannot be used as known targets. Environment keeps full map only for execution checks.

Current formal run:

- Dataset: `artifacts/agrl/train.json`, `val.json`, `test.json`, split 160/40/40, size 11.
- Q table: `artifacts/agrl/q_table.json`, 2851 Q states.
- Validation success rate: 0.90.
- Test results in `artifacts/agrl/evaluation_summary.json`.
- Demo visualization: `artifacts/agrl/demo_run.html`.
- Remote old AlphaMaze LoRA training was stopped; GPU status after stop was `7 MiB, 0 %`.
