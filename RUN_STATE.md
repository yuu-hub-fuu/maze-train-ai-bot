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

## 2026-06-23 RL/DQN and LoRA Fix Run

The user correctly pointed out that the RL training path had real bugs, not just weak training.

Fixed issues:

- Large mixed dataset was not filtered by expert success, so DQN/Q-learning learned from bad or infeasible mazes.
- Valid dataset generator used `len(rows)` to pick difficulty. It could get stuck forever on the 95th kept sample because that index mapped to `Extreme`; fixed to rotate by attempts.
- DQN exploration sampled uniformly from all high-level actions, including illegal or currently impossible actions. Added valid-action mask for behavior policy, target-Q bootstrap, and inference.
- DQN was trained with high random exploration and no curriculum. Added classic-teacher guidance that anneals to zero by the later episodes.
- AlphaMaze LoRA `nan` was a separate bug: labels could become all `-100` when long prompts truncated away the answer. Fixed truncation to always preserve answer tokens, changed fp16 to bf16/float32, lowered LR to `5e-5`, and added non-finite/empty-label checks.

Remote environment actually used:

- Conda env: `maze-agent`
- Torch: `2.3.1+cu118`
- GPU: RTX 3090
- Remote project: `/hy-tmp/maze-train-ai-bot`

Valid dataset:

- Remote and local path: `artifacts/agrl_large_valid`
- Split: train 1000, val 200, test 200
- Generator algorithms: mixed DFS, Prim, Kruskal, recursive division
- Filter: keep only mazes where `classic` expert succeeds under 3x3 plus memory observation
- Generation attempts: train 1615, val 310, test 321

Q-learning baseline on valid dataset:

- `q_states`: 10594
- validation success: 0.965
- validation avg score: 4.0895
- validation avg steps: 48.66

DQN run:

- Remote log: `/hy-tmp/maze-train-ai-bot/logs/train_dqn_valid.log`
- Local policy: `artifacts/agrl_large_valid/dqn_policy.pt`
- Local metrics: `artifacts/agrl_large_valid/dqn_metrics.json`
- Episodes: 3000
- Device recorded by trainer: `cuda`
- Last 100 training success: 1.0
- Last 100 average reward: 766.17
- Teacher ratio was 0 after episode 1950 and success stayed near 1.0.

Test evaluation on 200 valid test mazes:

- DQN: success 1.0, boss 1.0, avg resource 190.8, avg steps 47.86, avg score 4.4172
- Classic: success 1.0, avg resource 190.2, avg steps 45.77, avg score 4.5673
- Greedy3x3: success 1.0, avg resource 190.2, avg steps 45.77, avg score 4.5673
- Q-learning: success 0.98, avg resource 191.25, avg steps 47.06, avg score 4.4699
- Shortest: success 0.885, avg resource 163.9, avg steps 44.34, avg score 3.6766

Evaluation artifacts:

- `artifacts/agrl_large_valid/evaluation_summary_v2.json`
- `artifacts/agrl_large_valid/demo_dqn.html`

AlphaMaze LoRA valid run:

- Valid SFT export: `artifacts/agrl_large_valid/hf_sft_messages.jsonl`, 34371 records
- Remote log: `/hy-tmp/maze-train-ai-bot/logs/train_alphamaze_agrl_lora_valid.log`
- Local log: `logs/train_alphamaze_agrl_lora_valid.log`
- Base: `Menlo/AlphaMaze-v0.2-1.5B`
- Local adapter: `artifacts/agrl_large_valid/alphamaze_agrl_lora`
- 30-step finite-loss check passed: losses at step 10/20/30 were finite.
- 300-step run passed without `nan`; loss decreased from about 6.15 to 0.37.

Important conclusion:

- Bad expert filtering and unmasked random DQN actions caused the poor RL result.
- They did not directly cause AlphaMaze LoRA `nan`.
- The direct LoRA `nan` risk was empty supervised labels after truncating away the assistant answer, plus fp16/high-LR instability.

## 2026-06-23 15x15 Generalization Test

Purpose: test the current DQN policy trained on 11x11 valid mazes without retraining on 15x15 mazes.

Important correction: an initial helper run accidentally generated an 11x11 temporary test set into `artifacts/agrl_large_valid`; the local official 11x11 split/manifest was immediately restored to the server afterward. The actual 15x15 test uses a separate directory.

15x15 dataset:

- Path: `artifacts/agrl_15_valid`
- Remote path: `/hy-tmp/maze-train-ai-bot/artifacts/agrl_15_valid`
- Split: test 100, train 0, val 0
- Size: 15
- Seed: 15015
- Generator: mixed DFS/Prim/Kruskal/division
- Filter: classic expert success only
- Attempts: 157 to keep 100 valid test mazes

15x15 test results with current 11x11-trained DQN:

- Classic: success 1.0, avg resource 186.8, avg steps 91.38, avg score 2.2315
- Greedy3x3: success 1.0, avg resource 186.8, avg steps 91.38, avg score 2.2315
- DQN: success 1.0, avg resource 186.7, avg steps 95.14, avg score 2.1660
- Q-learning: success 0.96, avg resource 186.8, avg steps 92.94, avg score 2.1870
- Shortest: success 0.93, avg resource 175.0, avg steps 94.70, avg score 1.8884

Conclusion: the 11x11-trained DQN generalizes to 15x15 in success and BOSS clear rate on expert-filtered valid maps, but it is not score-optimal. It walks about 3.76 more steps than classic/greedy on average and therefore loses resource/step score.

Artifacts:

- `artifacts/agrl_15_valid/test.json`
- `artifacts/agrl_15_valid/evaluation_summary_v2.json`
- `artifacts/agrl_15_valid/demo_dqn.html`

## 2026-06-23 Server-First Correction

User correction: for this project, local validation must not be treated as real progress. The default target is the remote GPU server at `/hy-tmp/maze-train-ai-bot` using conda env `maze-agent`. Read this file before acting. Upload code changes, run on the server, write logs under `/hy-tmp/maze-train-ai-bot/logs`, and check/tail those logs. Do not store the SSH password in files.

Current active correction path:

- New generator/teacher direction: randomized unbiased maze generation plus ratio-optimal teacher objective `remaining_resource / steps`.
- New online teacher direction: `safe_ratio_planner` with 3x3 observation plus memory, not full-map online control.
- Remote validation script uploaded/running attempt: `/hy-tmp/maze-train-ai-bot/artifacts/validate_safe_ratio_server.py`.
- Remote validation log path: `/hy-tmp/maze-train-ai-bot/logs/safe_ratio_server_validate.log`.
- Last local wrapper reported `REMOTE_EXIT -1`; next action is to SSH-check the remote process/log, not to continue local validation.

## 2026-06-23 Safe-Ratio 15x15 Server Run

Critical correction implemented after user feedback:

- Old patterned maps and fixed S/E/BOSS assumptions are no longer the main data route.
- New generator uses randomized start, exit, BOSS, coins, traps, and mixed DFS/Prim/Kruskal/division mazes.
- Exact full-map ratio oracle is bounded and optional only; it is too expensive for 15x15 and cannot be the large-scale generation gate.
- Main teacher is now `safe_ratio_planner`: strict 3x3 observation plus remembered cells, online exploration, safe resource-aware pathing, BOSS gate, and final metric `remaining_resource / steps`.
- Fixed a real negative-cycle bug: Dijkstra path costs in `safe_memory_path` cannot use negative coin edge costs; coin value is scored at target selection time.
- DQN action execution was changed to use the same safe planner primitives as the teacher; the old DQN action executor was mismatched with the new teacher and produced weak learning.

Server paths:

- Remote project: `/hy-tmp/maze-train-ai-bot`
- Conda env: `maze-agent`
- Dataset/model path: `/hy-tmp/maze-train-ai-bot/artifacts/agrl_safe_ratio_15`
- Local downloaded path: `artifacts/agrl_safe_ratio_15`

Generated 15x15 online-teacher dataset:

- train: 1000 kept from 1185 attempts
- val: 200 kept from 231 attempts
- test: 200 kept from 244 attempts
- generator: unbiased randomized maze with random S/E/BOSS/coin/trap placement
- teacher: `safe_ratio_planner` under strict 3x3 observation plus memory

Teacher test result on 200 held-out 15x15 maps:

- safe_ratio_planner success_rate: 1.0
- boss_success_rate: 1.0
- avg_remaining_resource: 172.0
- avg_steps: 108.31
- avg_resource_per_step: 1.7267031848156134

Low-level BC result, important negative finding:

- 60 epochs on GPU, final val action acc about 0.616
- closed-loop success_rate: 0.0
- conclusion: naked next-direction imitation is not a viable final model under partial observability because errors compound.

Aligned DQN result:

- Training: 5000 episodes on server GPU, `cuda`, teacher_start 0.5 annealed to 0 after episode 3250.
- Training last100 success: 0.96.
- Test on 200 held-out 15x15 maps using safe-aligned executor:
  - success_rate: 0.975
  - boss_success_rate: 1.0
  - avg_remaining_resource: 179.05
  - avg_steps: 110.575
  - avg_resource_per_step: 1.764622776819408
  - avg_traps: 2.715
  - avg_coins: 5.21

Baseline/negative comparison on same 15x15 test:

- classic/greedy3x3 from old executor: success_rate 0.31, avg_score 1.6449659990039314
- shortest: success_rate 0.235, avg_score 1.1684728939845161
- lowlevel_bc: success_rate 0.0

Downloaded local artifacts:

- `artifacts/agrl_safe_ratio_15/train.json`
- `artifacts/agrl_safe_ratio_15/val.json`
- `artifacts/agrl_safe_ratio_15/test.json`
- `artifacts/agrl_safe_ratio_15/dqn_policy.pt`
- `artifacts/agrl_safe_ratio_15/dqn_metrics.json`
- `artifacts/agrl_safe_ratio_15/evaluation_dqn_safe_aligned.json`
- `artifacts/agrl_safe_ratio_15/evaluation_safe_ratio_teacher.json`
- `artifacts/agrl_safe_ratio_15/evaluation_lowlevel_bc.json`
- logs under `logs/generate_safe_ratio_15.log`, `logs/train_dqn_safe_ratio_15.log`, `logs/eval_dqn_safe_aligned_15.log`.

## 2026-06-23 DQN Failure Diagnosis and Mask Fix

Diagnosis target: the 5 failed maps from `artifacts/agrl_safe_ratio_15/evaluation_dqn_safe_aligned.json` after the 5000-episode DQN run.

Server diagnosis log:

- `/hy-tmp/maze-train-ai-bot/logs/diagnose_dqn_failures_15.log`
- local copy: `logs/diagnose_dqn_failures_15.log`

Root cause:

- The trained DQN itself had learned a mostly working policy, but the legal-action mask was wrong after BOSS was defeated.
- `safe_target_path_from_high_action(..., "GO_BOSS")` still allowed `GO_BOSS` when `state.boss_defeated == True`.
- If the agent was currently standing on the BOSS tile, `safe_memory_path` returned an empty path to the same tile.
- `valid_action_mask` treated that empty-path `GO_BOSS` as legal.
- The DQN repeatedly selected `GO_BOSS`, `step_high_action` returned `empty_path`, state did not change, and evaluation eventually failed by decision cap.

Fix:

- In `maze_gpt_agent/agrl_dqn.py`, `GO_BOSS` is invalid once BOSS is defeated.
- Empty paths are no longer returned as valid high-level targets.

Post-fix result, without retraining the model:

- Same `artifacts/agrl_safe_ratio_15/dqn_policy.pt`
- Same 200 held-out 15x15 test maps
- `dqn_safe_aligned` success_rate: 1.0
- boss_success_rate: 1.0
- avg_remaining_resource: 175.4
- avg_steps: 108.61
- avg_resource_per_step: 1.7546087329031783
- avg_traps: 2.97
- avg_coins: 5.29
- log: `logs/eval_dqn_safe_aligned_15_after_mask_fix.log`
- updated eval JSON: `artifacts/agrl_safe_ratio_15/evaluation_dqn_safe_aligned.json`

## 2026-06-23 DQN HTML Process Demos

Generated and downloaded DQN safe-aligned HTML process visualizations from the server.

Directories:

- `artifacts/agrl_safe_ratio_15/html_dqn_10/index.html`: first 10 test processes, all successful.
- `artifacts/agrl_safe_ratio_15/html_dqn_10_mixed/index.html`: mixed difficulty 10-process set, 3 Easy + 4 Medium + 3 Hard, all successful.

Each HTML page contains frame-by-frame model execution with action/event/resource/steps/score and an interactive play/scrub control.

## 2026-06-23 Oracle-Ratio Dataset and DQN Result

User pointed out that generation must know the optimal path at generation time, and that the previous safe-ratio DQN often did not explore enough. Fixed pipeline:

- Added `scripts/agrl_generate_safe_ratio_dataset.py` oracle-required mode: primary `expert_solution.label_source` is now `bounded_full_map_ratio_oracle` when available.
- Added `scripts/agrl_generate_oracle_ratio_parallel.py` for server-side parallel oracle generation.
- Added oracle-guided teacher warm-start in `maze_gpt_agent/agrl_dqn.py`; DQN training now checks `sample.expert_solution.label_source == bounded_full_map_ratio_oracle` before falling back to the old online teacher.
- Generated on server `/hy-tmp/maze-train-ai-bot` with 24 workers into `artifacts/agrl_oracle_ratio_15`.
- Dataset counts: train 1000, val 200, test 200. All kept rows require bounded full-map ratio oracle. Generation stats: train attempts 1698 / teacher_fail 259 / oracle_fail 368; val attempts 372; test attempts 376.
- DQN training: `scripts/agrl_train_dqn.py --train artifacts/agrl_oracle_ratio_15/train.json --episodes 8000 --teacher-start 0.75`; device `cuda`; output `artifacts/agrl_oracle_ratio_15/dqn_policy.pt`.
- Training curve: recent_success reached 0.958 at episode 7600 and 0.940 at episode 8000, but metrics last100 success was 0.89.
- Safe-aligned held-out test eval (`scripts/agrl_evaluate_dqn_safe_oracle.py`): success_rate 1.0, boss_success_rate 1.0, avg_remaining_resource 169.9, avg_steps 111.89, avg_resource_per_step 1.6577638526027594, avg_traps 2.595, avg_coins 4.955.
- Oracle reference on the same 200 test maps: avg_oracle_score 4.848163792438519, avg_teacher_score 1.6528788485851547, avg_teacher_to_oracle 0.3409288380815602, avg_oracle_path_coverage 0.42410572553543163.
- Important conclusion: map generation is now corrected to know/store the bounded full-map ratio-optimal path, but the current 83-dim-state / 7-high-level-action DQN still behaves near the old online teacher score, not near the oracle score. The remaining bottleneck is model/action-space architecture, not dataset generation.

## 2026-06-23 Model Failure Diagnosis After Oracle Dataset

User asked why the model learned so poorly against full-map oracle labels. Follow-up experiments on the server:

- Low-level oracle BC from `artifacts/agrl_oracle_ratio_15`: val action accuracy only ~0.56; closed-loop success 0.0. Conclusion: flat MLP behavior cloning overfits and fails under compounding error.
- Neural target ranker that scores candidate remembered-map coordinates: val binary acc ~0.85 but closed-loop success 0.625, avg score 1.0688. It preserves target identity better than high-level EXPLORE but still fails closed loop.
- State alias analysis: compact DQN state conflict_step_rate 0.0441; full-memory low-level state conflict_step_rate 0.00537. So aliasing exists but does not fully explain failure; generalization and compounding error are bigger.
- CNN low-level oracle BC fixed action generalization somewhat: val action accuracy ~0.81, but closed-loop success only 0.025. Conclusion: even good one-step oracle imitation collapses over long trajectories.
- Hybrid CNN+DQN safety fallback was worse than DQN alone: thresholds 0.95/0.90/0.85 yielded success 0.805/0.815/0.82 and scores 1.519/1.469/1.414, below DQN safe-aligned score 1.6578 and success 1.0.

Key conclusion:

- Full-map oracle is a useful upper bound and generation audit, but it is not a directly learnable policy target for a strict 3x3 online agent on random hidden resources. The oracle knows hidden empty branches, hidden resources, BOSS, and exit positions; the online model must discover them. Direct BC from full-map oracle creates covariate shift/compounding errors and can reduce success.
- Current best deliverable remains the safe-aligned DQN: success 1.0 and boss 1.0 on 200 held-out 15x15 oracle-ratio maps, score 1.6578, which essentially matches the online safe teacher 1.6529. The poor ratio against oracle 4.8482 is mainly an online observability gap plus architecture/action-space limitation, not just insufficient GPU training.
- Proper next fix should be DRQN/Double-Dueling DQN or a goal-conditioned recurrent value model trained by online rollouts/DAgger-style aggregation, with full-map oracle used only as upper bound/evaluation, not as direct action imitation labels.

## 2026-06-23 Architecture Redesign Attempts After User Rejected Hyperparameter Tuning

User correctly said tuning reward/hyperparameters is less useful than fixing design flaws. Actions taken on server:

1. Stopped the ratio-focused enhanced-DQN hyperparameter run (`train_enhanced_dqn_ratio_15.pid`) after realizing it was just reward tuning.
2. Implemented `maze_gpt_agent/agrl_enhanced_dqn.py`: CNN encoder over remembered map + Dueling Double DQN + expanded high-level actions (`BEST_COIN`, `NEAREST_COIN`, `GO_BOSS`, `GO_EXIT`, `EXPLORE_NEAREST`, `EXPLORE_INFO`, `EXPLORE_FAR`).
   - Training: 3000 episodes, CUDA, log `logs/train_enhanced_dqn_15.log`, model `artifacts/agrl_oracle_ratio_15/enhanced_dqn.pt`.
   - Test result: success_rate 0.995, boss_success_rate 0.995, avg_remaining_resource 175.9, avg_steps 139.505, avg_resource_per_step 1.4956343314073424.
   - Interpretation: stronger exploration and more resource than old DQN, but too many steps, so task score is worse than safe-aligned DQN (1.6578).
3. Implemented `maze_gpt_agent/agrl_waypoint_q.py`: coordinate heatmap / 225-action WaypointQ over remembered reachable cells. This is a real architecture change: model selects target coordinates, not hand-written action categories.
   - Training: 2500 episodes, CUDA, model `artifacts/agrl_oracle_ratio_15/waypoint_q.pt`.
   - Test result: success_rate 0.745, boss_success_rate 0.87, avg_remaining_resource 185.25, avg_steps 504.81, avg_resource_per_step 1.0019634853251094.
   - Interpretation: it learns to collect/explore (highest resource) but lacks termination/option stopping value, causing huge step counts and failed exits.

Current best model for deliverable is still `artifacts/agrl_safe_ratio_15/dqn_policy.pt` or the oracle-set safe-aligned DQN `artifacts/agrl_oracle_ratio_15/dqn_policy.pt` depending which dataset narrative is used:

- `agrl_safe_ratio_15/dqn_policy.pt`: success 1.0, avg_score 1.7546 on its 200 test maps after mask fix.
- `agrl_oracle_ratio_15/dqn_policy.pt`: success 1.0, avg_score 1.6578 on harder oracle-required test maps.

The architecture experiments show the actual design flaw: stronger exploration models can collect more resource but do not learn when to stop and cash out. Next real architecture should be termination-aware options: a model predicts both waypoint value and stop/commit-to-boss-exit value, or uses hierarchical RL with option termination. Do not continue plain oracle BC or reward-only tuning.

## 2026-06-23 Termination-Aware OptionQ Attempt

User asked to continue and specifically criticized hyperparameter tuning over architecture fixes. Implemented and tested a real architecture change:

- `maze_gpt_agent/agrl_option_q.py`: termination-aware OptionQ.
  - Action space: 225 remembered-coordinate waypoint actions plus explicit `CASH_BOSS` and `CASH_EXIT` option heads.
  - Network: CNN map encoder + scalar state encoder + coordinate heatmap head + separate option head.
  - Purpose: fix WaypointQ's design flaw where BOSS/EXIT were ordinary coordinates and the model had no salient cash-out/termination channel.
- Training: server CUDA, 2500 episodes, log `logs/train_option_q_15.log`, model `artifacts/agrl_oracle_ratio_15/option_q.pt`.
- Training result: last100 success 0.94, avg_reward_last_100 10.01.
- Held-out test result: success_rate 0.625, boss_success_rate 0.77, avg_remaining_resource 180.95, avg_steps 705.835, avg_resource_per_step 0.7453.

Conclusion:

- Explicit termination option heads did not solve the coordinate-action instability. OptionQ still over-explores, takes huge step counts, and fails many exits.
- Compared architecture attempts:
  - safe-aligned DQN: success 1.0, score 1.6578 on oracle-required test.
  - CNN Dueling Double DQN with expanded hand actions: success 0.995, score 1.4956; more resource but too many steps.
  - WaypointQ coordinate heatmap: success 0.745, score 1.002; high resource but huge steps.
  - termination-aware OptionQ: success 0.625, score 0.745; explicit cash-out head still unstable.
- Therefore the current best model remains the safe-aligned DQN. The next useful fix is not another coordinate Q model; it should be a conservative policy-improvement layer around the stable DQN, such as a learned/analytic cash-out gate or value-of-information gate, because all freer exploration architectures increase resource but destroy step ratio.

## 2026-06-23 Conservative Ratio Gate Around Stable DQN

User asked to implement the next fix instead of stopping. Implemented `scripts/agrl_evaluate_dqn_ratio_gate.py`, an analytic conservative ratio gate around the stable DQN:

- It does not retrain the model; it evaluates the current DQN action and only overrides coin/explore actions if immediate cashout to BOSS/EXIT has a better projected final resource/steps score.
- Tested margins 0, 0.03, 0.06, 0.10 with and without `force_boss_known` on the 200 oracle-required test maps.

Results:

- `force_boss_known=False`: success 1.0, avg_resource 169.15, avg_steps 111.45, avg_score 1.6558 for all margins. This is basically the same as original oracle-set safe DQN (score 1.6578), slightly lower.
- `force_boss_known=True`: success 0.785, avg_resource 145.3, avg_steps 93.165, avg_score 1.6159. It reduces steps but breaks too many runs.

Conclusion:

- The stable DQN is not mainly losing score because it ignores obvious late cashout. A conservative cashout gate does not improve score.
- Forcing BOSS early is unsafe and hurts success. Do not use this in final system.
- The remaining gap is earlier value-of-information/exploration quality under partial observability, not a late-stage cashout heuristic.

## 2026-06-23 Frontier-Action DQN Architecture Fix

User pushed to stop treating poor performance as hyperparameter noise and change the model design itself. Implemented and trained a real DQN architecture change on the GPU server:

- Added `maze_gpt_agent/agrl_frontier_dqn.py`.
  - The old single `EXPLORE` action is split into explicit model actions: `EXPLORE_NEAREST`, `EXPLORE_INFO_DENSITY`, `EXPLORE_CASHOUT_AWARE`.
  - State vector expands from the compact DQN state to 102 dims by adding frontier candidate features: availability, path length, unknown-neighbor count, path risk, and post-frontier cashout distance for nearest/info-density/cashout-aware frontiers.
  - The model now learns when to choose exploration style instead of having a fixed external heuristic choose every frontier.
- Added `scripts/agrl_train_frontier_dqn.py` and `scripts/agrl_evaluate_frontier_dqn.py`.
- Trained on server `/hy-tmp/maze-train-ai-bot` in conda env `maze-agent`, CUDA confirmed.
  - Training command: `scripts/agrl_train_frontier_dqn.py --train artifacts/agrl_oracle_ratio_15/train.json --out artifacts/agrl_oracle_ratio_15/frontier_dqn.pt --episodes 3000 --teacher-start 0.70 --lr 8e-4 --log-every 100`.
  - Final training metrics: input_dim 102, device cuda, success_last_100 0.96, avg_reward_last_100 583.5035.
  - Action counts show the new model actually used the new exploration heads: `EXPLORE_NEAREST` 72543, `EXPLORE_INFO_DENSITY` 38627, `EXPLORE_CASHOUT_AWARE` 35355.
- Held-out 200-map oracle-required 15x15 test result:
  - success_rate: 1.0
  - boss_success_rate: 1.0
  - avg_remaining_resource: 179.85
  - avg_steps: 108.13
  - avg_resource_per_step: 1.7926313055433778
  - avg_traps: 2.13
  - avg_coins: 4.875
- This is the new best core model on the harder oracle-required test split. It improves over the previous oracle-set stable DQN (`avg_score 1.6578`, success 1.0) without losing success rate, and beats the fixed frontier heuristics where pure info-density scored 1.7568 but failed 1/200 maps.
- Downloaded local artifacts:
  - `artifacts/agrl_oracle_ratio_15/frontier_dqn.pt`
  - `artifacts/agrl_oracle_ratio_15/evaluation_frontier_dqn.json`
  - `logs/train_frontier_dqn_15.log`
  - `logs/eval_frontier_dqn_15.log`
  - `logs/frontier_dqn_job.log`

## 2026-06-23 Frontier Failure HTML and Boss Gate Bug

Rendered the failing pure `info_density` frontier trajectory for sample `test-ratio-prim-Medium-241137`:

- Original failure HTML: `artifacts/agrl_oracle_ratio_15/html_frontier_failure/test-ratio-prim-Medium-241137_info_density_failure.html`
- After GO_BOSS resource gate fix HTML: `artifacts/agrl_oracle_ratio_15/html_frontier_failure/test-ratio-prim-Medium-241137_info_density_after_boss_gate.html`

Diagnosis:

- Original pure info-density failure ended with resource 10, steps 152, boss not cleared. The old executor/mask allowed `GO_BOSS` whenever BOSS was known and undefeated, even if `state.resource < boss_config.revive_cost`; this let the agent walk 22 steps to BOSS with only 10 resource and then fail.
- Fixed `maze_gpt_agent/agrl_dqn.py` so `safe_target_path_from_high_action(..., "GO_BOSS")` returns `None` unless `state.resource >= sample.boss_config.revive_cost`.
- After the gate fix, the same pure info-density policy no longer suicides into BOSS, but it still fails: final resource 10, steps 130, boss not cleared, event `no_safe_target`.
- Deeper reason: info-density over-explores unknown frontier density, takes long detours and 3 traps while collecting only 2 coins. That is exactly why the newer FrontierDQN architecture is better: it learns when to choose nearest/info-density/cashout-aware exploration instead of using one fixed frontier heuristic everywhere.

## 2026-06-23 HTML Metric and Visibility Fix

User pointed out the failure HTML lacked comparison to the best ratio path and that the unseen 3x3 visibility shadow was not obvious enough. Fixed:

- `maze_gpt_agent/visualizer.py`
  - Stats now include `oracle_score`, `score_of_oracle_pct`, and `score_gap_pct` when present.
  - Frame detail table now shows oracle score and percent gap.
  - Unseen cells now have a strong dark overlay (`.cell.unseen::after`), seen memory cells have a lighter overlay, and current 3x3 cells remain bright.
- `scripts/agrl_render_frontier_failure_html.py`
  - Reads `sample.expert_solution.final_score/final_resource/total_steps` as the best ratio route.
  - Adds `oracle_score`, `oracle_gold`, `oracle_steps`, `score_of_oracle_pct`, and `score_gap_pct` to HTML summary.
- Re-rendered and downloaded:
  - `artifacts/agrl_oracle_ratio_15/html_frontier_failure/test-ratio-prim-Medium-241137_info_density_failure.html`
  - `artifacts/agrl_oracle_ratio_15/html_frontier_failure/test-ratio-prim-Medium-241137_info_density_after_boss_gate.html`

For `test-ratio-prim-Medium-241137`, oracle best is resource 250 / steps 45 = score 5.5556. The fixed-gate info-density failure is resource 10 / steps 130 = score 0.0769, so it achieves 1.3846% of oracle and has a 98.6154% gap.
