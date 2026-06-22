# AGRL-Maze Implementation Notes

This project now implements the course design route described in `AGRL-Maze Documentation.docx`:

`maze generation -> legality check -> expert strategy -> Q-learning high-level policy -> strategy evaluation -> visualization`.

## Vision Rule

The evaluated AI player never receives the full maze as observation.

- The environment stores the full maze only for collision, reward, trap, coin, BOSS and exit checks.
- The player observes only the current 3x3 window.
- Cells seen before are stored in `PlayerState.known`.
- Unknown cells are hidden and are not usable for PCTSP/RCSPP target selection.
- Exploration uses frontier cells: known passable cells adjacent to unknown cells.

## Implemented Modules

- `maze_gpt_agent/agrl_core.py`
  - `MazeSample`, `PlayerState`, `Target`, `PathResult`, `BossResult`, `ExpertSolution`.
  - DFS-backtracking maze generator with S/E/B, main-route coins, branch coins, bait coins, light traps and required traps.
  - Legality checks for S, E, B, S->B, B->E and coin availability.
  - 3x3 greedy local resource scorer.
  - PCTSP-style known-gold target scoring.
  - RCSPP-style memory-constrained path search with trap loss.
  - BOSS branch-and-bound skill search.
  - Q-learning over high-level actions.
  - Four strategies: `shortest`, `greedy3x3`, `classic`, `rl`.

- `scripts/agrl_generate_dataset.py`
  - Generates separated train/val/test JSON datasets.
  - Saves BOSS config, skill config and expert solution.

- `scripts/agrl_train_qlearning.py`
  - Trains high-level Q-learning policy.
  - Saves Q table and validation curve summary.

- `scripts/agrl_evaluate.py`
  - Evaluates strategy comparison metrics.
  - Generates HTML process visualization.

## Current Formal Run

Command sequence:

```powershell
python scripts\agrl_generate_dataset.py --train 160 --val 40 --test 40 --size 11 --seed 42 --out-dir artifacts\agrl
python scripts\agrl_train_qlearning.py --train artifacts\agrl\train.json --val artifacts\agrl\val.json --episodes 1200 --out artifacts\agrl\q_table.json --curve artifacts\agrl\qlearning_curve.json
python scripts\agrl_evaluate.py --test artifacts\agrl\test.json --q-table artifacts\agrl\q_table.json --out artifacts\agrl\evaluation_summary.json --html artifacts\agrl\demo_run.html
```

Validation after training:

- Q states: 2851
- Validation success rate: 0.90
- Validation average resource/step score: 3.8951
- Validation average steps: 47.8

Independent test set, 40 mazes:

| Strategy | Success | Boss Success | Avg Resource | Avg Steps | Avg Resource/Step |
|---|---:|---:|---:|---:|---:|
| classic | 0.925 | 0.925 | 198.25 | 44.00 | 4.4096 |
| greedy3x3 | 0.925 | 0.925 | 198.25 | 44.00 | 4.4096 |
| rl | 0.900 | 0.925 | 200.75 | 47.80 | 4.0988 |
| shortest | 0.875 | 0.875 | 179.25 | 43.70 | 3.8706 |

## Output Artifacts

- Dataset manifest: `artifacts/agrl/manifest.json`
- Train split: `artifacts/agrl/train.json`
- Validation split: `artifacts/agrl/val.json`
- Test split: `artifacts/agrl/test.json`
- Q table: `artifacts/agrl/q_table.json`
- Training curve summary: `artifacts/agrl/qlearning_curve.json`
- Evaluation summary: `artifacts/agrl/evaluation_summary.json`
- Demo visualization: `artifacts/agrl/demo_run.html`
