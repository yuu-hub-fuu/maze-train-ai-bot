from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maze_gpt_agent.agrl_core import (  # noqa: E402
    BossResult,
    HIGH_LEVEL_ACTIONS,
    PlayerState,
    RunResult,
    aggregate_results,
    apply_move,
    frame,
    load_samples,
    observe_3x3,
    rcspp_path,
    run_strategy,
    save_json,
    solve_boss_battle,
    target_from_high_action,
    tile_event,
)
from maze_gpt_agent.visualizer import render_run_html  # noqa: E402
from scripts.agrl_export_hf_sft import prompt  # noqa: E402


def parse_action(text: str) -> str | None:
    cleaned = text.strip().splitlines()[0].strip() if text.strip() else ""
    cleaned = cleaned.replace("`", "").replace(".", "").replace(":", "").strip()
    for action in HIGH_LEVEL_ACTIONS:
        if cleaned == action or cleaned.startswith(action):
            return action
    upper = cleaned.upper()
    for action in HIGH_LEVEL_ACTIONS:
        if action in upper:
            return action
    return None


class LoraPolicy:
    def __init__(self, base_model: str, adapter: str, max_new_tokens: int = 8):
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        print(f"lora_eval_device={self.device} dtype={dtype}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.to(self.device)
        self.model.eval()

    def choose(self, sample, state: PlayerState) -> tuple[str | None, str]:
        text = "User:\n" + prompt(sample, state) + "\nAssistant:\n"
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(gen, skip_special_tokens=True)
        return parse_action(decoded), decoded


def run_lora_strategy(sample, policy: LoraPolicy, max_steps: int | None = None) -> tuple[RunResult, dict]:
    started = time.perf_counter()
    state = PlayerState(position=sample.start, path_history=[sample.start])
    observe_3x3(sample, state)
    frames = [frame(sample, state, "START", "start", None)]
    max_steps = max_steps or sample.rows * sample.cols * 4
    boss_result = BossResult(False, 0, [], False, state.resource)
    invalid_outputs = 0
    infeasible_actions = 0
    raw_outputs: list[str] = []
    while state.alive and not state.done and state.steps < max_steps:
        high_action, raw = policy.choose(sample, state)
        raw_outputs.append(raw)
        if high_action is None:
            invalid_outputs += 1
            state.done = True
            state.alive = False
            break
        target = target_from_high_action(sample, state, high_action)
        if target is None:
            infeasible_actions += 1
            state.done = True
            state.alive = False
            break
        path = rcspp_path(sample, state, target.position, require_boss_resource=(target.target_type == "boss"))
        if not path.feasible or not path.actions:
            infeasible_actions += 1
            state.done = True
            state.alive = False
            break
        state.decision_history.append(
            {
                "action": high_action,
                "reason": "alphamaze_lora_policy",
                "raw_output": raw,
                "target": asdict(target),
                "path_score": path.score,
            }
        )
        for action, pos in zip(path.actions, path.path[1:]):
            apply_move(sample, state, pos)
            frames.append(frame(sample, state, action, tile_event(sample, state, pos), target))
            if state.steps >= max_steps:
                break
        if state.position == sample.boss and not state.boss_defeated:
            boss_result = solve_boss_battle(sample.boss_config, sample.skill_config, state.resource)
            if boss_result.success and state.resource >= sample.boss_config.revive_cost:
                state.boss_defeated = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_clear:" + ",".join(boss_result.skill_sequence), target))
            else:
                state.alive = False
                state.done = True
                frames.append(frame(sample, state, "BOSS_FIGHT", "boss_fail", target))
        if state.position == sample.end:
            state.done = True
            if not state.boss_defeated:
                state.alive = False
    success = state.alive and state.done and state.position == sample.end and state.boss_defeated
    result = RunResult(
        strategy="alphamaze_lora",
        sample_id=sample.sample_id,
        difficulty=sample.difficulty,
        success=success,
        boss_success=state.boss_defeated,
        final_resource=state.resource,
        total_steps=state.steps,
        final_score=state.resource / max(1, state.steps),
        trap_count=len(state.triggered_traps),
        coin_count=len(state.collected_coins),
        boss_rounds=boss_result.min_rounds,
        runtime_ms=(time.perf_counter() - started) * 1000,
        frames=frames,
    )
    extra = {
        "invalid_outputs": invalid_outputs,
        "infeasible_actions": infeasible_actions,
        "raw_outputs_head": raw_outputs[:5],
    }
    return result, extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AlphaMaze LoRA as a closed-loop AGRL policy.")
    parser.add_argument("--test", default="artifacts/agrl_large_valid/test.json")
    parser.add_argument("--base-model", default="Menlo/AlphaMaze-v0.2-1.5B")
    parser.add_argument("--adapter", default="artifacts/agrl_large_valid/alphamaze_agrl_lora")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", default="artifacts/agrl_large_valid/evaluation_lora.json")
    parser.add_argument("--html", default="artifacts/agrl_large_valid/demo_lora.html")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_samples(args.test)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]
    policy = LoraPolicy(args.base_model, args.adapter)
    results = []
    extras = []
    demo = None
    for idx, sample in enumerate(samples, 1):
        if args.include_baselines:
            for strategy in ["shortest", "greedy3x3", "classic"]:
                results.append(run_strategy(sample, strategy))
        result, extra = run_lora_strategy(sample, policy)
        results.append(result)
        extras.append({"sample_id": sample.sample_id, **extra})
        if demo is None:
            demo = result
        if idx % 5 == 0 or idx == len(samples):
            lora_results = [r for r in results if r.strategy == "alphamaze_lora"]
            success = sum(r.success for r in lora_results) / max(1, len(lora_results))
            invalid = sum(x["invalid_outputs"] for x in extras)
            infeasible = sum(x["infeasible_actions"] for x in extras)
            print(f"eval_lora {idx}/{len(samples)} success={success:.3f} invalid={invalid} infeasible={infeasible}", flush=True)
    summary = {
        "aggregate": aggregate_results(results),
        "runs": [r.summary() for r in results],
        "extras": extras,
        "limit": len(samples),
        "adapter": args.adapter,
        "base_model": args.base_model,
        "vision_rule": "closed loop; prompt contains only current 3x3 plus remembered cells; unknown cells hidden as ?",
    }
    save_json(args.out, summary)
    if demo is not None:
        render_run_html(args.html, "AGRL-Maze AlphaMaze LoRA Closed Loop Demo", demo.frames, demo.summary())
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {args.out}", flush=True)
    print(f"demo_html: {args.html}", flush=True)


if __name__ == "__main__":
    main()
