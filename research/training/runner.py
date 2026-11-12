"""Bounded SFT and on-policy GRPO. Never use a teacher to generate RL actions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .policy import canonical, parse_action, sft_tokens, sha256_file, prompt_version
from .model import generate, load_model, read_config, token_logps


@dataclass
class Deadline:
    seconds: int
    started: float = field(default_factory=time.monotonic)
    stopped: bool = False

    def install(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopped", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stopped", True))

    def check(self):
        if self.stopped or time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("training wall-clock limit reached")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def runtime_versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "trl", "unsloth", "unsloth_zoo"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def save_checkpoint(model, tokenizer, output: Path, config: dict, stage: str, steps: int, dataset_hash: str,
                    extra: dict | None = None):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    files = {p.name: sha256_file(p) for p in output.glob("adapter*") if p.is_file()}
    checkpoint_id = f"{stage}-" + hashlib.sha256(canonical(files).encode()).hexdigest()[:16]
    manifest = {"checkpoint_id": checkpoint_id, "policy_id": stage, "stage": stage,
        "model": config["model"], "revision": config["revision"], "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(), "completed_updates": steps,
        "dataset_hash": dataset_hash, "prompt_version": prompt_version(config),
        "thinking": False, "precision": "bfloat16", "vision_frozen": True,
        "runtime_versions": runtime_versions(), "artifact_hashes": files,
        "measured": True, **(extra or {})}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def rollout(model, tokenizer, scenario, config, deadline=None):
    from research.environment import ResearchEnvironment
    env = ResearchEnvironment()
    state = env.reset(scenario, seed=config["seed"])
    decisions, events = [], []
    for _ in range(scenario.get("max_steps", 4) + 1):
        if deadline:
            deadline.check()
        decision, usage = generate(model, tokenizer, state, config, sample=True)
        decisions.append(decision)
        try:
            action = parse_action(decision.text)
        except (ValueError, TypeError, json.JSONDecodeError):
            action = {"action": "invalid", "arguments": {}}
        state, reward, done, info = env.step(action)
        events.append({"action": action, "raw_output": decision.text, "usage": usage, "info": info})
        if done:
            return decisions, reward, {"events": events, "truncated": False, "score": env.score()}
    # A missing model-authored finish is a failure. Do not auto-finish on its behalf.
    return decisions, -4.0, {"events": events, "truncated": True, "score": {"terminal": False}}


def group_advantages(rewards):
    import torch
    values = torch.tensor(rewards, dtype=torch.float32)
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-4)


def clipped_objective(logps, old_logps, advantage, clip=0.2):
    import torch
    ratio = (logps - old_logps).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()


def sft(model, tokenizer, rows, config, deadline, callback):
    import torch
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["sft_learning_rate"])
    rng = random.Random(config["seed"])
    rows = list(rows)
    if not rows:
        raise ValueError("SFT training set is empty")
    if not config.get("sft_preserve_stratified_order", False):
        rng.shuffle(rows)
    model.train()
    for step in range(config["sft_steps"]):
        deadline.check()
        row = rows[step % len(rows)]
        state = row.get("state", row.get("observation"))
        decision = sft_tokens(tokenizer, state, row["action"], config["max_context_tokens"], config["max_new_tokens"], prompt_version(config))
        optimizer.zero_grad(set_to_none=True)
        loss = -token_logps(model, decision).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite SFT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        callback(step + 1, {"loss": float(loss.detach()), "completion_tokens": len(decision.completion)})


def grpo(model, tokenizer, scenarios, config, deadline, callback):
    import torch
    if config["group_size"] < 2 or config.get("grpo_beta", 0) != 0:
        raise ValueError("This bounded GRPO implementation needs group_size>=2 and beta=0")
    if not scenarios:
        raise ValueError("GRPO training scenarios are empty")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["rl_learning_rate"])
    rng = random.Random(config["seed"])
    for step in range(config["rl_steps"]):
        deadline.check()
        schedule = config.get("rl_scenario_schedule")
        if schedule:
            by_id = {item["id"]: item for item in scenarios}
            scenario = by_id[schedule[step % len(schedule)]]
        else:
            scenario = rng.choice(scenarios)
        # All group members reset the exact same scenario. The current student generates every token.
        group = [rollout(model, tokenizer, scenario, config, deadline) for _ in range(config["group_size"])]
        advantages = group_advantages([item[1] for item in group])
        model.eval()  # disable dropout in both behavior and updated policy likelihoods
        with torch.no_grad():
            old = [[token_logps(model, d).detach() for d in item[0]] for item in group]
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for index, (decisions, reward, record) in enumerate(group):
            token_count = sum(len(d.completion) for d in decisions)
            for decision, old_logps in zip(decisions, old[index]):
                deadline.check()
                current = token_logps(model, decision)
                loss = clipped_objective(current, old_logps, advantages[index].to(current.device), config["grpo_clip"])
                # Normalize by generated tokens within each episode, then by group size.
                loss = loss * len(decision.completion) / max(1, token_count) / len(group)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite GRPO loss")
                loss.backward()
                loss_total += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        updated = bool(float(advantages.abs().sum()) > 0 and float(grad_norm) > 0)
        if updated:
            optimizer.step()
        callback(step + 1, {"loss": loss_total, "grad_norm": float(grad_norm), "optimizer_updated": updated,
            "group_rewards": [x[1] for x in group], "group_advantages": advantages.tolist(),
            "scenario_id": scenario["id"], "rollouts": [x[2] for x in group],
            "zero_variance_group": float(advantages.abs().sum()) == 0})


def normalize_sft_row(row):
    if "prompt" in row and "completion" in row:
        from research.data.convert_sft import PROMPT_VERSION as DATA_PROMPT_VERSION
        metadata = row["metadata"]
        if metadata["prompt_version"] != DATA_PROMPT_VERSION:
            raise ValueError("SFT prompt version differs from installed data converter")
        users = [message["content"] for message in row["prompt"] if message["role"] == "user"]
        completions = row["completion"]
        if len(users) != 1 or len(completions) != 1 or completions[0]["role"] != "assistant":
            raise ValueError("Expected one state and one assistant action per SFT record")
        return {"state": json.loads(users[0]), "action": json.loads(completions[0]["content"]), **metadata}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sft", "rl"])
    parser.add_argument("--data", required=True, help="SFT state/action JSONL or scenario JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", help="Required saved SFT adapter for RL")
    parser.add_argument("--config")
    parser.add_argument("--smoke-report", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    smoke = json.loads(Path(args.smoke_report).read_text())
    if not smoke.get("passed") or smoke.get("model") != config["model"] or smoke.get("revision") != config["revision"]:
        raise SystemExit("A successful smoke report for this exact model/revision is required")
    if smoke.get("runtime_versions") != runtime_versions() or smoke.get("config") != config:
        raise SystemExit("Smoke runtime/config differs from this run; rerun smoke")
    if args.stage == "rl" and not args.adapter:
        raise SystemExit("RL must start from a saved SFT adapter")
    if args.adapter:
        parent = json.loads((Path(args.adapter) / "manifest.json").read_text())
        if parent["policy_id"] != "sft" or parent["completed_updates"] <= 0:
            raise SystemExit("RL requires a measured SFT checkpoint")
    rows = [json.loads(s) for s in Path(args.data).read_text().splitlines() if s.strip()]
    if args.stage == "sft":
        rows = [normalize_sft_row(row) for row in rows]
    if any(row.get("split") != "train" for row in rows):
        raise SystemExit("Input must contain train split only; refuse validation/test training")
    output = Path(args.output)
    if (output / "manifest.json").exists():
        raise SystemExit("Choose a new output directory; never overwrite a checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    deadline = Deadline(config["max_train_seconds"])
    deadline.install()
    model, tokenizer = load_model(config, args.adapter, training=True)
    data_hash = sha256_file(args.data)
    completed, last = 0, {}
    def callback(step, metrics):
        nonlocal completed, last
        completed += int(metrics.get("optimizer_updated", True))
        last = metrics
        with (output / "training.jsonl").open("a") as f:
            f.write(canonical({"step": step, **metrics}) + "\n")
        print(canonical({"step": step, **{k:v for k,v in metrics.items() if k != "rollouts"}}), flush=True)
        if step % 10 == 0 and completed:
            save_checkpoint(model, tokenizer, output / f"step-{step:04d}", config, args.stage, completed, data_hash)
    stopped = False
    try:
        (sft if args.stage == "sft" else grpo)(model, tokenizer, rows, config, deadline, callback)
    except TimeoutError:
        stopped = True
    finally:
        # SIGTERM and deadlines preserve a checkpoint even for a bounded partial run.
        if completed:
            manifest = save_checkpoint(model, tokenizer, output, config, args.stage, completed, data_hash,
                {"stopped_at_deadline": stopped, "parent_checkpoint": args.adapter,
                 "algorithm": "sft" if args.stage == "sft" else "on_policy_grpo_beta0", "last_metrics": {k:v for k,v in last.items() if k != "rollouts"}})
            print(canonical(manifest), flush=True)
    if not completed:
        raise SystemExit("No optimizer updates completed; no trained checkpoint claimed")


if __name__ == "__main__":
    main()
