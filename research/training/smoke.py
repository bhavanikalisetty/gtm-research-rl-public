"""Mandatory real-GPU load, backward, rollout/mask, and adapter round-trip gate."""
from __future__ import annotations
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from .model import load_model, read_config, generate, token_logps
from .policy import sft_tokens, prompt_version
from .runner import atomic_json, runtime_versions, save_checkpoint, rollout, Deadline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, help="train-split scenario JSONL")
    p.add_argument("--report", required=True)
    p.add_argument("--config")
    args = p.parse_args()
    import torch
    from research.environment import ResearchEnvironment, load_scenarios
    from research.environment.rules import decide
    config = read_config(args.config)
    report = {"passed": False, "model": config["model"], "revision": config["revision"],
              "config": config, "runtime_versions": runtime_versions(), "checks": {}}
    try:
        scenario = load_scenarios(args.scenario, split="train")[0]
        state = ResearchEnvironment().reset(scenario)
        model, tokenizer = load_model(config, training=True)
        report["checks"]["load"] = True
        decision = sft_tokens(tokenizer, state, decide(state), config["max_context_tokens"], config["max_new_tokens"], prompt_version(config))
        assert all(x == 0 for x in decision.loss_mask[:len(decision.prompt)])
        assert all(x == 1 for x in decision.loss_mask[len(decision.prompt):])
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=2e-4)
        model.train()
        loss = -token_logps(model, decision).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in trainable)
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, trainable))
        report["checks"]["backward_update"] = True
        report["loss"] = float(loss.detach())
        del optimizer, before, trainable
        decisions, reward, trace = rollout(model, tokenizer, scenario, config, Deadline(600))
        assert decisions and all(len(d.input_ids) == len(d.loss_mask) and sum(d.loss_mask) == len(d.completion) for d in decisions)
        report["checks"]["rollout_and_tool_mask"] = True
        report["rollout"] = trace
        with tempfile.TemporaryDirectory(prefix="gtm-smoke-") as temp:
            save_checkpoint(model, tokenizer, Path(temp), config, "smoke", 1, "smoke-only")
            model.eval()
            with torch.no_grad():
                expected_logps = token_logps(model, decision).detach().cpu()
            expected, _ = generate(model, tokenizer, state, config)
            del model
            torch.cuda.empty_cache()
            model, tokenizer = load_model(config, adapter=temp)
            model.eval()
            with torch.no_grad():
                actual_logps = token_logps(model, decision).detach().cpu()
            torch.testing.assert_close(actual_logps, expected_logps, rtol=.01, atol=.03)
            actual, _ = generate(model, tokenizer, state, config)
            assert expected.completion == actual.completion, "greedy round-trip changed tokens"
            report["checks"]["checkpoint_round_trip"] = True
        report["gpu"] = torch.cuda.get_device_name(0)
        report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        report["passed"] = all(report["checks"].values()) and len(report["checks"]) == 4
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(Path(args.report), report)
        if report["passed"]:
            lock = subprocess.check_output([__import__('sys').executable, "-m", "pip", "freeze"], text=True)
            Path(args.report).with_name("runtime-lock.txt").write_text(lock)

if __name__ == "__main__":
    main()
