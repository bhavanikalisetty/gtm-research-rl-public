"""Evaluate rules or an HTTP policy using identical scenario budgets and evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.request
import math
from datetime import datetime, timezone
from pathlib import Path

from research.data.convert_sft import PROMPT_VERSION
from research.environment import ResearchEnvironment, dataset_hash, load_scenarios
from research.environment.environment import VERSION, canonical
from research.environment.rules import decide


def environment_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "environment"
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in ("environment.py", "rules.py"))).hexdigest()


def http_policy(endpoint: str, policy_id: str, timeout: float, token: str | None = None):
    def policy(state):
        payload = canonical({"policy_id": policy_id, "state": state}).encode()
        headers = {"Content-Type": "application/json"}
        secret = token or os.environ.get("POLICY_API_TOKEN") or os.environ.get("POLICY_TOKEN")
        if secret:
            headers["Authorization"] = "Bearer " + secret
        request = urllib.request.Request(endpoint, data=payload, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if result.get("policy_id") != policy_id:
            raise ValueError("Response policy ID does not match requested policy")
        if policy_id in {"sft", "rl"} and not result.get("checkpoint_id"):
            raise ValueError("Learned policy response is missing its checkpoint ID")
        return result
    return policy


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(scenarios: list[dict], policy=decide, policy_id="rules", metadata=None) -> dict:
    if not scenarios:
        raise ValueError("No scenarios selected")
    metadata = metadata or {}
    if len({s.get("evaluation_split", s["split"]) for s in scenarios}) != 1:
        raise ValueError("Evaluate one split at a time")
    model_policy = policy_id in {"base", "sft", "rl"}
    if model_policy and any(not metadata.get(key) for key in ("model", "revision", "template_version", "thinking_mode", "decoding")):
        raise ValueError("Actual model, revision, template, thinking mode and decoding metadata are required")
    cases = []
    checkpoints = set()
    for scenario in scenarios:
        env = ResearchEnvironment(scenario)
        started = time.perf_counter()
        actions, usage, errors = [], [], []
        for _ in range(5):
            try:
                action = policy(env.state())
                if not isinstance(action, dict):
                    raise ValueError("Policy returned a non-object")
                if model_policy and action.get("policy_id") != policy_id:
                    raise ValueError("Policy ID mismatch")
                if model_policy and not action.get("checkpoint_id"):
                    raise ValueError("Model response lacks a checkpoint ID")
                if model_policy and metadata.get("checkpoint_id") and action["checkpoint_id"] != metadata["checkpoint_id"]:
                    raise ValueError("Checkpoint ID mismatch")
                if model_policy:
                    checkpoints.add(action["checkpoint_id"])
                    if len(checkpoints) > 1:
                        raise ValueError("Multiple checkpoints returned within one evaluation")
            except Exception as exc:
                # No provider response bodies, credentials or tracebacks in artifacts.
                errors.append(type(exc).__name__)
                action = {"action": "invalid", "arguments": {}}
            actions.append(action)
            raw_usage = action.get("usage", {})
            clean_usage = {}
            if isinstance(raw_usage, dict):
                for key in ("input_tokens", "output_tokens", "actual_cost_usd"):
                    value = raw_usage.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                        clean_usage[key] = value
            usage.append(clean_usage)
            _, _, done, _ = env.step(action)
            if done:
                break
        score = env.score()
        actual_costs = [u.get("actual_cost_usd") for u in usage]
        actual_cost = (0.0 if policy_id == "rules" else
                       sum(actual_costs) if actual_costs and all(isinstance(x, (int, float)) for x in actual_costs) else None)
        cases.append({"case_id": scenario["id"], "group_id": scenario["group_id"], "split": scenario["split"],
                      "provenance": scenario["provenance"], "terminal": env.terminal, "actions": actions,
                      "expected_disposition": scenario["hidden_reference"]["disposition"], "score": score,
                      "research_calls": env.calls, "actual_cost_usd": actual_cost,
                      "simulated_cost_usd": round(env.simulated_cost_usd, 6),
                      "simulated_tool_latency_ms": env.simulated_latency_ms,
                      "wall_clock_latency_ms": (time.perf_counter() - started) * 1000,
                      "usage": {"input_tokens": sum(u.get("input_tokens", 0) or 0 for u in usage),
                                "output_tokens": sum(u.get("output_tokens", 0) or 0 for u in usage)},
                      "errors": errors})
    count = len(cases)
    predicted = [c for c in cases if c["terminal"]["disposition"] == "qualified"]
    positive = [c for c in cases if c["expected_disposition"] == "qualified"]
    supported_tp = [c for c in predicted if c["score"]["decision_correct"] and c["score"]["evidence_supported"]]
    decision_tp = [c for c in predicted if c["score"]["decision_correct"]]
    simulated_total = sum(c["simulated_cost_usd"] for c in cases)
    actual_total = sum(c["actual_cost_usd"] for c in cases) if all(c["actual_cost_usd"] is not None for c in cases) else None
    per_class = {}
    for disposition in ("qualified", "rejected", "unresolved"):
        selected = [c for c in cases if c["expected_disposition"] == disposition]
        per_class[disposition] = {"case_count": len(selected), "decision_accuracy": ratio(sum(c["score"]["decision_correct"] for c in selected), len(selected))}
    metrics = {
        "decision_accuracy": sum(c["score"]["decision_correct"] for c in cases) / count,
        "qualification_precision": ratio(len(supported_tp), len(predicted)),
        "qualification_recall": ratio(len(supported_tp), len(positive)),
        "decision_qualification_precision": ratio(len(decision_tp), len(predicted)),
        "decision_qualification_recall": ratio(len(decision_tp), len(positive)),
        "unsupported_claim_rate": ratio(sum(c["score"]["unsupported_field_count"] for c in cases), sum(c["score"]["asserted_field_count"] for c in cases)),
        "unsupported_case_rate": sum(bool(c["score"]["unsupported_field_count"] or c["score"]["invented_evidence_count"]) for c in cases) / count,
        "citation_support_rate": sum(c["score"]["citation_support"] for c in cases) / count,
        "field_accuracy": statistics.mean(c["score"]["field_accuracy"] for c in cases),
        "unresolved_rate": sum(c["terminal"]["disposition"] == "unresolved" for c in cases) / count,
        "false_positives": len(predicted) - len(decision_tp), "false_negatives": len(positive) - len(decision_tp),
        "supported_true_positives": len(supported_tp), "mean_tool_calls": statistics.mean(c["research_calls"] for c in cases),
        "total_actual_cost_usd": actual_total, "total_simulated_cost_usd": round(simulated_total, 6),
        "simulated_cost_per_supported_qualified_lead_usd": ratio(simulated_total, len(supported_tp)),
        "actual_cost_per_supported_qualified_lead_usd": ratio(actual_total, len(supported_tp)) if actual_total is not None else None,
        "median_latency_ms": statistics.median(c["wall_clock_latency_ms"] for c in cases),
        "mean_reward": statistics.mean(c["score"]["reward"] for c in cases), "per_class": per_class,
    }
    fixture = all(s["provenance"] == "synthetic" for s in scenarios)
    now = datetime.now(timezone.utc).isoformat()
    return {"id": "eval_" + hashlib.sha256((now + policy_id).encode()).hexdigest()[:12], "policy_id": policy_id,
            "checkpoint_id": metadata.get("checkpoint_id") or (next(iter(checkpoints)) if len(checkpoints) == 1 else None), "model": metadata.get("model"), "revision": metadata.get("revision"),
            "prompt_version": PROMPT_VERSION, "template_version": metadata.get("template_version"),
            "thinking_mode": metadata.get("thinking_mode", "not_applicable" if policy_id == "rules" else "unspecified"),
            "decoding": metadata.get("decoding", {}), "environment_version": VERSION, "environment_hash": environment_hash(),
            "dataset_hash": dataset_hash(scenarios), "split": scenarios[0].get("evaluation_split", scenarios[0]["split"]), "case_count": count,
            "case_ids": [s["id"] for s in scenarios], "provenance": "fixture" if fixture else "measured",
            "execution_mode": "deterministic_recorded_tools", "metrics": metrics, "created_at": now,
            "cost_basis": {"tool_prices": "simulated fixed scenario prices; no live vendor calls",
                           "actual": "Local rules incurred no external API bill; local compute is unpriced." if policy_id == "rules" else "Reported policy API usage only; null means unavailable.",
                           "training_cost_included": False},
            "limitations": ["Controlled fixture template overlap limits generalization claims." if fixture else "Public evidence is a recorded snapshot, not a live crawl during evaluation.",
                            "Structured fields and citations are scored; freeform reason text is not independently fact-checked.",
                            "This run measures research decisions, not sales conversion or meeting rate."],
            "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test", "live_transfer"])
    parser.add_argument("--policy", default="rules")
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--allow-test", action="store_true", help="Acknowledge test is for final frozen checkpoint comparison only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test is frozen. Select on validation, then use --allow-test for the final comparison.")
    if args.policy != "rules" and not args.endpoint:
        parser.error("A non-rules policy requires --endpoint")
    if args.policy in {"base", "sft", "rl"} and not args.metadata:
        parser.error("An actual model evaluation requires --metadata with pinned model and decoding settings")
    if args.output is None:
        args.output = Path(__file__).parent / "artifacts" / (args.policy + "-" + args.split + ".json")
    if args.policy in {"base", "sft", "rl"} and args.output.exists():
        parser.error("Model evaluation output exists; choose a new checkpoint-specific file")
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    scenarios = load_scenarios(args.dataset, args.split) if args.dataset else load_scenarios(split=args.split)
    policy = decide if args.policy == "rules" else http_policy(args.endpoint, args.policy, args.timeout)
    result = evaluate(scenarios, policy, args.policy, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".summary.json").write_text(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "case_count": result["case_count"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
