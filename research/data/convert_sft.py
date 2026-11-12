"""Convert observation-only rules demonstrations to assistant-only SFT records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.environment import ResearchEnvironment, dataset_hash, load_scenarios
from research.environment.environment import canonical
from research.environment.rules import decide

SYSTEM_PROMPT = (
    "You are a bounded prospect research policy. Return one JSON object with action, arguments and reason. "
    "Choose a currently available research action or finish. Use only inspected evidence. "
    "Resolve conflicting current sources before qualifying. Stale employment is not current employment. "
    "Catch-all is not a verified individual mailbox. Finish within four research calls. "
    "For finish, arguments must contain disposition (qualified, rejected or unresolved), fields, evidence_ids and reason. "
    "Cite inspected source IDs and include only supported fields. Follow the approved ICP qualification_rules. "
    "Unknown facts stay unknown. A source URL alone is not evidence."
)
PROMPT_VERSION = "research-policy-v1"


def messages_for_state(state: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": canonical(state)}]


def convert(scenarios: list[dict]) -> tuple[list[dict], list[dict]]:
    examples, trajectories = [], []
    for scenario in scenarios:
        env = ResearchEnvironment(scenario)
        trace = []
        for turn in range(5):
            state = env.state()
            raw = decide(state)
            action = {key: raw[key] for key in ("action", "arguments", "reason")}
            prompt = messages_for_state(state)
            completion = [{"role": "assistant", "content": canonical(action)}]
            metadata = {"case_id": scenario["id"], "group_id": scenario["group_id"], "split": scenario["split"],
                        "turn": turn, "provenance": scenario["provenance"], "teacher": "rules-v1", "prompt_version": PROMPT_VERSION}
            examples.append({"prompt": prompt, "completion": completion, "metadata": metadata})
            trace.extend([{**message, "loss_mask": 0} for message in prompt])
            trace.append({**completion[0], "loss_mask": 1})
            _, _, done, _ = env.step(action)
            if done:
                break
        if not env.score().get("evidence_supported") or not env.score().get("decision_correct"):
            raise ValueError(f"Unsupported teacher demonstration: {scenario['id']}")
        trajectories.append({"case_id": scenario["id"], "split": scenario["split"], "messages": trace})
    return examples, trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "sft")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    counts = {}
    for split in ("train", "validation"):
        rows, traces = convert([s for s in scenarios if s["split"] == split])
        for name, data in ((split, rows), (split + "_traces", traces)):
            (args.output_dir / (name + ".jsonl")).write_text("".join(canonical(row) + "\n" for row in data))
        counts[split] = {"examples": len(rows), "trajectories": len(traces)}
    manifest = {"dataset_hash": dataset_hash(scenarios), "prompt_version": PROMPT_VERSION, "counts": counts,
                "loss": "assistant completion tokens only; prompt, observations and metadata masked",
                "test_exported": False, "teacher": "rules-v1", "provenance": "synthetic"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
