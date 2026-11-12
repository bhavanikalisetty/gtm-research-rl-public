"""Build a train-only stopping-recovery SFT set from student-visited states.

This is a separate supervised repair, never teacher-driven GRPO. The rules
teacher receives public state only. Validation/test trajectories are rejected.
"""
import argparse
import json
import random
from pathlib import Path
from research.environment import ResearchEnvironment, load_scenarios
from research.environment.rules import decide
from .policy import canonical, public_state, sha256_file
from .runner import atomic_json, normalize_sft_row


def build(original_rows, trajectories, scenarios):
    by_id = {s["id"]: s for s in scenarios if s["split"] == "train"}
    original = [normalize_sft_row(row) for row in original_rows]
    if any(row.get("split") != "train" for row in original):
        raise ValueError("Recovery SFT accepts training demonstrations only")
    seen = {canonical(row["state"]) for row in original}
    extra = []
    for group in trajectories:
        if group["scenario_id"] not in by_id:
            raise ValueError("Nontraining scenario in recovery trajectories")
        scenario = by_id[group["scenario_id"]]
        for rollout in group["rollouts"]:
            env = ResearchEnvironment(scenario)
            for event in rollout["events"]:
                state = public_state(env.state())
                teacher = decide(state)
                key = canonical(state)
                if (teacher["action"] == "finish" and event["action"].get("action") != "finish"
                        and key not in seen):
                    seen.add(key)
                    extra.append({"state": state,
                        "action": {k: teacher[k] for k in ("action", "arguments", "reason")},
                        "split": "train", "case_id": scenario["id"], "group_id": scenario["group_id"],
                        "provenance": "synthetic", "teacher": "rules-v1-student-state-recovery"})
                _, _, done, _ = env.step(event["action"])
                if done:
                    break
    # Keep every original action example and cap added finishes at a 1:1 mix.
    original_finish = sum(row["action"]["action"] == "finish" for row in original)
    original_research = len(original) - original_finish
    random.Random(3407).shuffle(extra)
    selected = extra[:max(0, original_research - original_finish)]
    rows = original + selected
    return rows, {"original_rows": len(original), "candidate_recovery_rows": len(extra),
                  "selected_recovery_rows": len(selected), "rows": len(rows),
                  "finish_rows": original_finish + len(selected), "research_rows": original_research,
                  "groups": len({row["group_id"] for row in rows}), "split": "train",
                  "teacher_saw_hidden_labels": False,
                  "method": "Public-state rules finish targets where the student continued researching"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise SystemExit("Do not overwrite a recovery dataset")
    original_path = Path("research/data/sft/train.jsonl")
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    groups = [json.loads(line) for line in Path(args.trajectories).read_text().splitlines()]
    rows, audit = build(original, groups, load_scenarios(split="train"))
    if not audit["selected_recovery_rows"]:
        raise SystemExit("No training-state stopping corrections found")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(canonical(row) + "\n" for row in rows))
    atomic_json(target.with_suffix(".audit.json"), {**audit, "dataset_hash": sha256_file(target),
        "source_demonstrations_hash": sha256_file(original_path),
        "source_trajectories_hash": sha256_file(args.trajectories)})
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
