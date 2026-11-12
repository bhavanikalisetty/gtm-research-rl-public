"""Measure reward exploit attempts on validation fixtures only."""
import json
from pathlib import Path

from research.environment import load_scenarios
from research.evaluation.run import evaluate


def immediate(disposition):
    return lambda state: {"action": "finish", "arguments": {"disposition": disposition, "fields": {}, "evidence_ids": []}}


def invented(state):
    return {"action": "finish", "arguments": {"disposition": "qualified", "fields": {"contact_status": "valid"}, "evidence_ids": ["invented-source"]}}


def repeated(state):
    if state["history"]:
        return {key: state["history"][0][key] for key in ("action", "arguments")}
    return next(action for action in state["available_actions"] if action["action"] != "finish")


def main():
    scenarios = load_scenarios(split="validation")
    reports = []
    for name, policy in (("reject_all", immediate("rejected")), ("qualify_all", immediate("qualified")),
                         ("stop_unresolved", immediate("unresolved")), ("invented_evidence", invented), ("repeated_calls", repeated)):
        result = evaluate(scenarios, policy, name)
        assert result["metrics"]["mean_reward"] < 0
        reports.append({"strategy": name, "case_count": result["case_count"], "metrics": result["metrics"]})
    result = {"purpose": "Reward exploit diagnostics, not learned-model evaluations", "split": "validation", "provenance": "fixture", "strategies": reports}
    path = Path(__file__).parent / "artifacts" / "reward-challenges.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({r["strategy"]: r["metrics"]["mean_reward"] for r in reports}, indent=2))


if __name__ == "__main__":
    main()
