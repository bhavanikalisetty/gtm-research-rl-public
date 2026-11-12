"""Evaluator-side audit. Never put these outputs into training prompts."""
from __future__ import annotations

import copy
import json
from collections import Counter, deque
from pathlib import Path

from research.environment import ResearchEnvironment, dataset_hash, load_scenarios
from research.environment.rules import decide


def audit_case(scenario):
    reference = scenario["hidden_reference"]
    queue = deque([(ResearchEnvironment(scenario), [])])
    solutions = []
    while queue:
        env, path = queue.popleft()
        terminal = copy.deepcopy(env)
        terminal.step({"action": "finish", "arguments": {
            "disposition": reference["disposition"], "fields": reference["fields"],
            "evidence_ids": reference["required_evidence_ids"]}})
        if terminal.score()["evidence_supported"]:
            solutions.append({"calls": env.calls, "simulated_cost_usd": env.simulated_cost_usd, "path": path})
            continue
        for action in env.state()["available_actions"]:
            if action["action"] == "finish":
                continue
            branch = copy.deepcopy(env)
            branch.step(action)
            queue.append((branch, path + [{"action": action["action"], "arguments": action["arguments"]}]))
    if not solutions:
        raise ValueError("No evidence-supported solution within budget: " + scenario["id"])
    best = min(solutions, key=lambda item: (item["simulated_cost_usd"], item["calls"]))
    baseline = ResearchEnvironment(scenario)
    for _ in range(5):
        _, _, done, _ = baseline.step(decide(baseline.state()))
        if done:
            break
    return {"case_id": scenario["id"], "group_id": scenario["group_id"], "split": scenario["split"],
            "provenance": scenario["provenance"], "label": reference["disposition"],
            "audit_type": "programmatic reachability and exact structured field/citation support",
            "independent_human_review": False, "reachable_supported_label": True,
            "reference_fields": reference["fields"], "required_evidence_ids": reference["required_evidence_ids"],
            "minimum_simulated_cost_solution": best,
            "baseline_supported": baseline.score()["evidence_supported"],
            "baseline_simulated_cost_usd": baseline.simulated_cost_usd}


def main():
    scenarios = load_scenarios()
    rows = [audit_case(s) for s in scenarios]
    path = Path(__file__).parent / "label_audit.json"
    result = {"dataset_hash": dataset_hash(scenarios), "case_count": len(rows),
              "label_counts": dict(Counter(row["label"] for row in rows)),
              "all_labels_reachable": all(row["reachable_supported_label"] for row in rows),
              "independent_human_review": False, "cases": rows}
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
