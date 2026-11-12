"""Fixed, observation-only baseline. It never receives scenario labels."""
from __future__ import annotations


def resolved_facts(state: dict) -> dict:
    candidates = {}
    for observation in state["observations"]:
        for fact in observation.get("facts", []):
            if fact.get("current", True):
                candidates.setdefault(fact["field"], []).append(fact)
    result = {}
    for field, facts in candidates.items():
        rank = max(f.get("authority", 1) for f in facts)
        best = [f for f in facts if f.get("authority", 1) == rank]
        if len({str(f["value"]) for f in best}) == 1:
            result[field] = best[-1]
    return result


def decide(state: dict) -> dict:
    facts = resolved_facts(state)
    constraints = state["icp"].get("qualification_rules", {})
    requirements = constraints.get("all", {})
    reject = constraints.get("reject_if", {})
    fields = {key: fact["value"] for key, fact in facts.items()}
    evidence = list(dict.fromkeys(fact["source_id"] for fact in facts.values()))
    disposition = None
    if any(key in fields and fields[key] in values for key, values in reject.items()):
        disposition = "rejected"
    elif requirements and all(key in fields and fields[key] in values for key, values in requirements.items()):
        disposition = "qualified"
    available = [a for a in state["available_actions"] if a["action"] != "finish"]
    if disposition is None and available:
        # Prefer a primary source, then resolve employment, then mailbox status.
        priority = {"inspect_source": 0, "verify_person": 1, "verify_contact": 2, "search": 3}
        def rank(action):
            if action.get("source_kind") == "news_archive":
                return 4
            if action["action"] == "search" and "region" not in fields and "headquarters" in action.get("arguments", {}).get("query", ""):
                return 2.5
            return priority[action["action"]]
        available.sort(key=rank)
        selected = available[0]
        return {"action": selected["action"], "arguments": selected["arguments"],
                "reason": "Inspect the next available source to resolve missing or conflicting facts.",
                "policy_id": "rules", "checkpoint_id": None}
    return {"action": "finish", "arguments": {"disposition": disposition or "unresolved",
            "fields": fields, "evidence_ids": evidence,
            "reason": "Evidence meets the approved criteria." if disposition == "qualified" else
                      "A sourced exclusion applies." if disposition == "rejected" else
                      "The available evidence does not settle every required criterion."},
            "reason": "Finish with the disposition supported by inspected sources.",
            "policy_id": "rules", "checkpoint_id": None}
