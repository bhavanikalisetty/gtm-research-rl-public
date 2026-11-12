"""Deterministic interactive research environment; labels stay evaluator-side."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

VERSION = "0.1.0"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "scenarios.jsonl"
RESEARCH_ACTIONS = {"search", "inspect_source", "verify_person", "verify_contact"}
DISPOSITIONS = {"qualified", "rejected", "unresolved"}
PRIVATE_KEYS = {"hidden_reference", "required_evidence_ids", "expected_disposition", "reward",
                "reward_score", "scoring_metadata", "fixture_kind", "group_id", "split",
                "evaluation_split", "case_id", "scenario_id", "tool_results", "label"}


def assert_policy_safe(value: Any) -> None:
    """Reject annotation metadata accidentally embedded in a public observation."""
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError("Private annotation key found in policy-visible data")
        for child in value.values():
            assert_policy_safe(child)
    elif isinstance(value, list):
        for child in value:
            assert_policy_safe(child)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dataset_hash(scenarios: list[dict]) -> str:
    return hashlib.sha256(canonical(scenarios).encode()).hexdigest()


def load_scenarios(path: str | Path = DATA_PATH, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [row for row in rows if split is None or (row.get("evaluation_split", row["split"]) if split == "live_transfer" else row["split"]) == split]


def action_key(action: dict) -> str:
    return canonical({"action": action.get("action"), "arguments": action.get("arguments", {})})


class ResearchEnvironment:
    """reset -> state; step -> (state, reward, done, public_info).

    `score()` is evaluator-only and must never be inserted into a policy prompt.
    The scenario object and this Python process are trusted harness inputs, not a
    sandbox against arbitrary policy-generated Python. Policies receive JSON only.
    """

    def __init__(self, scenario: dict | None = None):
        self._scenario = None
        if scenario is not None:
            self.reset(scenario)

    def reset(self, scenario: dict | None = None, seed: int = 0) -> dict:
        if scenario is not None:
            self._scenario = copy.deepcopy(scenario)
        if self._scenario is None:
            raise ValueError("reset requires a scenario on first use")
        s = self._scenario
        assert_policy_safe(s["icp"])
        assert_policy_safe(s["initial_observations"])
        for row in s["tool_results"]:
            assert_policy_safe(row["observation"])
        if not 0 <= s.get("max_steps", 4) <= 4:
            raise ValueError("max_steps must be between zero and four")
        self._rows = {action_key(row): row for row in s["tool_results"]}
        if len(self._rows) != len(s["tool_results"]):
            raise ValueError("duplicate tool action keys")
        if any(row.get("cost_usd", 0) < 0 for row in s["tool_results"]):
            raise ValueError("negative prices are not allowed")
        self.observations = copy.deepcopy(s["initial_observations"])
        self.history = []
        self.done = False
        self.terminal = None
        self.calls = 0
        self.simulated_cost_usd = 0.0
        self.simulated_latency_ms = 0
        self.observation_tokens = 0
        self._used = set()
        self._available = {}
        for observation in self.observations:
            self._discover(observation)
        return self.state()

    def _discover(self, observation: dict) -> None:
        for action in observation.get("available_actions", []):
            key = action_key(action)
            if key not in self._rows:
                raise ValueError("observation advertises unknown tool action")
            self._available[key] = copy.deepcopy(action)

    def state(self) -> dict:
        s = self._scenario
        available = []
        for key, action in self._available.items():
            row = self._rows[key]
            if key not in self._used and self.calls < s.get("max_steps", 4) and self._affordable(row):
                available.append(copy.deepcopy(action))
        available.append({"action": "finish", "arguments_schema": {
            "disposition": "qualified|rejected|unresolved", "fields": "object",
            "evidence_ids": "array of inspected source IDs", "reason": "short explanation"}})
        # Explicit allowlist: no scenario ID, split, branch catalog or labels.
        return copy.deepcopy({
            "icp": s["icp"], "observations": self.observations, "history": self.history,
            "remaining_budget_usd": round(max(0, s["budget_usd"] - self.simulated_cost_usd), 6),
            "remaining_steps": max(0, s.get("max_steps", 4) - self.calls),
            "remaining_observation_tokens": max(0, s.get("max_observation_tokens", 4000) - self.observation_tokens),
            "remaining_simulated_time_ms": max(0, s.get("max_simulated_time_ms", 20000) - self.simulated_latency_ms),
            "available_actions": available,
        })

    def _affordable(self, row: dict) -> bool:
        s = self._scenario
        return (self.simulated_cost_usd + row.get("cost_usd", 0) <= s["budget_usd"] + 1e-9
                and self.observation_tokens + row.get("observation_tokens", 200) <= s.get("max_observation_tokens", 4000)
                and self.simulated_latency_ms + row.get("latency_ms", 1000) <= s.get("max_simulated_time_ms", 20000))

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode finished; reset before stepping")
        if not isinstance(action, dict):
            action = {"action": "invalid", "arguments": {}}
        name = action.get("action")
        args = action.get("arguments", {})
        if name == "finish":
            valid = (isinstance(args, dict) and isinstance(args.get("disposition"), str) and args.get("disposition") in DISPOSITIONS
                     and isinstance(args.get("fields", {}), dict)
                     and isinstance(args.get("evidence_ids", []), list)
                     and all(isinstance(x, str) for x in args.get("evidence_ids", [])))
            self.terminal = copy.deepcopy(args) if valid else {"disposition": "invalid", "fields": {}, "evidence_ids": []}
            self.done = True
            self.history.append({"action": "finish", "arguments": copy.deepcopy(self.terminal)})
            return self.state(), self.get_reward(), True, self._info(None if valid else "invalid_finish")
        if self.calls >= self._scenario.get("max_steps", 4):
            # The fifth decision must be finish; no infinite no-reward episodes.
            self.terminal = {"disposition": "invalid", "fields": {}, "evidence_ids": []}
            self.done = True
            self.history.append({"action": "invalid", "error": "finish_required"})
            return self.state(), self.get_reward(), True, self._info("finish_required")
        self.calls += 1  # invalid and repeated attempts consume the action budget
        key = action_key(action)
        row = self._rows.get(key)
        error = None
        if not isinstance(name, str) or name not in RESEARCH_ACTIONS or not isinstance(args, dict):
            error = "invalid_action"
        elif key not in self._available:
            error = "action_not_discovered"
        elif key in self._used:
            error = "repeated_action"
        elif not self._affordable(row):
            error = "tool_budget_exhausted"
        if error:
            charge = min(0.002, max(0, self._scenario["budget_usd"] - self.simulated_cost_usd))
            self.simulated_cost_usd += charge
            self.simulated_latency_ms += min(100, max(0, self._scenario.get("max_simulated_time_ms", 20000) - self.simulated_latency_ms))
            self.observation_tokens += min(20, max(0, self._scenario.get("max_observation_tokens", 4000) - self.observation_tokens))
            self.history.append({"action": name, "arguments": copy.deepcopy(args), "error": error,
                                 "simulated_cost_usd": charge})
            return self.state(), 0.0, False, self._info(error)
        self._used.add(key)
        self.simulated_cost_usd += row["cost_usd"]
        self.simulated_latency_ms += row.get("latency_ms", 1000)
        self.observation_tokens += row.get("observation_tokens", 200)
        observation = copy.deepcopy(row["observation"])
        self.observations.append(observation)
        self._discover(observation)
        self.history.append({"action": name, "arguments": copy.deepcopy(args),
                             "observation_index": len(self.observations) - 1,
                             "simulated_cost_usd": row["cost_usd"]})
        return self.state(), 0.0, False, self._info()

    def _info(self, error: str | None = None) -> dict:
        return {"error": error, "research_calls": self.calls,
                "cost": {"actual_usd": None, "simulated_usd": round(self.simulated_cost_usd, 6)},
                "simulated_latency_ms": self.simulated_latency_ms}

    def score(self) -> dict:
        """Evaluator-only diagnostics; includes gold comparisons, never policy input."""
        if not self.done:
            return {"reward": 0.0, "terminal": False}
        ref, terminal = self._scenario["hidden_reference"], self.terminal
        evidence = set(terminal.get("evidence_ids", []))
        sources = {source["id"] for obs in self.observations for source in obs.get("sources", [])}
        facts = [fact for obs in self.observations for fact in obs.get("facts", []) if fact.get("current", True)]
        fields = terminal.get("fields", {})
        invented = len(evidence - sources)
        unsupported = sum(not any(f["field"] == key and f["value"] == value and f["source_id"] in evidence
                                  for f in facts) for key, value in fields.items() if value is not None)
        required = set(ref.get("required_evidence_ids", []))
        citation_support = bool(required) and required <= evidence & sources
        expected_fields = ref.get("fields", {})
        correct_fields = sum(key in fields and fields[key] == value for key, value in expected_fields.items())
        field_accuracy = correct_fields / len(expected_fields) if expected_fields else 1.0
        correct = terminal.get("disposition") == ref["disposition"]
        supported = citation_support and not invented and not unsupported and field_accuracy == 1.0
        # Correct but unsupported guessing is negative. Cheap reject-all cannot win.
        quality = 3.0 if correct and supported else (-1.0 if correct else -3.0)
        cost_penalty = min(0.5, 0.05 * self.calls + 0.3 * self.simulated_cost_usd / max(self._scenario["budget_usd"], 0.001))
        reward = quality - min(2.0, 0.5 * (unsupported + invented)) - cost_penalty
        return {"reward": round(reward, 6), "terminal": True, "decision_correct": correct,
                "evidence_supported": supported, "citation_support": citation_support,
                "invented_evidence_count": invented, "unsupported_field_count": unsupported,
                "asserted_field_count": sum(value is not None for value in fields.values()),
                "field_accuracy": field_accuracy, "expected_disposition": ref["disposition"],
                "cost_penalty": round(cost_penalty, 6)}

    def get_reward(self) -> float:
        return self.score()["reward"]
