"""One JSON action per decision; observations are always outside the loss mask."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.data.convert_sft import PROMPT_VERSION, SYSTEM_PROMPT as SYSTEM
STATE_KEYS = {"icp", "observations", "history", "remaining_budget_usd", "remaining_steps",
              "available_actions", "remaining_observation_tokens", "remaining_simulated_time_ms"}
ACTIONS = {"search", "inspect_source", "verify_person", "verify_contact", "finish"}
PROMPT_V2 = "research-policy-v2-current-actions"
PROMPT_V3 = "research-policy-v3-explicit-fields"


def prompt_version(config: dict | None = None) -> str:
    version = (config or {}).get("prompt_version", PROMPT_VERSION)
    if version not in {PROMPT_VERSION, PROMPT_V2, PROMPT_V3}:
        raise ValueError("Unknown policy prompt version")
    return version


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_state(state: dict) -> dict:
    # Accept only the environment public-state interface, never an entire scenario.
    if not isinstance(state, dict) or not {"icp", "observations", "available_actions"} <= state.keys():
        raise ValueError("Expected a policy-visible state, not a scenario")
    return {key: state[key] for key in STATE_KEYS if key in state}


def messages_for(state: dict, version: str = PROMPT_VERSION) -> list[dict]:
    prompt_version({"prompt_version": version})
    visible = public_state(state)
    if version == PROMPT_V3:
        from research.data.v3.policy_view import messages_for_state
        return messages_for_state(visible)
    if version == PROMPT_V2:
        from research.data.public_v2.policy_view import messages_for_state
        return messages_for_state(visible)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(visible)}]


def prompt_ids(tokenizer, state: dict, max_context: int, max_new_tokens: int,
               version: str = PROMPT_VERSION) -> list[int]:
    rendered = tokenizer.apply_chat_template(messages_for(state, version), tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    # Fail explicitly. Left truncation could delete the ICP or evidence IDs.
    if len(ids) + max_new_tokens > max_context:
        raise ValueError(f"state exceeds context budget: {len(ids)} + {max_new_tokens} > {max_context}")
    return ids


def parse_action(text: str) -> dict:
    obj = json.loads(text.strip())
    if not isinstance(obj, dict) or obj.get("action") not in ACTIONS or not isinstance(obj.get("arguments"), dict):
        raise ValueError("model output does not match action schema")
    args = obj["arguments"]
    if obj["action"] == "finish":
        if args.get("disposition") not in {"qualified", "rejected", "unresolved"}:
            raise ValueError("invalid disposition")
        if not isinstance(args.get("fields", {}), dict) or not isinstance(args.get("evidence_ids", []), list):
            raise ValueError("invalid finish fields/evidence")
        if not all(isinstance(e, str) for e in args.get("evidence_ids", [])):
            raise ValueError("invalid evidence ID")
    if not isinstance(obj.get("reason", ""), str):
        raise ValueError("invalid action reason")
    return {"action": obj["action"], "arguments": args, "reason": obj.get("reason", args.get("reason", ""))}


@dataclass
class DecisionTokens:
    prompt: list[int]
    completion: list[int]
    text: str = ""

    @property
    def input_ids(self):
        return self.prompt + self.completion

    @property
    def loss_mask(self):
        return [0] * len(self.prompt) + [1] * len(self.completion)

    @property
    def labels(self):
        return [-100] * len(self.prompt) + self.completion


def sft_tokens(tokenizer, state: dict, action: dict, max_context: int, max_new_tokens: int,
               version: str = PROMPT_VERSION) -> DecisionTokens:
    # Header, observations and history live only in the prompt. The target is one action.
    target = canonical(parse_action(canonical(action)))
    completion = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(completion) > max_new_tokens:
        raise ValueError("SFT action exceeds output token budget")
    return DecisionTokens(prompt_ids(tokenizer, state, max_context, max_new_tokens, version), completion, target)
