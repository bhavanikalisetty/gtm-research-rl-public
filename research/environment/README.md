# Bounded prospect research

Run from the repository root with Python 3.10 or later. There are no third-party dependencies.

```python
from research.environment import ResearchEnvironment, load_scenarios
from research.environment.rules import decide

scenario = load_scenarios(split="train")[0]
env = ResearchEnvironment()
state = env.reset(scenario, seed=0)
while True:
    action = decide(state)  # replace with the policy under training
    state, reward, terminated, info = env.step(action)
    if terminated:
        break
assert reward == env.get_reward()
```

`reset(scenario=None, seed=0) -> dict` resets counters and returns the contract policy state. Pass a scenario on the first reset. Later resets can reuse it. The environment has no random transitions. Identical scenarios return identical initial states for every GRPO group member, independent of seed. Create an independent environment per member.

`step(action: dict) -> (state: dict, reward: float, terminated: bool, info: dict)` accepts `{action, arguments, reason}`. Research actions are `search`, `inspect_source`, `verify_person` and `verify_contact`. The exact currently valid arguments are listed in `state.available_actions`. Only discovered branches are listed. An inspection can reveal a new person, contact or search action. It never reveals another branch's result. Actions return different evidence. Repeated and undiscovered actions reveal nothing and consume a research step plus a small simulated cost. Four research attempts are allowed. The fifth decision must be `finish`; any other fifth action terminates as invalid with a negative reward. A finished environment raises on further steps.

`finish` arguments are `{disposition: "qualified"|"rejected"|"unresolved", fields: {...}, evidence_ids: [...], reason: "..."}`. Fields must have exact values supported by the cited inspected sources. Source catalog links alone are not evidence. A valid contact requires a contact-verification observation, and catch-all does not imply a valid mailbox.

`get_reward() -> float` returns zero before termination and the final scalar afterward. `score() -> dict` is evaluator-only: it includes gold comparisons and must never enter a policy prompt. `info` contains errors, research call count and cost, without hidden labels. Raw scenario objects, split manifests and `env._scenario` are trusted harness inputs. Give the policy only `state` JSON, never a whole environment object or scenario. This is an environment interface, not a sandbox for arbitrary model-generated Python.

The state includes the required contract fields plus remaining observation-token and simulated-time budgets. `observation_tokens` is a declared simulated tool-output budget, not measured tokenizer usage. Latency fields from tools are simulated. Actual policy latency and token use belong in evaluation usage. Per-action simulated costs are fixed in the scenario; failed and repeated calls are charged. An action that would exceed money, token or simulated-time limits returns an error without revealing its result.

Reward is +3 for a correct disposition with all required fields and observed citations, -1 for a correct but unsupported decision, and -3 for a wrong or invalid decision. Unsupported structured fields and invented citations subtract up to 2. Research cost and attempts subtract at most 0.5. Correctness dominates possible tool-cost savings. The scorer evaluates structured fields and citation coverage; it does not independently fact-check arbitrary prose in `reason`.

The first dataset contains 60 synthetic company groups: 36 train, 12 validation and 12 test. There are 24 positive, 24 negative and 12 unresolved cases across 10 templates. Splits were written before scenarios. All company variants stay in one split. Template reuse across splits makes this a controlled pilot rather than evidence of real-world generalization. Company domains use the reserved `.example` suffix. No production data or real contact addresses are included.

For SFT, train only on assistant action completions. User observations, tool results, hidden references and metadata receive no loss. The conversion script emits prompt/completion rows and a separate trace format with explicit role loss masks. Do not concatenate metadata into prompts. For RL, the current policy must generate every action; the rules baseline is not an RL rollout generator.
