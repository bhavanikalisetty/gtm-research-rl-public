import json
from pathlib import Path
import pytest
from research.environment import load_scenarios
from research.training.recovery import build


def test_recovery_rejects_holdout_rollouts():
    rows = [json.loads(line) for line in Path("research/data/sft/train.jsonl").read_text().splitlines()]
    heldout = load_scenarios(split="validation")[0]
    with pytest.raises(ValueError, match="Nontraining"):
        build(rows, [{"scenario_id": heldout["id"], "rollouts": []}], load_scenarios(split="train"))


def test_recovery_preserves_original_data_without_student_states():
    rows = [json.loads(line) for line in Path("research/data/sft/train.jsonl").read_text().splitlines()]
    result, audit = build(rows, [], load_scenarios(split="train"))
    assert len(result) == len(rows) == 121
    assert audit["selected_recovery_rows"] == 0
    assert all(r["split"] == "train" and "hidden_reference" not in r["state"] for r in result)
