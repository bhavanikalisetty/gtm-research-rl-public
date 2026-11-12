import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from research.training.policy import DecisionTokens, sft_tokens, parse_action, public_state, prompt_ids
from research.training.runner import group_advantages, clipped_objective, normalize_sft_row, Deadline
from research.training.model import token_logps
from research.environment import load_scenarios, ResearchEnvironment
from research.environment.rules import decide
from research.data.convert_sft import convert


class ByteTokenizer:
    eos_token_id = 256
    def encode(self, s, **kwargs): return list(s.encode())
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages) + '<assistant>'


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(257, 257)
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.table(input_ids))


def test_loss_excludes_observation_tokens_and_has_real_gradient():
    model = TinyModel()
    d = DecisionTokens([10, 11, 12, 13], [20, 21])
    loss = -token_logps(model, d).mean()
    loss.backward()
    assert model.table.weight.grad[10:13].abs().sum() == 0
    assert model.table.weight.grad[13].abs().sum() > 0
    assert model.table.weight.grad[20].abs().sum() > 0
    assert d.labels == [-100] * 4 + [20, 21]
    assert d.loss_mask == [0] * 4 + [1, 1]


def test_actual_data_records_convert_without_hidden_truth():
    scenario = load_scenarios(split='train')[0]
    records, _ = convert([scenario])
    row = normalize_sft_row(records[0])
    assert row['split'] == 'train'
    assert 'hidden_reference' not in row['state']
    assert 'tool_results' not in row['state']
    d = sft_tokens(ByteTokenizer(), row['state'], row['action'], 40000, 2000)
    assert sum(d.loss_mask) == len(d.completion)
    assert b'hidden_reference' not in bytes(d.prompt)


def test_context_never_silently_truncates_evidence():
    state = ResearchEnvironment(load_scenarios(split='train')[0]).state()
    with pytest.raises(ValueError, match='context budget'):
        prompt_ids(ByteTokenizer(), state, 10, 5)


def test_grpo_rewards_move_likelihood_in_correct_direction():
    adv = group_advantages([3., -3., 1., -1.])
    assert abs(float(adv.mean())) < 1e-6
    logps = torch.zeros(4, requires_grad=True)
    losses = [clipped_objective(logps[i:i+1], torch.zeros(1), adv[i]) for i in range(4)]
    sum(losses).backward()
    assert logps.grad[0] < 0 and logps.grad[1] > 0
    assert torch.isfinite(group_advantages([1., 1., 1., 1.])).all()
    assert group_advantages([1., 1., 1., 1.]).abs().sum() == 0


def test_malformed_policy_output_is_rejected():
    for value in ['hello', '{"action":"send_email","arguments":{}}',
                  '{"action":"finish","arguments":{"disposition":"yes"}}']:
        with pytest.raises(ValueError): parse_action(value)


def test_deadline():
    deadline = Deadline(0)
    with pytest.raises(TimeoutError): deadline.check()
