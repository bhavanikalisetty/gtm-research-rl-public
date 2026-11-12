import copy
import json
import unittest

from research.data.convert_sft import convert
from research.environment import ResearchEnvironment, load_scenarios
from research.environment.rules import decide
from research.evaluation.run import evaluate


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = load_scenarios()

    def sample(self, kind="easy_positive"):
        return next(copy.deepcopy(s) for s in self.scenarios if s["hidden_reference"]["fixture_kind"] == kind)

    def rollout(self, scenario, policy=decide):
        env = ResearchEnvironment(scenario)
        for _ in range(5):
            _, _, done, _ = env.step(policy(env.state()))
            if done:
                return env
        self.fail("episode did not end in five decisions")

    def test_rules_complete_every_case_with_supported_fields(self):
        for scenario in self.scenarios:
            with self.subTest(case=scenario["id"]):
                env = self.rollout(scenario)
                self.assertTrue(env.score()["decision_correct"])
                self.assertTrue(env.score()["evidence_supported"])
                self.assertGreater(env.get_reward(), 2)
                self.assertLessEqual(env.calls, 4)

    def test_hidden_truth_and_uninspected_results_never_enter_state(self):
        scenario = self.sample()
        scenario["hidden_reference"]["secret"] = "HIDDEN_SENTINEL"
        scenario["id"] = "ANSWER_BEARING_ID"
        scenario["tool_results"][-1]["observation"]["private_sentinel"] = "UNSEEN_BRANCH"
        env = ResearchEnvironment(scenario)
        before = json.dumps(env.state())
        for forbidden in ("HIDDEN_SENTINEL", "hidden_reference", "ANSWER_BEARING_ID", "UNSEEN_BRANCH", "required_evidence_ids", "fixture_kind", "group_id", '"split"'):
            self.assertNotIn(forbidden, before)
        env.step(env.state()["available_actions"][0])
        self.assertNotIn("UNSEEN_BRANCH", json.dumps(env.state()))

    def test_same_initial_state_for_all_grpo_members(self):
        scenario = self.sample("conflicting_location")
        states = [ResearchEnvironment().reset(scenario, seed=n) for n in range(4)]
        self.assertTrue(all(state == states[0] for state in states))
        states[0]["icp"]["regions"].append("mutated")
        self.assertNotEqual(states[0], states[1])

    def test_annotations_in_observations_are_rejected(self):
        scenario = self.sample()
        scenario["tool_results"][0]["observation"]["expected_disposition"] = "qualified"
        with self.assertRaises(ValueError):
            ResearchEnvironment(scenario)

    def test_actions_reveal_different_results_and_require_discovery(self):
        scenario = self.sample()
        env = ResearchEnvironment(scenario)
        hidden_action = scenario["tool_results"][1]
        _, _, _, info = env.step(hidden_action)
        self.assertEqual(info["error"], "action_not_discovered")
        self.assertEqual(len(env.observations), 1)
        env.reset()
        first = env.state()["available_actions"][0]
        env.step(first)
        company = copy.deepcopy(env.observations[-1])
        env.step(next(a for a in env.state()["available_actions"] if a["action"] == "verify_contact"))
        self.assertNotEqual(company, env.observations[-1])
        self.assertEqual(env.observations[-1]["facts"][0]["field"], "contact_status")

    def test_repeat_consumes_steps_cost_and_reveals_nothing(self):
        env = ResearchEnvironment(self.sample())
        action = env.state()["available_actions"][0]
        env.step(action)
        count, cost = len(env.observations), env.simulated_cost_usd
        _, _, _, info = env.step(action)
        self.assertEqual(info["error"], "repeated_action")
        self.assertEqual(len(env.observations), count)
        self.assertGreater(env.simulated_cost_usd, cost)
        self.assertEqual(env.calls, 2)
        for _ in range(3):
            _, reward, done, _ = env.step(action)
        self.assertTrue(done)
        self.assertLess(reward, 0)
        self.assertEqual(env.calls, 4)

    def test_money_tokens_and_time_block_unaffordable_results(self):
        for key, value in (("budget_usd", 0), ("max_observation_tokens", 0), ("max_simulated_time_ms", 0)):
            scenario = self.sample()
            scenario[key] = value
            env = ResearchEnvironment(scenario)
            self.assertEqual([a["action"] for a in env.state()["available_actions"]], ["finish"])
            _, _, _, info = env.step(scenario["tool_results"][0])
            self.assertEqual(info["error"], "tool_budget_exhausted")
            self.assertEqual(len(env.observations), 1)

    def test_reject_all_and_stop_immediately_lose_to_baseline(self):
        for disposition in ("qualified", "rejected", "unresolved"):
            def shortcut(_state):
                return {"action": "finish", "arguments": {"disposition": disposition, "fields": {}, "evidence_ids": []}}
            results = [self.rollout(s, shortcut).get_reward() for s in self.scenarios]
            self.assertLess(sum(results) / len(results), 0)
        result = evaluate([s for s in self.scenarios if s["split"] == "validation"], shortcut, "shortcut")
        self.assertEqual(result["metrics"]["qualification_recall"], 0)
        self.assertIsNone(result["metrics"]["qualification_precision"])
        self.assertIsNone(result["metrics"]["simulated_cost_per_supported_qualified_lead_usd"])

    def test_invented_evidence_and_unsupported_fields_are_negative(self):
        scenario = self.sample()
        for evidence, fields in ((["invented"], scenario["hidden_reference"]["fields"]),
                                 (scenario["hidden_reference"]["required_evidence_ids"], scenario["hidden_reference"]["fields"])):
            env = ResearchEnvironment(scenario)
            env.step({"action": "finish", "arguments": {"disposition": "qualified", "fields": fields, "evidence_ids": evidence}})
            self.assertLess(env.get_reward(), 0)
            self.assertFalse(env.score()["evidence_supported"])
        env = ResearchEnvironment(scenario)
        while decide(env.state())["action"] != "finish":
            env.step(decide(env.state()))
        action = decide(env.state())
        action["arguments"]["fields"]["contact_status"] = "catch_all"
        env.step(action)
        self.assertGreater(env.score()["unsupported_field_count"], 0)
        self.assertLess(env.get_reward(), 0)

    def test_stale_fact_cannot_support_current_employment(self):
        scenario = self.sample("stale_employment")
        env = ResearchEnvironment(scenario)
        env.step(env.state()["available_actions"][0])
        source = scenario["tool_results"][0]["observation"]["sources"][0]["id"]
        env.step({"action": "finish", "arguments": {"disposition": "qualified", "fields": {"employment_status": "current"}, "evidence_ids": [source]}})
        self.assertEqual(env.score()["unsupported_field_count"], 1)

    def test_split_groups_are_frozen_and_disjoint(self):
        manifest_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        frozen = {g["group_id"]: g["split"] for g in manifest["groups"]}
        seen = {}
        for scenario in self.scenarios:
            self.assertEqual(frozen[scenario["group_id"]], scenario["split"])
            self.assertEqual(seen.setdefault(scenario["group_id"], scenario["split"]), scenario["split"])
        self.assertEqual(len(seen), 60)

    def test_sft_only_trains_assistant_actions(self):
        rows, traces = convert([s for s in self.scenarios if s["split"] == "train"])
        for row in rows:
            self.assertEqual(row["completion"][0]["role"], "assistant")
            self.assertNotIn("hidden_reference", json.dumps(row["prompt"]))
            self.assertNotIn(row["metadata"]["case_id"], json.dumps(row["prompt"]))
        for trace in traces:
            for message in trace["messages"]:
                self.assertEqual(message["loss_mask"], int(message["role"] == "assistant"))

    def test_duplicate_challenges_inherit_the_original_company_split(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / "data"
        duplicates = load_scenarios(root / "challenges" / "scenarios.jsonl")
        self.assertEqual(len(duplicates), 6)
        original_groups = {s["group_id"]: s["split"] for s in self.scenarios}
        for scenario in duplicates:
            self.assertEqual(original_groups[scenario["group_id"]], scenario["split"])
            env = self.rollout(scenario)
            self.assertTrue(env.score()["decision_correct"])
            self.assertTrue(env.score()["evidence_supported"])

    def test_malformed_finish_and_research_are_bounded(self):
        for action in (None, [], {"action": {}}, {"action": "finish", "arguments": {"disposition": []}},
                       {"action": "finish", "arguments": {"disposition": "qualified", "evidence_ids": [{}]}}):
            env = ResearchEnvironment(self.sample())
            for _ in range(5):
                _, _, done, _ = env.step(action)
                if done:
                    break
            self.assertTrue(done)
            self.assertLess(env.get_reward(), 0)

    def test_evaluator_counts_failed_attempts_without_fallback(self):
        def fail(_state):
            raise RuntimeError("Simulated model failure")
        scenarios = [s for s in self.scenarios if s["split"] == "validation"]
        result = evaluate(scenarios, fail, "failed_policy")
        self.assertEqual(result["metrics"]["decision_accuracy"], 0)
        self.assertEqual(result["metrics"]["mean_tool_calls"], 4)
        self.assertGreater(result["metrics"]["total_simulated_cost_usd"], 0)
        self.assertIsNone(result["metrics"]["total_actual_cost_usd"])
        self.assertTrue(all(len(case["errors"]) == 5 for case in result["cases"]))

    def test_model_metadata_and_identity_are_required(self):
        scenarios = [s for s in self.scenarios if s["split"] == "validation"]
        with self.assertRaises(ValueError):
            evaluate(scenarios, decide, "sft")
        metadata = {"checkpoint_id": "actual-checkpoint", "model": "test-model", "revision": "test-revision",
                    "template_version": "test", "thinking_mode": "disabled", "decoding": {"temperature": 0}}
        result = evaluate(scenarios, decide, "sft", metadata)
        self.assertEqual(result["metrics"]["decision_accuracy"], 0)
        self.assertTrue(all(case["errors"] for case in result["cases"]))

    def test_http_policy_uses_private_bearer_and_checks_identity(self):
        import io
        from unittest.mock import patch
        from research.evaluation.run import http_policy
        response = {"action": "finish", "arguments": {"disposition": "unresolved"}, "policy_id": "sft", "checkpoint_id": "test-checkpoint"}
        with patch("urllib.request.urlopen", return_value=io.StringIO(json.dumps(response))) as opened:
            policy = http_policy("http://localhost/policy/decide", "sft", 1, token="unit-test-token")
            self.assertEqual(policy({})["checkpoint_id"], "test-checkpoint")
            self.assertEqual(opened.call_args.args[0].get_header("Authorization"), "Bearer unit-test-token")
        response["policy_id"] = "rules"
        with patch("urllib.request.urlopen", return_value=io.StringIO(json.dumps(response))):
            with self.assertRaises(ValueError):
                policy({})


if __name__ == "__main__":
    unittest.main()
