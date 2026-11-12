"""Extra controlled duplicate-account variants; original group splits stay fixed."""
import copy
import json
from pathlib import Path

from research.data.build_fixtures import OBSERVED, opaque, write_jsonl
from research.environment import load_scenarios


def main():
    corpus = load_scenarios()
    cases = []
    for kind in ("easy_positive", "stale_employment"):
        for split in ("train", "validation", "test"):
            original = next(s for s in corpus if s["hidden_reference"]["fixture_kind"] == kind and s["split"] == split)
            scenario = copy.deepcopy(original)
            domain = scenario["initial_observations"][0]["domain"]
            source_id = opaque(scenario["group_id"] + ":campaign-record", "s_")
            scenario["id"] = opaque(scenario["id"] + ":duplicate-variant", "c_")
            scenario["icp"]["exclusions"].append("Company already present in this campaign")
            scenario["icp"]["qualification_rules"]["reject_if"]["duplicate_account"] = [True]
            action = {"action": "inspect_source", "arguments": {"source_id": source_id},
                      "title": "Existing campaign account record", "source_kind": "campaign_record"}
            scenario["tool_results"][0]["observation"]["available_actions"].append(action)
            source = {"id": source_id, "url": None, "title": "Controlled campaign account record",
                      "excerpt": "The campaign already contains a record with canonical domain " + domain + ". This is a synthetic duplicate record.",
                      "observed_at": OBSERVED, "provenance": "synthetic"}
            scenario["tool_results"].append({**action, "observation": {"status": "ok", "sources": [source], "available_actions": [],
                "facts": [{"field": "duplicate_account", "value": True, "source_id": source_id, "current": True, "authority": 3},
                          {"field": "canonical_domain", "value": domain, "source_id": source_id, "current": True, "authority": 3}]},
                "cost_usd": .005, "latency_ms": 200, "observation_tokens": 100})
            scenario["hidden_reference"] = {"disposition": "rejected", "fields": {"duplicate_account": True},
                                            "required_evidence_ids": [source_id], "fixture_kind": "duplicate_account",
                                            "label_basis": "Controlled same-domain campaign duplicate; not a real CRM record."}
            cases.append(scenario)
    folder = Path(__file__).parent / "challenges"
    folder.mkdir(exist_ok=True)
    write_jsonl(folder / "scenarios.jsonl", cases)
    (folder / "manifest.json").write_text(json.dumps({"case_count": len(cases), "existing_group_count": len({s['group_id'] for s in cases}),
                                                     "new_company_groups": 0, "type": "synthetic duplicate-account variants",
                                                     "included_in_primary_training": False,
                                                     "split_counts": {split: sum(s["split"] == split for s in cases) for split in ("train", "validation", "test")}}, indent=2) + "\n")
    print(f"Wrote {len(cases)} duplicate-account challenge variants; primary corpus unchanged.")


if __name__ == "__main__":
    main()
