"""Create controlled scenarios. All names, people, claims and prices are synthetic."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OBSERVED = "2026-10-30T19:30:00Z"
KINDS = ["easy_positive", "stale_employment", "conflicting_location", "catch_all_allowed",
         "excluded_industry", "excluded_region", "former_employee", "invalid_contact",
         "unavailable_source", "conflicting_employment"]


def opaque(value: str, prefix: str = "") -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build_scenario(group: dict) -> dict:
    kind, name, domain = group["fixture_kind"], group["company_name"], group["domain"]
    gid = group["group_id"]
    sid = lambda key: opaque(gid + key, "s_")
    person = ["Avery Chen", "Morgan Patel", "Jordan Kim", "Taylor Ross", "Casey Silva", "Riley Park"][group["replica"]]
    email = "research-buyer@" + domain
    actions = {
        "overview": {"action": "inspect_source", "arguments": {"source_id": sid("overview")}, "title": "Company overview", "source_kind": "official_profile"},
        "person": {"action": "verify_person", "arguments": {"person_name": person, "company_domain": domain}, "title": "Current employment check"},
        "contact": {"action": "verify_contact", "arguments": {"email": email}, "title": "Mailbox verification"},
        "search": {"action": "search", "arguments": {"query": domain + " headquarters official"}, "title": "Find an official headquarters record"},
        "registry": {"action": "inspect_source", "arguments": {"source_id": sid("registry")}, "title": "Current company registry", "source_kind": "company_registry"},
        "discovery": {"action": "search", "arguments": {"query": domain + " company profile"}, "title": "Search for company profiles"},
        "archive": {"action": "inspect_source", "arguments": {"source_id": sid("archive")}, "title": "Company news archive", "source_kind": "news_archive"},
    }
    facts, rows = [], []

    def fact(field, value, source="overview", current=True, authority=2):
        return {"field": field, "value": value, "source_id": sid(source), "current": current, "authority": authority}

    def row(key, excerpt, values, next_actions=(), title=None, status="ok"):
        source = {"id": sid(key), "url": f"https://{domain}/{key}",
                  "title": title or {"overview": "Company overview", "person": "Current employment check",
                                    "contact": "Controlled contact verification", "search": "Public-source search",
                                    "registry": "Current company registry", "discovery": "Company profile search", "archive": "Company news archive"}[key],
                  "excerpt": excerpt, "observed_at": OBSERVED, "provenance": "synthetic"}
        observation = {"status": status, "sources": [source], "facts": values,
                       "available_actions": [actions[a] for a in next_actions]}
        rows.append({**actions[key], "observation": observation,
                     "cost_usd": {"overview": .01, "person": .02, "contact": .015, "search": .005, "registry": .01, "discovery": .005, "archive": .01}[key],
                     "latency_ms": 800 if key == "search" else 1200, "observation_tokens": 250})
        facts.extend(values)

    all_rules = {"industry": ["B2B SaaS"], "region": ["US"], "employment_status": ["current"], "contact_status": ["valid"]}
    reject_rules = {"industry": ["Consumer retail"], "region": ["Germany", "Canada"],
                    "employment_status": ["former"], "contact_status": ["invalid"]}
    industry = "Consumer retail" if kind == "excluded_industry" else "B2B SaaS"
    region = "Germany" if kind == "excluded_region" else "US"
    base = [fact("industry", industry), fact("region", region), fact("person_name", person),
            fact("person_role", "VP Sales"), fact("email", email)]
    if kind == "unavailable_source":
        row("overview", "The source is unavailable. No company or buyer facts could be read.",
            [fact("source_status", "unavailable")], status="unavailable")
        expected, required_fields, required_sources = "unresolved", {"source_status": "unavailable"}, [sid("overview")]
    elif kind == "conflicting_location":
        base.append(fact("region", "Canada", authority=2))
        base.append(fact("employment_status", "current"))
        row("overview", f"{name} sells B2B SaaS. Two same-date records disagree on US versus Canada headquarters. {person} is VP Sales. A current registry can resolve location.", base, ["search"])
        row("search", "An official current company registry was found; inspect it to read the location.", [], ["registry"])
        row("registry", f"The current official company registry places {name} headquarters in the US.",
            [fact("region", "US", "registry", authority=3)], ["contact"])
        row("contact", "Controlled mailbox check: valid. This is a synthetic vendor result.", [fact("contact_status", "valid", "contact")])
        expected, required_fields, required_sources = "qualified", {"industry": industry, "region": "US", "employment_status": "current", "contact_status": "valid"}, [sid("overview"), sid("registry"), sid("contact")]
    else:
        delayed_person = kind in {"stale_employment", "former_employee", "conflicting_employment"}
        if not delayed_person:
            base.append(fact("employment_status", "current"))
        else:
            base.append(fact("employment_status", "current", current=False, authority=1))
        if kind == "catch_all_allowed":
            all_rules["contact_status"] = ["valid", "catch_all"]
        row("overview", f"{name} operates in {industry} and lists {region} headquarters. {person} is listed as VP Sales. " +
            ("The employment listing is dated 2023 and requires a current check." if delayed_person else "The current team page confirms employment.") +
            " The contact address has not been verified.", base, ["person", "contact"] if delayed_person and kind != "conflicting_employment" else ["person"] if delayed_person else ["contact"])
        if delayed_person:
            if kind == "conflicting_employment":
                row("person", "Two current sources of equal authority disagree: one lists current employment and one lists former employment. No stronger source is available.",
                    [fact("employment_status", "current", "person"), fact("employment_status", "former", "person"),
                     fact("employment_verification", "conflicting", "person")])
            else:
                employment = "former" if kind == "former_employee" else "current"
                row("person", f"The current team record confirms {person} is a {employment} employee.",
                    [fact("employment_status", employment, "person")], ["contact"])
        if kind != "conflicting_employment":
            contact = "catch_all" if kind == "catch_all_allowed" else "invalid" if kind == "invalid_contact" else "valid"
            row("contact", "Controlled mailbox check: " + contact + ". A catch-all result does not verify an individual mailbox.",
                [fact("contact_status", contact, "contact")])
        if kind in {"excluded_industry", "excluded_region", "former_employee", "invalid_contact"}:
            field = {"excluded_industry": "industry", "excluded_region": "region", "former_employee": "employment_status", "invalid_contact": "contact_status"}[kind]
            value = {"excluded_industry": industry, "excluded_region": region, "former_employee": "former", "invalid_contact": "invalid"}[kind]
            expected, required_fields = "rejected", {field: value}
            required_sources = [sid("person" if kind == "former_employee" else "contact" if kind == "invalid_contact" else "overview")]
        elif kind == "conflicting_employment":
            expected, required_fields, required_sources = "unresolved", {"employment_verification": "conflicting"}, [sid("person")]
        else:
            expected = "qualified"
            required_fields = {"industry": industry, "region": region, "employment_status": "current", "contact_status": contact}
            required_sources = [sid("overview"), sid("contact")] + ([sid("person")] if delayed_person else [])
    # Plausible extra research competes for the same budget. Its content contains
    # no qualification labels, and is invisible until that action is executed.
    row("discovery", "Search located a company overview and an older company news archive. Open a result to inspect its content.", [], ["overview", "archive"])
    row("archive", "The archive discusses an old logo refresh and a community event. It does not establish product category, current headquarters, employment or mailbox status.", [])
    for item in rows:
        if item["arguments"] == actions["overview"]["arguments"]:
            item["observation"]["available_actions"].append(actions["archive"])
    return {
        "id": opaque(gid + ":scenario", "c_"), "group_id": gid, "split": group["split"], "provenance": "synthetic",
        "icp": {"industries": ["B2B SaaS"], "regions": ["US"], "company_size": "Any",
                "buyer_roles": ["VP Sales"], "exclusions": ["Consumer retail", "Non-US headquarters", "Former buyer", "Invalid mailbox"],
                "required_evidence": ["Current company and buyer evidence", "Explicit contact verification result"],
                "qualification_rules": {"all": all_rules, "reject_if": reject_rules}},
        "initial_observations": [{"company_name": name, "domain": domain,
                                  "task": "Assess this prospect against the approved ICP using inspected evidence. Check current employment and contact status when needed.",
                                  "available_actions": [actions["overview"], actions["discovery"]]}],
        "tool_results": rows,
        "hidden_reference": {"disposition": expected, "fields": required_fields, "required_evidence_ids": required_sources,
                             "label_basis": "Controlled fixture truth; not a claim about a real company.", "fixture_kind": kind},
        "budget_usd": .10, "max_steps": 4, "max_observation_tokens": 1600, "max_simulated_time_ms": 6000,
    }


def main():
    manifest_path = ROOT / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        groups = []
        for i, kind in enumerate(KINDS):
            splits = (["train"] * 4 + ["validation", "test"] if i < 6 else
                      ["train"] * 3 + ["validation"] * (2 if i < 8 else 1) + ["test"] * (1 if i < 8 else 2))
            ordered = sorted(range(6), key=lambda n: opaque(f"split-v1:{kind}:{n}"))
            for n, split in zip(ordered, splits):
                token = opaque(f"company-v1:{kind}:{n}")[:8]
                groups.append({"group_id": opaque(token, "g_"), "domain": f"company-{token}.example",
                               "company_name": "Company " + token.upper(), "split": split, "fixture_kind": kind, "replica": n})
        manifest = {"version": 1, "frozen_at": OBSERVED, "method": "Company groups fixed before scenario generation; stratified deterministic allocation.", "groups": groups}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    scenarios = [build_scenario(group) for group in manifest["groups"]]
    write_jsonl(ROOT / "scenarios.jsonl", scenarios)
    report = {"scenario_count": len(scenarios), "group_count": len({s["group_id"] for s in scenarios}),
              "split_counts": dict(Counter(s["split"] for s in scenarios)),
              "disposition_counts": dict(Counter(s["hidden_reference"]["disposition"] for s in scenarios)),
              "provenance": "synthetic", "template_count": len(KINDS),
              "limitations": "Six controlled variations per template. Template reuse across splits limits generalization claims. No production trajectories or real company labels are in this file."}
    (ROOT / "fixture_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
