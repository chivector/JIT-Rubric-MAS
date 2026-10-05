"""Deterministic budget recommendations are public-task based and auditable."""

from jit_mas.agent_pool import seed_pool
from jit_mas.planning import ADAPTIVE_BUDGET_VERSION, GlobalAnalyzer, adaptive_budget_profile
from jit_mas.schemas import PublicTask


def test_simple_task_prefers_one_role_and_one_call():
    profile = adaptive_budget_profile(PublicTask(task_id="simple", question="Explain the result"),
                                     max_agents=4, total_max_calls=16)
    assert profile["version"] == ADAPTIVE_BUDGET_VERSION
    assert profile["complexity_band"] == "simple"
    assert profile["preferred_max_agents"] == 1
    assert profile["preferred_total_max_calls"] == 1
    assert profile["signals"]["constraint_count"] == 0


def test_complex_task_retains_configured_roster_and_records_signals():
    task = PublicTask(
        task_id="complex",
        question=("Research and compare alternatives across countries and periods, "
                   "calculate a reproducible table, and cite evidence for every claim. " * 8),
        constraints=["include limitations", "cover each requested country", "show assumptions"],
        attachments=["brief", "data"], tools=["search"], capabilities=["research", "review"],
    )
    profile = adaptive_budget_profile(task, max_agents=4, total_max_calls=16)
    assert profile["complexity_band"] == "complex"
    assert profile["preferred_max_agents"] == 4
    assert profile["preferred_total_max_calls"] == 12
    assert "evidence" in profile["signals"]["complexity_terms"]


def test_task_adaptive_profile_is_present_in_every_planning_payload():
    task = PublicTask(task_id="audit", question="Compare evidence for two alternatives")
    seen = []

    def model(messages):
        import json
        payload = json.loads(messages[1]["content"])
        seen.append(payload["limits"]["adaptive_budget"])
        return json.dumps({"graph": {"rubrics": []}, "candidates": [{
            "agent_id": "writer", "role": "Writer", "capability": "writing",
            "rubric_ids": [], "max_calls": 1, "max_tokens": 128,
        }]})

    GlobalAnalyzer(model, max_agents=4, total_max_calls=16).predict(task)
    assert seen and seen[0]["version"] == ADAPTIVE_BUDGET_VERSION
    assert seen[0]["preferred_max_agents"] == 2


def test_only_evolving_pool_enforces_preferred_roster_cap():
    import json

    task = PublicTask(task_id="easy", question="Explain the result")
    requests = []

    def model(messages, **kwargs):
        requests.append((json.loads(messages[1]["content"]), kwargs))
        return json.dumps({"graph": {"rubrics": []}, "candidates": [{
            "agent_id": "writer", "role": "Writer", "capability": "writing",
            "rubric_ids": [], "max_calls": 1, "max_tokens": 128,
            "pool_agent_id": "writer", "pool_agent_version": 1,
        }]})

    pooled = GlobalAnalyzer(model, max_agents=3, agent_pool=seed_pool())
    pooled.predict(task)
    assert requests[0][0]["limits"]["max_agents"] == 1
    assert requests[0][0]["limits"]["adaptive_budget"]["enforced"] is True

    baseline_requests = []

    def baseline_model(messages, **kwargs):
        baseline_requests.append(json.loads(messages[1]["content"]))
        return json.dumps({"graph": {"rubrics": []}, "candidates": [{
            "agent_id": "writer", "role": "Writer", "capability": "writing",
            "rubric_ids": [], "max_calls": 1, "max_tokens": 128,
        }]})

    GlobalAnalyzer(baseline_model, max_agents=3).predict(task)
    assert baseline_requests[0]["limits"]["max_agents"] == 3
    assert "adaptive_budget" in baseline_requests[0]["limits"]
    assert baseline_requests[0]["limits"]["adaptive_budget"]["enforced"] is False
