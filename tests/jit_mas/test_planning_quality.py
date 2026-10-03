"""Contract and prompt regressions, not evidence of real-model quality gains."""

import copy
import json

import pytest

from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import LocalPlan, Prediction, PublicTask


def prediction():
    return Prediction.model_validate({
        "graph": {"rubrics": [{"rubric_id": "accuracy", "requirement": "Check claims and assumptions",
                                "source": "inferred", "confidence": 0.7}]},
        "candidates": [
            {"agent_id": "author", "role": "Writer", "capability": "technical exposition",
             "rubric_ids": ["accuracy"], "max_calls": 1},
            {"agent_id": "check", "role": "Reviewer", "capability": "claim verification",
             "rubric_ids": ["accuracy"], "depends_on": ["author"], "max_calls": 1},
        ],
    })


def review_plan(**changes):
    return {"agent_id": "check", "capability": "claim verification", "rubric_ids": ["accuracy"],
            "depends_on": ["author"], "required_inputs": ["The author's complete draft and derivation"],
            "expected_outputs": ["Claim-specific counterexamples and corrections"], "tools": [],
            "max_calls": 1, **changes}


def test_local_plan_correction_names_all_invalid_tools_and_dependencies_without_rewriting():
    task = PublicTask(task_id="local", question="Explain a technical result accurately")
    draft = prediction()
    invalid = review_plan(tools=["reasoning", "web_search"],
                          depends_on=["author", "check", "author", "Writer", "accuracy"])
    original = copy.deepcopy(invalid)
    calls = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        if "response_correction" not in payload:
            return json.dumps(invalid)
        correction = payload["response_correction"]
        assert json.loads(correction["previous_response"]) == original
        error = correction["validation_errors"][0]["message"]
        for detail in (
            "unavailable_tools=['reasoning', 'web_search']", "allowed_tools=[]", "requires tools=[]",
            "unknown_dependency_ids=['Writer', 'accuracy']", "self_dependency=True",
            "duplicate_dependency_ids=['author']", "allowed_dependency_ids=['author']",
            "required_inputs", "uncovered/challenge",
        ):
            assert detail in error
        return json.dumps(review_plan())

    analyzer = GlobalAnalyzer(model)
    result = analyzer.local_plan(task, draft, draft.candidates[1])
    assert result.depends_on == ["author"]
    assert result.tools == []
    assert result.required_inputs == original["required_inputs"]
    assert invalid == original
    assert len(calls) == len(analyzer.call_records) == 2
    assert calls[0] == {key: value for key, value in calls[1].items() if key != "response_correction"}
    assert json.loads(analyzer.call_records[0]["response"]) == original
    assert "validation_errors" in analyzer.call_records[0]
    assert "validation_errors" not in analyzer.call_records[1]


@pytest.mark.parametrize("changes,error", [
    ({"tools": ["writing"]}, "unavailable_tools=\\['writing'\\]"),
    ({"depends_on": ["Writer"]}, "unknown_dependency_ids=\\['Writer'\\]"),
    ({"depends_on": ["author draft"]}, "unknown_dependency_ids=\\['author draft'\\]"),
    ({"depends_on": ["accuracy"]}, "unknown_dependency_ids=\\['accuracy'\\]"),
    ({"depends_on": ["check"]}, "self_dependency=True"),
    ({"depends_on": ["author", "author"]}, "duplicate_dependency_ids=\\['author'\\]"),
])
def test_repeated_invalid_local_contract_is_rejected_after_only_one_correction(changes, error):
    invalid = review_plan(**changes)
    original = copy.deepcopy(invalid)
    analyzer = GlobalAnalyzer(lambda messages: json.dumps(invalid))
    draft = prediction()
    with pytest.raises(ValueError, match=error):
        analyzer.local_plan(PublicTask(task_id="t", question="Explain"), draft, draft.candidates[1])
    assert invalid == original
    assert len(analyzer.call_records) == 2
    assert all(json.loads(row["response"]) == original for row in analyzer.call_records)
    assert all(row["validation_errors"] for row in analyzer.call_records)


def test_available_tools_and_dependency_order_are_preserved_exactly():
    draft = prediction()
    draft.candidates.append(draft.candidates[0].model_copy(update={"agent_id": "source"}))
    value = review_plan(tools=["lookup"], depends_on=["source", "author"])
    analyzer = GlobalAnalyzer(lambda messages: json.dumps(value))
    task = PublicTask(task_id="t", question="Compare claims", tools=["lookup", "calculator"])
    result = analyzer.local_plan(task, draft, draft.candidates[1])
    assert result.tools == ["lookup"]
    assert result.depends_on == ["source", "author"]
    assert len(analyzer.call_records) == 1


def test_reviewer_correction_identifies_required_primary_artifact_dependency():
    draft = prediction()
    invalid = {
        "graph": draft.graph.model_dump(mode="json"),
        "team": {"agents": [agent.model_dump(mode="json") for agent in draft.candidates],
                 "synthesizer_id": "check", "coverage": {"accuracy": ["author", "check"]},
                 "primary": {"accuracy": "check"}, "reviewers": {"accuracy": ["author"]}},
    }
    original = copy.deepcopy(invalid)
    requests = []

    def model(messages):
        payload = json.loads(messages[1]["content"])
        requests.append(payload)
        if len(requests) == 2:
            correction = payload["response_correction"]
            assert json.loads(correction["previous_response"]) == original
            message = correction["validation_errors"][0]["message"]
            for detail in ("rubric_id='accuracy'", "reviewer_id='author'", "primary_owner_id='check'",
                           "reviewer_ancestor_ids=[]", "backward dependency or cycle"):
                assert detail in message
            fixed = copy.deepcopy(invalid)
            fixed["team"]["primary"] = {"accuracy": "author"}
            fixed["team"]["reviewers"] = {"accuracy": ["check"]}
            return json.dumps(fixed)
        return json.dumps(invalid)

    analyzer = GlobalAnalyzer(model)
    result = analyzer.reconcile(PublicTask(task_id="t", question="Explain accurately"), draft, [])
    assert result.team.primary == {"accuracy": "author"}
    assert result.team.reviewers == {"accuracy": ["check"]}
    assert result.team.agents[1].depends_on == ["author"]
    assert invalid == original
    assert len(requests) == len(analyzer.call_records) == 2


@pytest.mark.parametrize("mode", ["single_pass", "iterative_shared_ledger"])
@pytest.mark.parametrize("fenced", [False, True])
def test_terminal_owner_correction_removes_all_optional_upstream_reviews(mode, fenced):
    draft = prediction()
    draft.graph.rubrics.append(draft.graph.rubrics[0].model_copy(update={"rubric_id": "clarity"}))
    for candidate in draft.candidates:
        candidate.rubric_ids = ["accuracy", "clarity"]
        if mode == "iterative_shared_ledger":
            candidate.max_calls = None
    invalid = {"graph": draft.graph.model_dump(mode="json"), "team": {
        "execution_mode": mode, "agents": [agent.model_dump(mode="json") for agent in draft.candidates],
        "synthesizer_id": "check", "coverage": {rid: ["author", "check"] for rid in ("accuracy", "clarity")},
        "primary": {rid: "check" for rid in ("accuracy", "clarity")},
        "reviewers": {rid: ["author"] for rid in ("accuracy", "clarity")},
        "total_max_calls": None if mode == "iterative_shared_ledger" else 2,
    }}
    observed = []

    def model(messages):
        request = json.loads(messages[1]["content"])
        observed.append(request)
        assert "If primary[rubric_id] == synthesizer_id" in messages[0]["content"]
        if "response_correction" not in request:
            content = json.dumps(invalid)
            return "```json\n" + content + "\n```" if fenced else content
        assert "primary_owner_id='check'" in str(request["response_correction"])
        assert "rubric_id='accuracy'" in str(request["response_correction"])
        assert "rubric_id='clarity'" in str(request["response_correction"])
        audit = request["response_correction"]["assignment_audit"]
        assert audit["synthesizer_id"] == "check"
        assert audit["dependencies"] == {"author": [], "check": ["author"]}
        assert audit["terminal_owner_reviews_to_remove_if_synthesizer_unchanged"] == [
            {"rubric_id": "accuracy", "required_reviewers": []},
            {"rubric_id": "clarity", "required_reviewers": []}]
        assert "audit every actual rubric_id" in messages[0]["content"]
        assert "omit every optional reviewer assignment" in messages[0]["content"]
        fixed = copy.deepcopy(invalid)
        fixed["team"]["reviewers"] = {"accuracy": [], "clarity": []}
        return json.dumps(fixed)

    result = GlobalAnalyzer(model, execution_mode=mode,
                            total_max_calls=invalid["team"]["total_max_calls"]).reconcile(
        PublicTask(task_id="terminal-review", question="Explain accurately and clearly"), draft, [])
    assert result.team.primary == invalid["team"]["primary"]
    assert result.team.reviewers == {"accuracy": [], "clarity": []}
    assert result.team.agents[1].depends_on == ["author"]
    assert len(observed) == 2


def test_missing_producer_can_be_proposed_as_a_gap_without_inventing_dependency():
    draft = prediction()
    value = review_plan(depends_on=[], required_inputs=["A source record not yet provided"],
                        uncovered=["No candidate produces independently checked source records"],
                        challenge="Reconcile a source-checking capability or reduce the unsupported claim")
    analyzer = GlobalAnalyzer(lambda messages: json.dumps(value))
    result = analyzer.local_plan(PublicTask(task_id="t", question="Explain"), draft, draft.candidates[1])
    assert result == LocalPlan.model_validate(value)
    assert result.challenge and result.uncovered


@pytest.mark.parametrize("actual_id", ["editor", "changed-editor"])
def test_role_misfilled_as_capability_gets_exact_immutable_identity_correction(actual_id):
    draft = prediction()
    candidate = draft.candidates[1].model_copy(update={
        "agent_id": "editor", "role": "Report Formatting and Quality Assurance",
        "capability": "Review and refine the drafted report for structure, conciseness, clarity, "
                      "and adherence to technical report standards.",
    })
    draft.candidates[1] = candidate
    expected = {"agent_id": "editor", "capability": candidate.capability}
    invalid = review_plan(agent_id=actual_id, capability=candidate.role)
    original = copy.deepcopy(invalid)

    def model(messages):
        payload = json.loads(messages[1]["content"])
        assert payload["immutable_identity"] == expected
        assert payload["candidate"]["role"] != payload["immutable_identity"]["capability"]
        assert "Copy immutable_identity.agent_id and immutable_identity.capability verbatim" in messages[0]["content"]
        if "response_correction" not in payload:
            return json.dumps(invalid)
        correction = payload["response_correction"]
        assert json.loads(correction["previous_response"]) == original
        error = correction["validation_errors"][0]["message"]
        assert "expected_identity=" + json.dumps(expected) in error
        assert "actual_identity=" + json.dumps({"agent_id": actual_id, "capability": candidate.role}) in error
        assert "candidate.role is not candidate.capability" in error
        return json.dumps(review_plan(**payload["immutable_identity"]))

    analyzer = GlobalAnalyzer(model)
    result = analyzer.local_plan(PublicTask(task_id="identity", question="Write a technical report"),
                                 draft, candidate)
    assert result.agent_id == candidate.agent_id
    assert result.capability == candidate.capability
    assert invalid == original
    assert len(analyzer.call_records) == 2
    assert json.loads(analyzer.call_records[0]["response"]) == original
    assert "validation_errors" not in analyzer.call_records[1]


def test_repeated_role_capability_mismatch_is_rejected_without_silent_repair():
    draft = prediction()
    candidate = draft.candidates[1]
    value = review_plan(capability=candidate.role)
    original = copy.deepcopy(value)
    analyzer = GlobalAnalyzer(lambda messages: json.dumps(value))
    with pytest.raises(ValueError, match="expected_identity=.*actual_identity"):
        analyzer.local_plan(PublicTask(task_id="identity", question="Explain"), draft, candidate)
    assert value == original
    assert len(analyzer.call_records) == 2
    assert all(json.loads(row["response"]) == original for row in analyzer.call_records)
    assert all(row["validation_errors"] for row in analyzer.call_records)


@pytest.mark.parametrize("allowed", [[], ["lookup"]])
def test_prediction_unavailable_tools_correction_includes_exact_allowlist(allowed):
    invalid = prediction().model_dump(mode="json")
    invalid["candidates"][0]["tools"] = ["calculator", "reasoning"]
    original = copy.deepcopy(invalid)

    def model(messages):
        payload = json.loads(messages[1]["content"])
        if "response_correction" not in payload:
            return json.dumps(invalid)
        correction = payload["response_correction"]
        error = correction["validation_errors"][0]["message"]
        assert "requested_tools=['calculator', 'reasoning']" in error
        assert "unavailable_tools=['calculator', 'reasoning']" in error
        assert f"allowed_tools={allowed}" in error
        assert "empty allowlist requires tools=[]" in error
        assert json.loads(correction["previous_response"]) == original
        corrected = copy.deepcopy(invalid)
        corrected["candidates"][0]["tools"] = allowed
        return json.dumps(corrected)

    analyzer = GlobalAnalyzer(model)
    result = analyzer.predict(PublicTask(task_id="tools", question="Explain", tools=allowed))
    assert result.candidates[0].tools == allowed
    assert invalid == original
    assert len(analyzer.call_records) == 2
    assert json.loads(analyzer.call_records[0]["response"]) == original


def test_local_planning_uses_shared_task_applicability_for_every_experience_bank(monkeypatch):
    draft = prediction()
    task = PublicTask(task_id="t", question="Explain the assumptions behind a result")
    experiences = [{"experience_id": str(i), "bank": bank, "applicable": applicable}
                   for i, (bank, applicable) in enumerate([
                       ("rubric", True), ("rubric", False), ("organization", True),
                       ("organization", False), ("execution", True), ("execution", False)])]
    before = copy.deepcopy(experiences)
    decisions = []

    def applicability(entry, public_task, *, capability=None):
        decisions.append((copy.deepcopy(entry), public_task, capability))
        return {"matched": entry["applicable"], "reason": "fixture", "signal_matches": [],
                "task_grounded": entry["applicable"]}

    monkeypatch.setattr("jit_mas.planning.experience_applicability", applicability)

    def model(messages):
        payload = json.loads(messages[1]["content"])
        assert [entry["experience_id"] for entry in payload["experiences"]] == ["0", "2", "4"]
        return json.dumps(review_plan())

    analyzer = GlobalAnalyzer(model)
    analyzer.local_plan(task, draft, draft.candidates[1], experiences)
    assert len(decisions) == len(experiences)
    assert all(public_task is task and capability == "claim verification"
               for entry, public_task, capability in decisions)
    assert experiences == before


def test_all_planning_prompts_preserve_uncertainty_and_require_single_pass_ledger():
    draft = prediction()
    team = {"agents": [a.model_dump(mode="json") for a in draft.candidates],
            "synthesizer_id": "check", "coverage": {"accuracy": ["author", "check"]},
            "primary": {"accuracy": "author"}, "reviewers": {"accuracy": ["check"]},
            "total_max_calls": 6, "max_parallel": 1}
    prompts = {}

    def model(messages):
        payload = json.loads(messages[1]["content"])
        phase = payload["phase"]
        prompt = " ".join(messages[0]["content"].split("conforming to this JSON Schema:")[0].split())
        prompts[phase] = prompt
        if phase == "predict":
            return draft.model_dump_json()
        if phase == "local_plan":
            candidate = payload["candidate"]
            return json.dumps({"agent_id": candidate["agent_id"], "capability": candidate["capability"],
                               "rubric_ids": candidate["rubric_ids"],
                               "depends_on": candidate["depends_on"], "max_calls": 1})
        return json.dumps({"graph": draft.graph.model_dump(mode="json"), "team": team})

    analyzer = GlobalAnalyzer(model, max_agents=2, max_parallel=1, total_max_calls=6)
    result = analyzer.build(PublicTask(task_id="t", question="Explain the claims and assumptions"))
    assert "fallible planning hypotheses" in prompts["predict"]
    assert "Only a requirement actually stated in the public task may be labeled explicit" in prompts["predict"]
    assert "factual correctness takes precedence over satisfying a mistaken prediction" in prompts["predict"]
    for phase in ("predict", "local_plan", "reconcile"):
        assert "AgentSpec.max_calls" in prompts[phase]
        assert "formula" in prompts[phase]
        assert "uncertainty" in prompts[phase]
        assert "ledger" in prompts[phase]
        assert "tools=[]" in prompts[phase]
        for field in ("requirements", "outline", "evidence_spans", "source_references"):
            assert field in prompts[phase]
    for detail in ("tools must be []", "exact agent_id values", "missing",
                   "actual draft or derivation", "agreement between agents is not evidence",
                   "downstream reviewer later", "implicit second execution or backward edge"):
        assert detail in prompts["local_plan"]
    for detail in ("exactly one model call per selected role", "no JSON correction",
                   "Unused team.total_max_calls is a ceiling", "single-pass consumer",
                   "source producer as an ancestor", "incorporated future reviewer feedback",
                   "reviewer may reject a mistaken rubric premise"):
        assert detail in prompts["reconcile"]
    # Prompt guidance is not a hidden rule forcing a fixed team or spending the entire budget.
    assert [a.agent_id for a in result.team.agents] == ["author", "check"]
    assert sum(a.max_calls for a in result.team.agents) == 2 < result.team.total_max_calls == 6
    assert result.graph.rubrics[0].confidence == 0.7
    assert result.graph.rubrics[0].source == "inferred"


@pytest.mark.parametrize("phase", ["predict", "local_plan", "reconcile"])
def test_new_plans_reject_extra_execution_calls_without_silent_normalization(phase):
    draft = prediction()
    task = PublicTask(task_id="single-pass", question="Explain the claims")
    team = {"agents": [a.model_dump(mode="json") for a in draft.candidates],
            "synthesizer_id": "check", "coverage": {"accuracy": ["author", "check"]},
            "primary": {"accuracy": "author"}, "total_max_calls": 8}
    if phase == "predict":
        response = draft.model_dump(mode="json")
        response["candidates"][0]["max_calls"] = 2
    elif phase == "local_plan":
        response = review_plan(max_calls=2)
    else:
        team["agents"][-1]["max_calls"] = 2
        response = {"graph": draft.graph.model_dump(mode="json"), "team": team}
    original = copy.deepcopy(response)
    analyzer = GlobalAnalyzer(lambda _: json.dumps(response))
    with pytest.raises(ValueError, match="single-pass execution requires .*max_calls=1"):
        if phase == "predict":
            analyzer.predict(task)
        elif phase == "local_plan":
            analyzer.local_plan(task, draft, draft.candidates[1])
        else:
            analyzer.reconcile(task, draft, [])
    assert response == original
    assert len(analyzer.call_records) == 2
    assert all(json.loads(call["response"]) == original for call in analyzer.call_records)


def test_final_writer_cannot_plan_an_external_tool_batch():
    draft = prediction()
    agents = [a.model_dump(mode="json") for a in draft.candidates]
    agents[-1]["tools"] = ["lookup"]
    response = {"graph": draft.graph.model_dump(mode="json"), "team": {
        "agents": agents, "synthesizer_id": "check", "coverage": {"accuracy": ["author", "check"]},
        "primary": {"accuracy": "author"}}}
    original = copy.deepcopy(response)
    analyzer = GlobalAnalyzer(lambda _: json.dumps(response))
    with pytest.raises(ValueError, match="Final Writer check must have tools=\\[\\]"):
        analyzer.reconcile(PublicTask(task_id="writer", question="Compare sources", tools=["lookup"]),
                           draft, [])
    assert response == original


def test_contributor_tool_batch_remains_allowed_with_tool_free_writer():
    draft = prediction()
    agents = [a.model_dump(mode="json") for a in draft.candidates]
    agents[0]["tools"] = ["lookup"]
    response = {"graph": draft.graph.model_dump(mode="json"), "team": {
        "agents": agents, "synthesizer_id": "check", "coverage": {"accuracy": ["author", "check"]},
        "primary": {"accuracy": "author"}}}
    result = GlobalAnalyzer(lambda _: json.dumps(response)).reconcile(
        PublicTask(task_id="contributor", question="Compare sources", tools=["lookup"]), draft, [])
    assert result.team.agents[0].tools == ["lookup"]
    assert result.team.agents[-1].tools == []
