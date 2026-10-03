"""Exact, optional public task projection with synthetic content and no network."""
import copy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.execution import (
    TeamAction, TeamExecutor, TeamMemory, TeamPlanning, TeamServices, TeamToolPolicy,
    _SinglePassModel, compile_public_positional_draft_plan, content_hash,
    project_public_positional_draft_task,
)
from jit_mas.public_refinement import refine_public_answer
from jit_mas.public_word_slots import position_plan
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.models.base import ChatMessage


RULE = 'Place the word Willow as the fourth word in the third sentence'
QUESTION = 'Write a short river rescue in English. ' + RULE + '. Finish with a safe return.'


def config(**changes):
    options = dict(backend='scripted', public_refinement=True,
                   public_refinement_response_format='json_schema_review',
                   public_positional_construction=True, public_positional_draft_guidance=True,
                   public_positional_draft_projection=True)
    options.update(changes)
    return MASConfig(**options)


def task(question=QUESTION, **changes):
    return PublicTask(task_id='synthetic-projection', question=question, **changes)


def project(value, plan=None):
    if plan is None:
        plan = compile_public_positional_draft_plan(value, config())
    return project_public_positional_draft_task(value, plan)


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append({'messages': copy.deepcopy(messages), 'kwargs': copy.deepcopy(kwargs)})
        return ChatMessage(role='assistant', content=json.dumps(next(self.replies)))

    def get_token_counts(self):
        return {'input_token_count': 15, 'output_token_count': 10}


def execute(mode, settings=None, public_task=None):
    settings = settings or config()
    public_task = public_task or task()
    team = TeamSpec(execution_mode=mode, agents=[
        AgentSpec(agent_id='scout', role='Content contributor', capability='plot analysis', max_calls=1,
                  task_prompt='Retain this frozen planning text: ' + RULE),
        AgentSpec(agent_id='author', role='Writer', capability='writing', max_calls=1,
                  depends_on=['scout'], task_prompt='Retain this domain instruction: ' + RULE)],
        synthesizer_id='author', total_max_calls=2)
    team_data = team.model_dump(mode='json')
    agents = {row['agent_id']: row for row in team_data['agents']}
    original_team = copy.deepcopy(team_data)
    draft = 'A boat drifted toward rocks. Two friends reached the bank. They guided it to safety.'
    raw = {'scout': Replies([{'answer': 'Friends rescue a drifting boat without entering the rapids.',
        'ledger': {'requirements': ['Finish with a safe return.'],
                   'outline': ['Find a rope, secure the boat and return safely.'],
                   'evidence_spans': [], 'source_references': []}}]),
        'author': Replies([{'answer': draft, 'evidence_ids': [], 'checkpoints': {}}])}
    ledger = BudgetLedger(max_calls=4, max_tokens=2_000_000, max_tool_calls=0)
    executor = TeamExecutor(lambda _: None, ledger=ledger)
    services = TeamServices(lambda _: None, public_task, RubricGraph(rubrics=[]),
                            ledger=ledger, execution_mode=mode)
    if settings.public_positional_draft_guidance:
        executor.public_positional_draft_guidance_requested = True
        executor.public_positional_draft_plan = compile_public_positional_draft_plan(public_task, settings)
    executor.public_positional_draft_projection_requested = settings.public_positional_draft_projection
    services.public_positional_draft_plan = getattr(executor, 'public_positional_draft_plan', None)
    services.public_positional_draft_projection_requested = settings.public_positional_draft_projection
    services.model_factory = lambda aid: _SinglePassModel(
        MeteredModel(raw[aid], ledger, 'inference', aid, 8192), services, agents[aid], team_data)
    loaded = {'action': TeamAction(), 'memory': TeamMemory(), 'planning': TeamPlanning(),
              'tool_policy': TeamToolPolicy(), 'prompts': {'system_prompt': 'Coordinate content.'}}
    artifact = SimpleNamespace(backend='scripted', name='synthetic-projection-harness', code_hash='synthetic')
    result = executor._execute_bound(public_task, team_data, artifact, loaded, services, agents)
    assert team_data == original_team
    return public_task, result, raw, services, ledger


def payload(call):
    return next(json.loads(row['content']) for row in call['messages']
                if row['role'] == 'user' and row['content'].startswith('{'))


def test_projection_defaults_off():
    assert MASConfig().public_positional_draft_projection is False


@pytest.mark.parametrize('missing', [
    'public_refinement', 'public_positional_construction', 'public_positional_draft_guidance'])
def test_projection_requires_all_three_stages(missing):
    with pytest.raises(ValueError, match='projection requires'):
        config(**{missing: False})


def test_exact_slicing_preserves_all_other_text_metadata_and_original():
    value = task('河岸。 ' + QUESTION, constraints=['Keep 90 words and a hopeful tone.', RULE + '.'],
                 attachments=['PUBLIC_SOURCE_CANARY'], capabilities=['river safety'])
    original = value.model_dump(mode='json')
    plan = compile_public_positional_draft_plan(value, config())
    projected, audit = project(value, plan)
    expected = copy.deepcopy(original)
    expected['question'] = '河岸。 Write a short river rescue in English. . Finish with a safe return.'
    expected['constraints'][1] = '.'
    assert projected == expected and value.model_dump(mode='json') == original
    assert projected is not original
    assert audit['active'] and audit['final_constraints_retained']
    assert audit['projected_task_hash'] == content_hash(expected)
    assert audit['original_task_hash'] == content_hash(original)
    assert len(audit['span_hashes']) == 2
    assert all(row['text_sha256'] == hashlib.sha256(RULE.encode()).hexdigest()
               for row in audit['span_hashes'])
    assert RULE not in json.dumps(audit)
    projected['constraints'][0] = 'mutated copy'
    projected['attachments'].append('other')
    assert value.model_dump(mode='json') == original


def test_repeated_identical_rule_uses_offsets_in_each_public_source():
    question = 'Keep the plot. ' + RULE + '. Retain the ending. ' + RULE + '.'
    value = task(question, constraints=[RULE, 'Keep the boat and the rope.', RULE])
    projected, audit = project(value)
    assert audit['active'] and len(audit['span_hashes']) == 4
    assert projected['question'] == 'Keep the plot. . Retain the ending. .'
    assert projected['constraints'] == ['', 'Keep the boat and the rope.', '']


@pytest.mark.parametrize('literal', ['"Willow"', "'Willow'", '“Willow”', '‘Willow’', '`Willow`'])
def test_keyword_literal_quotes_are_not_outer_quotations(literal):
    quoted_rule = RULE.replace('Willow', literal)
    projected, audit = project(task('Write a rescue. ' + quoted_rule + '.'))
    assert audit['active'] and projected['question'] == 'Write a rescue. .'


@pytest.mark.parametrize('question', [
    'Describe the river.',
    'Place the word Willow somewhere later in the story.',
    QUESTION + ' Include exactly 6 numbers.',
    QUESTION + ' Place the word Rock as the second word in the first sentence.',
])
def test_unsupported_mixed_or_conflicting_task_is_unchanged(question):
    value = task(question)
    projected, audit = project(value)
    assert not audit['active'] and projected == value.model_dump(mode='json')


@pytest.mark.parametrize('question', [
    'Do not ' + RULE.lower() + '.',
    'Only if needed ' + RULE.lower() + '.',
    RULE + ', and discuss why this is just an example.',
    'Example: ' + RULE + '.',
    'Sample:\n' + RULE + '.',
    'Example: A river. ' + RULE + '.',
    'Examples:\nA river.\n' + RULE + '.',
    RULE + '. This rule applies only if a poem is requested.',
    'An excerpt. "A river. ' + RULE + '."',
    'An excerpt. “A river. ' + RULE + '.”',
    '"A river.\n' + RULE + '.\nA rope."',
    '```text\nA river.\n' + RULE + '.\n```',
    '~~~\nA river.\n' + RULE + '.\n~~~',
    '> A river. ' + RULE + '.',
    'Please ' + RULE.lower() + '.',
])
def test_quoted_negative_conditional_or_scoped_clause_declines_every_span(question):
    value = task(question, constraints=[RULE + '.'])
    projected, audit = project(value)
    assert not audit['active'] and projected == value.model_dump(mode='json')
    assert audit['span_hashes'] == []


@pytest.mark.parametrize('changes', [
    {'source': 'reference_answer'}, {'source': 'constraints[-1]'}, {'source': 'constraints[01]'},
    {'source': 'constraints[4]'}, {'start': -1}, {'end': 9999}, {'start': True},
    {'start': 1.5}, {'text': 'a different public clause'},
])
def test_invalid_span_source_range_or_text_cannot_delete_anything(changes, monkeypatch):
    value = task(constraints=[RULE + '.'])
    plan = compile_public_positional_draft_plan(value, config())
    bad = replace(plan, matched_public_spans=(replace(plan.matched_public_spans[0], **changes),
                                             plan.matched_public_spans[1]))
    # Even a faulty/stale compiler result must pass source/range/text validation.
    monkeypatch.setattr('jit_mas.public_word_slots.position_plan', lambda _: bad)
    projected, audit = project(value, bad)
    assert not audit['active'] and projected == value.model_dump(mode='json')


def test_stale_task_or_modified_plan_declines_and_copies_original():
    value = task()
    plan = compile_public_positional_draft_plan(value, config())
    value.question = 'A new preface. ' + value.question
    projected, audit = project(value, plan)
    assert not audit['active'] and projected == value.model_dump(mode='json')
    assert project(value, replace(plan, keyword='Cedar'))[1]['active'] is False


def test_duplicate_or_overlapping_spans_decline(monkeypatch):
    value = task()
    plan = compile_public_positional_draft_plan(value, config())
    bad = replace(plan, matched_public_spans=plan.matched_public_spans * 2)
    monkeypatch.setattr('jit_mas.public_word_slots.position_plan', lambda _: bad)
    projected, audit = project(value, bad)
    assert not audit['active'] and projected == value.model_dump(mode='json')


def test_public_compiler_exception_declines_without_exposing_error_text(monkeypatch):
    value = task()
    plan = compile_public_positional_draft_plan(value, config())

    def failing_compiler(_):
        raise RuntimeError('PRIVATE_ERROR_CANARY')

    monkeypatch.setattr('jit_mas.public_word_slots.position_plan', failing_compiler)
    projected, audit = project(value, plan)
    assert not audit['active'] and projected == value.model_dump(mode='json')
    assert 'PRIVATE_ERROR_CANARY' not in json.dumps(audit)


def test_private_fields_cannot_activate_projection_and_are_never_rewritten():
    value = {'question': 'Write a rescue story.', 'constraints': ['Keep it in English.'],
             'reference_answer': RULE, 'private_rubric': {'text': RULE},
             'other_public_metadata': {'text': RULE}}
    before = copy.deepcopy(value)
    projected, audit = project(value)
    assert not audit['active'] and projected == before and value == before
    value['question'] += ' ' + RULE + '.'
    projected, audit = project(value)
    assert audit['active']
    assert projected['reference_answer'] == value['reference_answer']
    assert projected['private_rubric'] == value['private_rubric']
    assert projected['other_public_metadata'] == value['other_public_metadata']


@pytest.mark.parametrize('question', [RULE, RULE + '.', ' \n' + RULE + '! \n'])
def test_rule_only_question_keeps_original_instead_of_an_empty_draft_task(question):
    value = task(question)
    projected, audit = project(value)
    assert not audit['active'] and projected == value.model_dump(mode='json')


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_both_actual_modes_project_only_task_field_keep_caps_and_original_final_task(mode):
    private_record = {'reference': 'PRIVATE_REFERENCE_CANARY', 'criteria': 'PRIVATE_RUBRIC_CANARY'}
    private_before = copy.deepcopy(private_record)
    value = task(constraints=['Keep a hopeful ending.'], attachments=['PUBLIC_EVIDENCE_CANARY', RULE])
    original = value.model_dump(mode='json')
    value, result, raw, services, ledger = execute(mode, public_task=value)
    assert result.terminated_reason == 'final_answer' and result.answer.endswith('safety.')
    assert ledger.snapshot()['model_calls'] == 2 and ledger.snapshot()['tokens'] == 50
    assert services.call_counts == {'scout': 1, 'author': 1}
    assert value.model_dump(mode='json') == original and services.public_task is value
    audit = result.metadata['public_positional_draft_projection']
    assert audit['active'] and audit['final_constraints_retained']
    assert result.metadata['public_positional_draft_guidance']['public_position_plan']['keyword'] == 'Willow'
    for provider in raw.values():
        assert len(provider.calls) == 1
        instruction = payload(provider.calls[0])
        assert instruction['public_task']['question'] == 'Write a short river rescue in English. . Finish with a safe return.'
        assert instruction['public_task']['constraints'] == original['constraints']
        assert instruction['public_task']['attachments'] == original['attachments']
        assert RULE in instruction['agent']['task_prompt']
        assert instruction['public_positional_draft_guidance'] == {
            'version': 'public-positional-internal-draft-v1', 'stage': 'internal_content_draft',
            'final_constraints_retained': True, 'projection_hash': audit['projected_task_hash']}
        assert 'PRIVATE_' not in json.dumps(provider.calls)
    with pytest.raises(RuntimeError, match='one model call per role|max_calls exhausted'):
        services.model_factory('author')([{'role': 'user', 'content': 'Another draft'}])
    assert ledger.snapshot()['model_calls'] == 2
    reviewer = Replies([{'issues': []}, {
        'preceding_sentences': ['A rope reached the boat.', 'Two friends secured the line.'],
        'prefix_words': ['They', 'waited', 'under'], 'keyword': 'Willow',
        'suffix_words': ['until', 'the', 'boat', 'was', 'safe'],
        'following_sentences': ['Everyone returned home.']}])
    models = SimpleNamespace(create=lambda role, aid, shared, stage: MeteredModel(
        reviewer, shared, stage, aid, 8192))
    refine_public_answer(value, result, models, ledger, config())
    assert ledger.snapshot()['model_calls'] == 4 and ledger.snapshot()['tokens'] == 100
    assert result.answer == ('A rope reached the boat. Two friends secured the line. '
        'They waited under Willow until the boat was safe. Everyone returned home.')
    for call in reviewer.calls:
        assert original['question'] in json.dumps(call, ensure_ascii=False)
        assert 'PRIVATE_' not in json.dumps(call)
    assert private_record == private_before and value.model_dump(mode='json') == original


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_inactive_flag_preserves_actual_actor_task_and_existing_guidance(mode):
    value, result, raw, _, ledger = execute(mode, config(public_positional_draft_projection=False))
    assert ledger.snapshot()['model_calls'] == 2
    assert 'public_positional_draft_projection' not in result.metadata
    for provider in raw.values():
        instruction = payload(provider.calls[0])
        assert instruction['public_task'] == value.model_dump(mode='json')
        assert instruction['public_positional_draft_guidance']['public_position_plan']['keyword'] == 'Willow'


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_requested_but_declined_projection_keeps_full_actual_task(mode):
    value, result, raw, _, _ = execute(mode, public_task=task('Example: ' + RULE + '.'))
    assert not result.metadata['public_positional_draft_projection']['active']
    for provider in raw.values():
        assert payload(provider.calls[0])['public_task'] == value.model_dump(mode='json')


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_executor_public_boundary_keeps_frozen_task_and_coordinator_input(mode, monkeypatch):
    value = task()
    original = value.model_dump(mode='json')
    team = TeamSpec(execution_mode=mode, agents=[
        AgentSpec(agent_id='author', role='Writer', capability='writing', max_calls=1)],
        synthesizer_id='author', total_max_calls=1)
    graph = RubricGraph(rubrics=[])
    provider = Replies([{'answer': 'Friends secured the boat and returned safely.',
                         'evidence_ids': [], 'checkpoints': {}}])
    ledger = BudgetLedger(max_calls=1, max_tokens=2_000_000, max_tool_calls=0)
    executor = TeamExecutor(lambda aid: MeteredModel(provider, ledger, 'inference', aid, 8192),
                            ledger=ledger)
    executor.public_positional_draft_guidance_requested = True
    executor.public_positional_draft_plan = compile_public_positional_draft_plan(value, config())
    executor.public_positional_draft_projection_requested = True

    class RecordingAction(TeamAction):
        def run(self, supplied_task, ctx):
            self.coordinator_question = supplied_task
            return super().run(supplied_task, ctx)

    action = RecordingAction()
    monkeypatch.setattr('jit_mas.execution.load_harness', lambda _: {
        'action': action, 'memory': TeamMemory(), 'planning': TeamPlanning(),
        'tool_policy': TeamToolPolicy(), 'prompts': {'system_prompt': 'Coordinate content.'}})
    artifact = SimpleNamespace(backend='scripted', name='synthetic-projection-boundary',
        code_hash='synthetic', task_hash=content_hash(original), team_hash=content_hash(team),
        sidecar={'rubrics': graph.model_dump(mode='json'), 'experiences': []},
        verify_integrity=lambda: None)
    result = executor.execute(value, team, artifact)
    assert result.terminated_reason == 'final_answer'
    assert result.metadata['public_positional_draft_projection']['active']
    assert action.coordinator_question == original['question']
    assert action.services.public_task.model_dump(mode='json') == original
    assert artifact.task_hash == content_hash(value) and value.model_dump(mode='json') == original
    assert RULE not in payload(provider.calls[0])['public_task']['question']
    assert ledger.snapshot()['model_calls'] == 1


@pytest.mark.parametrize("prefix,quoted_keyword,suffix,active", [
    ("Preserve these instructions as a quotation exactly. ", False,
     ". End of quotation.", False),
    ("Reproduce the following passage verbatim. ", False, ".", False),
    ("> A supplied line.\nA continuation in the same paragraph.\n", False,
     ".", False),
    ("Quote your sources. ", True, ".", True),
])
def test_public_literal_context_projection(prefix, quoted_keyword, suffix, active):
    literal = '"amber"' if quoted_keyword else "amber"
    rule = (f"Include keyword {literal} in the third sentence, "
            "as the fourth word of that sentence")
    task = PublicTask(task_id="synthetic-guard-regression", question=prefix + rule + suffix,
        constraints=["Retain useful meaning and source attribution."],
        attachments=["public-note-canary"], tools=["public-tool-canary"],
        capabilities=["public-capability-canary"])
    original = copy.deepcopy(task.model_dump(mode="json"))
    plan = position_plan(task)
    assert (plan is not None) is active
    projected, audit = project_public_positional_draft_task(task, plan)
    assert audit["active"] is active
    assert task.model_dump(mode="json") == original
    expected = copy.deepcopy(original)
    if active:
        for span in reversed(plan.matched_public_spans):
            assert span.source == "question"
            assert expected["question"][span.start:span.end] == span.text
            expected["question"] = expected["question"][:span.start] + expected["question"][span.end:]
        assert audit["span_hashes"]
    else:
        assert audit["reason"]
        assert audit["span_hashes"] == []
        assert audit["projected_task_hash"] == audit["original_task_hash"]
    assert projected == expected
