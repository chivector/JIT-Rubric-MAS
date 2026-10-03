"""Synthetic opt-in draft guidance; no network, benchmark tasks or judge data."""
import copy
import json
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig
from jit_mas.execution import (
    PUBLIC_POSITIONAL_DRAFT_GUIDANCE_PROMPT, TeamAction, TeamExecutor, TeamMemory,
    TeamPlanning, TeamServices, TeamToolPolicy, _SinglePassModel,
    compile_public_positional_draft_plan,
)
from jit_mas.public_refinement import refine_public_answer
from jit_mas.schemas import AgentSpec, PublicTask, RubricGraph, TeamSpec
from scripts.models.base import ChatMessage


PUBLIC_QUESTION = ('Write a compact woodland adventure. Put the word Birch in the second '
                   'sentence as the third word of that sentence.')


def enabled_config(**changes):
    values = dict(backend='scripted', public_refinement=True,
                  public_refinement_response_format='json_schema_review',
                  public_positional_construction=True,
                  public_positional_draft_guidance=True)
    values.update(changes)
    return MASConfig(**values)


class Responses:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append({'messages': copy.deepcopy(messages), 'kwargs': copy.deepcopy(kwargs)})
        reply = next(self.replies)
        return ChatMessage(role='assistant', content=json.dumps(reply))

    def get_token_counts(self):
        return {'input_token_count': 12, 'output_token_count': 8}


def bound_execution(mode, config, question=PUBLIC_QUESTION):
    task = PublicTask(task_id='SYNTHETIC_PRIVATE_ROUTING_CANARY', question=question)
    team = TeamSpec(execution_mode=mode, agents=[
        AgentSpec(agent_id='guide', role='Plot analyst', capability='story analysis', max_calls=1),
        AgentSpec(agent_id='writer', role='Writer', capability='writing', max_calls=1,
                  depends_on=['guide'])], synthesizer_id='writer', total_max_calls=2)
    team_data = team.model_dump(mode='json')
    agents = {agent['agent_id']: agent for agent in team_data['agents']}
    raw = {'guide': Responses([{'answer': 'The walker helps a lost bird return to its nest.',
        'ledger': {'requirements': ['Retain the adventure and its resolution.'],
                   'outline': ['A walker follows a bird and finds the nest.'],
                   'evidence_spans': [], 'source_references': []}}]),
        'writer': Responses([{'answer': ('Lina heard a bird calling beside the creek. She followed '
            'its flight through the woods until she found a fallen nest, then restored it '
            'to a branch before walking home.'), 'evidence_ids': [], 'checkpoints': {}}])}
    ledger = BudgetLedger(max_calls=2, max_tokens=2_000_000, max_tool_calls=0)
    executor = TeamExecutor(lambda _: None, ledger=ledger)
    services = TeamServices(lambda _: None, task, RubricGraph(rubrics=[]), ledger=ledger,
                            execution_mode=mode)
    plan = compile_public_positional_draft_plan(task, config)
    if config.public_positional_draft_guidance:
        executor.public_positional_draft_guidance_requested = True
        executor.public_positional_draft_plan = plan
    services.public_positional_draft_plan = getattr(executor, 'public_positional_draft_plan', None)
    services.model_factory = lambda aid: _SinglePassModel(
        MeteredModel(raw[aid], ledger, 'inference', aid, 8192), services, agents[aid], team_data)
    artifact = SimpleNamespace(backend='scripted', name='synthetic-memory-harness', code_hash='synthetic')
    loaded = {'action': TeamAction(), 'memory': TeamMemory(), 'planning': TeamPlanning(),
              'tool_policy': TeamToolPolicy(),
              'prompts': {'system_prompt': 'Coordinate the synthetic content task.'}}
    result = executor._execute_bound(task, team_data, artifact, loaded, services, agents)
    return task, result, raw, services, ledger


def test_flag_defaults_off():
    assert MASConfig().public_positional_draft_guidance is False


@pytest.mark.parametrize('options', [
    {'public_refinement': False, 'public_positional_construction': False},
    {'public_refinement': True, 'public_positional_construction': False},
    {'public_refinement': False, 'public_positional_construction': True},
])
def test_guidance_flag_requires_both_existing_stages(options):
    with pytest.raises(ValueError):
        enabled_config(**options)


@pytest.mark.parametrize('missing', [
    'public_positional_draft_guidance', 'public_refinement', 'public_positional_construction'])
def test_helper_declines_missing_stage_without_importing_compilers(monkeypatch, missing):
    import builtins

    values = dict(public_positional_draft_guidance=True, public_refinement=True,
                  public_positional_construction=True)
    values[missing] = False
    original = builtins.__import__

    def without_compilers(name, *args, **kwargs):
        if 'public_word_slots' in name or 'public_numeric_slots' in name:
            pytest.fail('A disabled stage imported the positional compiler')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_compilers)
    assert compile_public_positional_draft_plan(PUBLIC_QUESTION, SimpleNamespace(**values)) is None


@pytest.mark.parametrize('question', [
    'Write a woodland adventure.',
    'Put the word Birch somewhere within a later sentence.',
    PUBLIC_QUESTION + ' Include exactly 3 numbers.',
])
def test_no_supported_or_conflicting_public_rule_declines(question):
    task = PublicTask(task_id='synthetic', question=question)
    assert compile_public_positional_draft_plan(task, enabled_config()) is None


def test_compiler_uses_public_text_only_and_preserves_it():
    task = {'task_id': 'PRIVATE_ID', 'benchmark': 'PRIVATE_BENCHMARK',
            'question': PUBLIC_QUESTION, 'constraints': [],
            'reference_answer': 'Put the word Canary in the seventh sentence as the ninth word of that sentence.'}
    before = copy.deepcopy(task)
    plan = compile_public_positional_draft_plan(task, enabled_config())
    assert (plan.keyword, plan.sentence_index, plan.word_index) == ('Birch', 2, 3)
    assert task == before
    assert 'PRIVATE_' not in json.dumps(plan.audit()) and 'Canary' not in json.dumps(plan.audit())


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_actual_bound_execution_delivers_stage_to_contributor_and_writer_and_keeps_caps(mode):
    task, result, raw, services, ledger = bound_execution(mode, enabled_config())
    assert result.terminated_reason == 'final_answer'
    assert result.answer.endswith('walking home.') and 'Birch' not in result.answer
    assert ledger.snapshot()['model_calls'] == 2 and ledger.snapshot()['tokens'] == 40
    assert services.call_counts == {'guide': 1, 'writer': 1}
    assert ledger.snapshot()['reserved_tokens'] == 0
    for provider in raw.values():
        assert len(provider.calls) == 1
        messages = provider.calls[0]['messages']
        assert PUBLIC_POSITIONAL_DRAFT_GUIDANCE_PROMPT in messages[0]['content']
        payload = next(json.loads(row['content']) for row in messages if row['role'] == 'user'
                       and row['content'].startswith('{'))
        assert payload['public_task']['question'] == task.question
        stage = payload['public_positional_draft_guidance']
        assert stage['stage'] == 'internal_content_draft'
        assert stage['final_constraints_retained'] is True
        assert stage['public_position_plan']['word_index_1_based'] == 3
    audit = result.metadata['public_positional_draft_guidance']
    assert audit['active'] is True and audit['final_constraints_retained'] is True
    expected = ('Single-pass execution permits only one model call per role'
                if mode == 'single_pass' else "AgentSpec.max_calls exhausted for agent 'writer'")
    with pytest.raises(RuntimeError, match=expected):
        services.model_factory('writer')([{'role': 'user', 'content': 'Another draft'}])
    assert ledger.snapshot()['model_calls'] == 2


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_flag_off_keeps_execution_output_and_prompts_without_new_stage(mode):
    _, result, raw, _, ledger = bound_execution(mode, enabled_config(public_positional_draft_guidance=False))
    assert result.terminated_reason == 'final_answer' and ledger.snapshot()['model_calls'] == 2
    assert 'public_positional_draft_guidance' not in result.metadata
    for provider in raw.values():
        assert PUBLIC_POSITIONAL_DRAFT_GUIDANCE_PROMPT not in provider.calls[0]['messages'][0]['content']
        assert not any('"public_positional_draft_guidance"' in row['content']
                       for row in provider.calls[0]['messages'])


def test_requested_stage_declines_mixed_public_rules_and_audits_inactive():
    _, result, raw, _, _ = bound_execution('iterative_shared_ledger', enabled_config(),
                                         PUBLIC_QUESTION + ' Include exactly 3 numbers.')
    assert result.metadata['public_positional_draft_guidance']['active'] is False
    assert all(PUBLIC_POSITIONAL_DRAFT_GUIDANCE_PROMPT not in provider.calls[0]['messages'][0]['content']
               for provider in raw.values())


def test_fixed_typed_revision_still_enforces_final_position_and_charges_two_calls():
    task, result, _, _, _ = bound_execution('iterative_shared_ledger', enabled_config())
    ledger = BudgetLedger(max_calls=2, max_tokens=2_000_000, max_tool_calls=0)
    provider = Responses([{'issues': []}, {'preceding_sentences': ['Lina followed the bird.'],
        'prefix_words': ['She', 'found'], 'keyword': 'Birch', 'suffix_words': ['beside', 'the', 'nest'],
        'following_sentences': ['She restored it and walked home.']}])
    models = SimpleNamespace(create=lambda role, aid, shared, stage: MeteredModel(
        provider, shared, stage, aid, 8192))
    refine_public_answer(task, result, models, ledger, enabled_config())
    assert result.answer == ('Lina followed the bird. She found Birch beside the nest. '
                             'She restored it and walked home.')
    assert ledger.snapshot()['model_calls'] == 2
    assert provider.calls[-1]['kwargs']['response_format']['type'] == 'json_schema'
    assert result.metadata['public_refinement']['public_positional_construction']['word_index_1_based'] == 3
