"""Declared construction transport changes no local validation or task accounting.

Providers are local fixtures; no benchmark input, private checker or API is used.
"""

import copy
import json
import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_refinement import refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


class Responses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, messages, **kwargs):
        self.requests.append({'messages': copy.deepcopy(messages), 'kwargs': copy.deepcopy(kwargs)})
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        content = response if isinstance(response, str) else json.dumps(response)
        return ChatMessage(role='assistant', content=content)

    def get_token_counts(self):
        return {'input_token_count': 30, 'output_token_count': 20}


class Models:
    def __init__(self, responses):
        self.provider = Responses(responses)
        self.created = []

    def create(self, role, agent_id, ledger, stage):
        self.created.append((role, agent_id, ledger, stage))
        return MeteredModel(self.provider, ledger, stage, agent_id, 16000)


def scenario(kind, *, mode='json_schema_review', transport='json_object'):
    positional = kind == 'position'
    question = ('Write a woodland account. Put the word Birch in the second sentence as '
                'the third word of that sentence.' if positional else
                'Write a woodland account. Include exactly 2 numbers. '
                'Use at least 2 different coordinating conjunctions.')
    task = PublicTask(task_id='PRIVATE_ROUTING_CANARY', question=question)
    config_options = dict(backend='scripted', public_refinement=True,
        public_refinement_response_format=mode, public_positional_construction=positional,
        public_numeric_construction=not positional,
        public_numeric_construction_layout='named_objects' if kind == 'named' else 'array',
        models={'exec': ModelConfig(max_tokens=4096), 'global': ModelConfig(max_tokens=16000, timeout=37)})
    if transport is not None:
        config_options['public_construction_response_format'] = transport
    config = MASConfig(**config_options)
    draft = 'A walker followed a lost bird through the woods and helped it reach its nest.'
    result = RunResult(answer=draft, terminated_reason='final_answer',
        metadata={'private_score': 'PRIVATE_METADATA_CANARY', 'call_counts': {'writer': 1}})
    if positional:
        revision = {'preceding_sentences': ['A walker entered the wood.'],
            'prefix_words': ['They', 'found'], 'keyword': 'Birch',
            'suffix_words': ['beside', 'the', 'creek'],
            'following_sentences': ['The bird returned home.']}
        expected = 'A walker entered the wood. They found Birch beside the creek. The bird returned home.'
    else:
        numbers = [{'before': 'The walker saw', 'number': '8', 'after': 'birds beside the creek.'},
                   {'before': 'They counted', 'number': '42', 'after': 'stones along the trail.'}]
        clauses = [{'before': 'The bird settled', 'after': 'the walker waited.'},
                   {'before': 'Rain began', 'after': 'the walker stayed calm.'}]
        revision = {'opening': 'A walker entered the wood.', 'number_slots': numbers,
                    'conjunction_clauses': clauses, 'closing': 'They returned home.'}
        if kind == 'named':
            # Returned JSON key order must not replace declared rendering order.
            revision['number_slots'] = {'slot_2': numbers[1], 'slot_1': numbers[0]}
            revision['conjunction_clauses'] = {'but': clauses[1], 'and': clauses[0]}
        expected = ('A walker entered the wood. The walker saw 8 birds beside the creek. '
                    'They counted 42 stones along the trail. The bird settled and the walker waited. '
                    'Rain began but the walker stayed calm. They returned home.')
    return task, config, result, revision, expected


@pytest.mark.parametrize('kind', ['position', 'array', 'named'])
@pytest.mark.parametrize('mode', ['json_schema', 'json_schema_review'])
def test_constructed_json_object_uses_two_global_calls_same_ledger_and_strict_rendering(kind, mode):
    task, config, result, revision, expected = scenario(kind, mode=mode)
    models = Models([{'issues': []}, revision])
    ledger = BudgetLedger(None, 2_000_000, 0, timeout_seconds=900)
    earlier = ledger.reserve('inference', 'writer', 1, 2)
    ledger.settle(earlier, 1, 2)
    audits = []
    refine_public_answer(task, result, models, ledger, config, audit_writer=audits.append)
    assert result.answer == expected
    assert len(models.provider.requests) == 2
    assert [(role, agent_id, stage) for role, agent_id, _, stage in models.created] == [
        ('global', 'public-review', 'inference'), ('global', 'public-revision', 'inference')]
    assert all(shared is ledger for _, _, shared, _ in models.created)
    assert ledger.snapshot()['model_calls'] == 3 and ledger.snapshot()['tokens'] == 103
    assert ledger.snapshot()['reserved_tokens'] == 0
    first, second = models.provider.requests
    assert first['kwargs']['response_format']['type'] == 'json_schema'
    assert second['kwargs']['response_format'] == {'type': 'json_object'}
    for request in (first, second):
        assert request['kwargs']['max_tokens'] == 4096
        assert 0 < request['kwargs']['timeout'] <= 37
        assert 'frequency_penalty' not in request['kwargs']
        assert 'PRIVATE_METADATA_CANARY' not in json.dumps(request['messages'])
        assert 'PRIVATE_ROUTING_CANARY' not in json.dumps(request['messages'])
    # The strict local schema is still supplied even with object transport.
    schema = json.loads(second['messages'][0]['content'].rsplit('\n', 1)[-1])
    assert schema['additionalProperties'] is False
    assert 'answer' not in schema['properties']
    audit = result.metadata['public_refinement']
    assert audit['status'] == 'completed' and audit['revision'] == revision
    assert audit['calls'][1]['response_format'] == {'type': 'json_object'}
    assert audit['calls'][1]['response_format_hash'] == digest({'type': 'json_object'})
    assert audit['calls'][1]['model_call_options'] == second['kwargs']
    assert audit['calls'][1]['model_call_options_hash'] == digest(second['kwargs'])
    assert len(audit['budget_records']) == 2
    assert audits[-1]['audit_hash'] == audit['audit_hash']
    assert audit['draft'].startswith('A walker followed')
    assert result.metadata['call_counts'] == {'writer': 1}


@pytest.mark.parametrize('kind', ['position', 'array', 'named'])
def test_default_construction_transport_retains_requested_strict_schema(kind):
    task, config, result, revision, _ = scenario(kind, transport=None)
    assert MASConfig().public_construction_response_format == 'json_schema'
    models = Models([{'issues': []}, revision])
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), config)
    assert len(models.provider.requests) == 2
    for request in models.provider.requests:
        assert request['kwargs']['response_format']['type'] == 'json_schema'
        assert request['kwargs']['response_format']['json_schema']['strict'] is True


@pytest.mark.parametrize('kind', ['position', 'array', 'named'])
@pytest.mark.parametrize('invalidity', ['duplicate_json_key', 'ordinary_answer', 'blank_context', 'invalid_shape'])
def test_object_transport_invalid_raw_is_charged_once_no_fallback_or_draft_submission(kind, invalidity):
    task, config, result, revision, _ = scenario(kind)
    draft = result.answer
    invalid = copy.deepcopy(revision)
    if invalidity == 'duplicate_json_key':
        invalid = '{"opening":"One.","opening":"Two."}'
    elif invalidity == 'ordinary_answer':
        invalid = {'answer': 'An ordinary answer cannot replace the chosen construction.'}
    elif invalidity == 'blank_context':
        if kind == 'position':
            invalid['prefix_words'][0] = ' '
        elif kind == 'array':
            invalid['number_slots'][0]['before'] = '   '
        else:
            invalid['number_slots']['slot_1']['before'] = '   '
    else:
        if kind == 'position':
            invalid['prefix_words'] = ['OnlyOne']
        elif kind == 'array':
            invalid['number_slots'].pop()
        else:
            del invalid['number_slots']['slot_2']
    models = Models([{'issues': []}, invalid, revision])
    ledger = BudgetLedger(None, 2_000_000, 0)
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, config)
    assert result.answer == draft  # Failure propagates; draft is not submitted.
    assert len(models.provider.requests) == 2
    assert ledger.snapshot()['model_calls'] == 2 and ledger.snapshot()['tokens'] == 100
    assert ledger.snapshot()['reserved_tokens'] == 0
    audit = result.metadata['public_refinement']
    assert audit['status'] == 'failed' and audit['review'] == {'issues': []}
    assert audit['revision'] is None and audit['calls'][1]['status'] == 'failed'
    assert audit['calls'][1]['response'] == (invalid if isinstance(invalid, str) else json.dumps(invalid))
    assert all(request['kwargs']['response_format'] == {'type': 'json_object'}
               for request in models.provider.requests[1:])


@pytest.mark.parametrize('kind', ['array', 'named'])
def test_object_transport_still_rejects_extra_decimal_digits_in_context(kind):
    task, config, result, revision, _ = scenario(kind)
    if kind == 'array':
        revision['number_slots'][0]['after'] = 'birds after 8 steps.'
    else:
        revision['number_slots']['slot_1']['after'] = 'birds after 8 steps.'
    models = Models([{'issues': []}, revision])
    ledger = BudgetLedger(None, 2_000_000, 0)
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, config)
    assert ledger.snapshot()['model_calls'] == len(models.provider.requests) == 2
    assert result.metadata['public_refinement']['status'] == 'failed'


@pytest.mark.parametrize('mode,expected_revision', [
    ('json_schema', 'json_schema'), ('json_schema_review', 'json_object')])
@pytest.mark.parametrize('decline', ['mixed', 'unsupported'])
def test_object_override_does_not_change_ordinary_revision_when_no_plan_is_selected(mode, expected_revision, decline):
    task, config, result, _, _ = scenario('position', mode=mode)
    task.question = (task.question + ' Include exactly 2 numbers.' if decline == 'mixed' else
                     'Write a concise woodland account.')
    models = Models([{'issues': []}, {'answer': 'The walker helped the bird and returned home.'}])
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), config)
    assert len(models.provider.requests) == 2
    assert models.provider.requests[0]['kwargs']['response_format']['type'] == 'json_schema'
    assert models.provider.requests[1]['kwargs']['response_format']['type'] == expected_revision
    audit = result.metadata['public_refinement']
    assert audit['public_positional_construction']['active'] is False
    assert audit['status'] == 'completed'


def test_object_option_with_all_construction_flags_off_keeps_ordinary_object_mode():
    task, config, result, _, _ = scenario('position')
    config.public_positional_construction = False
    config.public_refinement_response_format = 'json_object'
    models = Models([{'issues': []}, {'answer': 'A coherent complete woodland account.'}])
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), config)
    assert len(models.provider.requests) == 2
    assert all(request['kwargs']['response_format'] == {'type': 'json_object'}
               for request in models.provider.requests)


def test_transport_failure_does_not_switch_format_or_dispatch_a_retry():
    task, config, result, revision, _ = scenario('named')
    models = Models([{'issues': []}, RuntimeError('Synthetic transport failure'), revision])
    ledger = BudgetLedger(None, 2_000_000, 0)
    with pytest.raises(RuntimeError, match='Synthetic transport failure'):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.provider.requests) == ledger.snapshot()['model_calls'] == 2
    assert ledger.snapshot()['reserved_tokens'] == 0
    assert ledger.snapshot()['records'][-1]['estimated'] is True
    assert models.provider.requests[-1]['kwargs']['response_format'] == {'type': 'json_object'}
    assert result.metadata['public_refinement']['status'] == 'failed'


def test_invalid_review_stops_before_any_constructed_revision_dispatch():
    task, config, result, revision, _ = scenario('named')
    models = Models(['{"issues":[],"issues":[]}', revision])
    ledger = BudgetLedger(None, 2_000_000, 0)
    with pytest.raises(ValueError, match='Duplicate JSON keys'):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.provider.requests) == ledger.snapshot()['model_calls'] == 1
    assert models.provider.requests[0]['kwargs']['response_format']['type'] == 'json_schema'
    assert result.metadata['public_refinement']['review'] is None


def test_unknown_construction_transport_is_rejected_at_configuration():
    with pytest.raises(ValueError):
        MASConfig(public_construction_response_format='fallback')
