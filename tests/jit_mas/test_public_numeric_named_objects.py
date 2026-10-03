"""Synthetic production-layout and real metered-refinement tests; no API/checker access."""
from __future__ import annotations

import copy
import json

import jsonschema
import pytest
from pydantic import ValidationError


from jit_mas import public_numeric_slots


@pytest.fixture(scope='module')
def numeric_slots():
    return public_numeric_slots


def task(n=4, k=None, **extra):
    rule = f'Write a fictional excursion. Include exactly {n} numbers.'
    if k is not None:
        rule += f' Use at least {k} different coordinating conjunctions.'
    return {'question': rule, 'constraints': [], **extra}


def payload(n=4, k=None):
    data = {'opening': 'We began our imagined excursion.', 'number_slots': {
        f'slot_{i}': {'before': 'We passed', 'number': str(i), 'after': 'quiet trees.'}
        for i in range(1, n + 1)}, 'closing': 'We returned home together.'}
    if k is not None:
        literals = ('and', 'but', 'or', 'so', 'yet', 'for', 'nor')[:k]
        data['conjunction_clauses'] = {word: {'before': 'The grove was quiet',
            'after': 'we remained curious.'} for word in literals}
    return data


def resolve(schema, entry):
    return schema['$defs'][entry['$ref'].split('/')[-1]] if '$ref' in entry else entry


@pytest.mark.parametrize('n', [0, 1, 4, 128])
@pytest.mark.parametrize('k', [None, 1, 7])
def test_named_boundaries_are_closed_required_objects_and_exact_local_counts(numeric_slots, n, k):
    plan = numeric_slots.numeric_plan(task(n, k), layout='named_objects')
    assert plan.layout == plan.audit()['layout'] == 'named_objects'
    schema = plan.schema
    numbers = resolve(schema, schema['properties']['number_slots'])
    assert numbers['type'] == 'object' and numbers['additionalProperties'] is False
    assert list(numbers['properties']) == [f'slot_{i}' for i in range(1, n + 1)]
    assert set(numbers.get('required', [])) == set(numbers['properties'])
    assert 'minItems' not in numbers and 'maxItems' not in numbers
    if k is not None:
        clauses = resolve(schema, schema['properties']['conjunction_clauses'])
        assert list(clauses['properties']) == list(numeric_slots.CONJUNCTION_REGISTRY[:k])
        assert set(clauses['required']) == set(clauses['properties'])
        assert clauses['additionalProperties'] is False
    data = payload(n, k)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(data)
    answer = numeric_slots.render(plan.model.model_validate(data))
    from jit_mas.public_output_metrics import public_output_metrics
    metrics = public_output_metrics(task(n, k), answer)
    assert metrics['numeric_token_diagnostics']['total_numeric_tokens'] == n
    if k is not None:
        assert metrics['fanboys_lexical_diagnostics']['different_lexical_types_present'] >= k


@pytest.mark.parametrize('layout', ['array', 'named_objects'])
def test_layout_is_public_audited_while_parser_spans_are_unchanged(numeric_slots, layout):
    source = task(6, 2, task_id='PRIVATE_ROUTE', metadata={'secret_count': 98}, score=99)
    plan = numeric_slots.numeric_plan(source, layout=layout)
    assert plan.number_count == 6 and plan.conjunction_count == 2
    for span in plan.matched_public_spans:
        assert span.source == 'question'
        assert source['question'][span.start:span.end] == span.text
    surface = json.dumps(plan.audit()) + json.dumps(plan.schema) + numeric_slots.prepare_prompt_instruction(plan)
    assert 'PRIVATE_ROUTE' not in surface and 'secret_count' not in surface


def test_default_array_layout_and_render_are_backward_compatible(numeric_slots):
    plan = numeric_slots.numeric_plan(task(4))
    assert plan.layout == 'array'
    assert plan.schema['properties']['number_slots']['minItems'] == 4
    data = payload(4)
    data['number_slots'] = list(data['number_slots'].values())
    assert numeric_slots.render(plan.model.model_validate(data)).count('quiet trees.') == 4


@pytest.mark.parametrize('bad_layout', ['object', None, '', 1])
def test_invalid_programmer_layout_fails_closed(numeric_slots, bad_layout):
    with pytest.raises((ValueError, TypeError)):
        numeric_slots.numeric_plan(task(), layout=bad_layout)


@pytest.mark.parametrize('field', ['number_slots', 'conjunction_clauses'])
@pytest.mark.parametrize('change', ['missing', 'extra', 'array', 'null'])
def test_exact_object_members_reject_omissions_additions_and_wrong_container(numeric_slots, field, change):
    plan = numeric_slots.numeric_plan(task(4, 3), layout='named_objects')
    data = payload(4, 3)
    if change == 'missing':
        del data[field][next(iter(data[field]))]
    elif change == 'extra':
        data[field]['unexpected'] = next(iter(data[field].values()))
    elif change == 'array':
        data[field] = list(data[field].values())
    else:
        data[field] = None
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, plan.schema)


@pytest.mark.parametrize('field', ['before', 'number', 'after'])
def test_incomplete_number_entry_never_passes_first_object_validation(numeric_slots, field):
    plan = numeric_slots.numeric_plan(task(4), layout='named_objects')
    data = payload(4)
    del data['number_slots']['slot_2'][field]
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


def test_order_is_numeric_and_registry_order_even_when_returned_object_order_changes(numeric_slots):
    plan = numeric_slots.numeric_plan(task(12, 3), layout='named_objects')
    data = payload(12, 3)
    data['number_slots'] = dict(reversed(list(data['number_slots'].items())))
    data['conjunction_clauses'] = dict(reversed(list(data['conjunction_clauses'].items())))
    answer = numeric_slots.render(plan.model.model_validate(data))
    import re
    assert re.findall(r'\b\d+\b', answer) == [str(i) for i in range(1, 13)]
    assert answer.index('quiet and we') < answer.index('quiet but we') < answer.index('quiet or we')


@pytest.mark.parametrize('n,k', [(0, None), (0, 2), (4, None), (128, 7)])
def test_sdk_strict_transformation_preserves_required_named_object_fields(numeric_slots, n, k):
    from openai.lib._pydantic import to_strict_json_schema
    plan = numeric_slots.numeric_plan(task(n, k), layout='named_objects')
    transformed = to_strict_json_schema(plan.model)
    numbers = resolve(transformed, transformed['properties']['number_slots'])
    assert numbers['required'] == list(numbers['properties'])
    assert numbers['additionalProperties'] is False
    jsonschema.validate(payload(n, k), transformed)
    if n:
        data = payload(n, k)
        del data['number_slots'][f'slot_{n}']
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(data, transformed)


@pytest.mark.parametrize('digit', ['9', '\u0662', '\uff13', '\U0001d7dc'])
@pytest.mark.parametrize('location', ['opening', 'closing', 'number_before', 'number_after', 'conjunction_before', 'conjunction_after'])
def test_unicode_decimal_digits_outside_integer_field_are_locally_rejected(numeric_slots, digit, location):
    plan = numeric_slots.numeric_plan(task(4, 2), layout='named_objects')
    data = payload(4, 2)
    if location in {'opening', 'closing'}:
        data[location] += digit
    elif location.startswith('number_'):
        data['number_slots']['slot_1'][location.split('_')[1]] += digit
    else:
        data['conjunction_clauses']['and'][location.split('_')[1]] += digit
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize('number', ['-', '+', '01', '3.2', '2e3', '\u0663', '4\n', 4, None])
def test_named_integer_fields_remain_strict_and_canonical(numeric_slots, number):
    plan = numeric_slots.numeric_plan(task(4), layout='named_objects')
    data = payload(4)
    data['number_slots']['slot_2']['number'] = number
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize('k', [None, 2])
def test_zero_numeric_entries_are_closed_and_complete_artifact_is_still_required(numeric_slots, k):
    plan = numeric_slots.numeric_plan(task(0, k), layout='named_objects')
    data = payload(0, k)
    data['opening'] = data['closing'] = ''
    if k is None:
        with pytest.raises(ValidationError, match='nonempty'):
            plan.model.model_validate(data)
    else:
        assert numeric_slots.render(plan.model.model_validate(data)).strip()
    data['number_slots']['slot_1'] = {'before': 'We saw', 'number': '1', 'after': 'tree.'}
    with pytest.raises(ValidationError):
        plan.model.model_validate(data)


@pytest.mark.parametrize('mutation', ['missing_number_slot', 'missing_conjunction', 'illegal_digit', 'empty_zero_artifact'])
def test_renderer_revalidates_constructed_and_mutated_named_models(numeric_slots, mutation):
    n = 0 if mutation == 'empty_zero_artifact' else 4
    k = None if n == 0 else 2
    plan = numeric_slots.numeric_plan(task(n, k), layout='named_objects')
    parsed = plan.model.model_validate(payload(n, k))
    if mutation == 'missing_number_slot':
        parsed.number_slots.__dict__.pop('slot_3')
    elif mutation == 'missing_conjunction':
        parsed.conjunction_clauses.__dict__.pop('but')
    elif mutation == 'illegal_digit':
        parsed.number_slots.slot_1.__dict__['number'] = '2.4'
    else:
        parsed.__dict__['opening'] = parsed.__dict__['closing'] = ''
    with pytest.raises(ValidationError):
        numeric_slots.render(parsed)
    malformed = plan.model.model_construct(**{'opening': 'A start.', 'number_slots': {}, 'closing': ''})
    if n:
        with pytest.warns(UserWarning, match='Pydantic serializer warnings'):
            with pytest.raises(ValidationError):
                numeric_slots.render(malformed)


def test_single_object_strict_json_never_accepts_prefix_of_extra_data(numeric_slots):
    from jit_mas.public_refinement import _strict_json
    plan = numeric_slots.numeric_plan(task(4), layout='named_objects')
    good = json.dumps(payload(4))
    with pytest.raises(json.JSONDecodeError):
        _strict_json(good + '  "number_slots": {}', plan.model)
    with pytest.raises(json.JSONDecodeError):
        _strict_json(good + good, plan.model)


def test_prompt_uses_exact_dynamic_fields_and_denies_extra_roots(numeric_slots):
    plan = numeric_slots.numeric_plan(task(6, 3), layout='named_objects')
    instruction = numeric_slots.prepare_prompt_instruction(plan)
    assert 'slot_1, slot_2, slot_3, slot_4, slot_5, slot_6' in instruction
    assert 'and, but, or' in instruction
    assert 'never an array' in instruction and 'no answer field' in instruction
    assert 'stop immediately after its closing brace' in instruction
    assert 'remote schema enforcement is not assumed' in instruction
    assert 'slot_7' not in instruction


@pytest.mark.parametrize('rule', [
    'Include exactly 129 numbers.',
    'Include exactly 2 positive numbers.',
    'Include exactly 2 numbers. Include exactly 3 numbers.',
    'Include exactly 2 numbers. Include keyword moss in the fourth sentence as the seventh word.',
])
def test_unsupported_public_rules_decline_identically_for_both_layouts(numeric_slots, rule):
    public = {'question': rule, 'constraints': []}
    assert numeric_slots.numeric_plan(public) is None
    assert numeric_slots.numeric_plan(public, layout='named_objects') is None


class _LocalResponses:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def __call__(self, messages, **kwargs):
        from scripts.models.base import ChatMessage
        self.requests.append({'messages': copy.deepcopy(messages), 'kwargs': copy.deepcopy(kwargs)})
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return ChatMessage(role='assistant', content=value if isinstance(value, str) else json.dumps(value))

    def get_token_counts(self):
        return {'input_token_count': 30, 'output_token_count': 20}


class _LocalModels:
    def __init__(self, replies):
        self.responses = _LocalResponses(replies)
        self.created = []

    def create(self, role, agent_id, ledger, stage):
        from jit_mas.budget import MeteredModel
        self.created.append((role, agent_id, ledger, stage))
        return MeteredModel(self.responses, ledger, stage, agent_id, 16000)


def _refiner_setup(mode='json_schema_review', public=None,
                   numeric=True, positional=False):
    from jit_mas.config import MASConfig, ModelConfig
    from jit_mas.schemas import PublicTask
    from scripts.kernel.types import RunResult
    public = public or task(4, 2)
    public['task_id'] = 'PRIVATE_ROUTE_CANARY'
    result = RunResult(answer='An earlier fictional draft.', terminated_reason='final_answer',
        metadata={'private_feedback': 'PRIVATE_FEEDBACK_CANARY'})
    cfg = MASConfig(backend='scripted', public_refinement=True,
        public_refinement_response_format=mode, public_numeric_construction=numeric,
        public_positional_construction=positional, public_numeric_construction_layout='named_objects',
        models={'exec': ModelConfig(max_tokens=4096), 'global': ModelConfig(max_tokens=16000)})
    return PublicTask.model_validate(public), result, cfg


@pytest.mark.parametrize('mode', ['json_schema', 'json_schema_review'])
def test_real_refinement_named_revision_has_two_calls_shared_charge_audit_and_no_private_input(mode):
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    public, result, cfg = _refiner_setup(mode)
    ledger = BudgetLedger(None, 2_000_000, 0)
    ticket = ledger.reserve('inference', 'earlier-writer', 25, 15)
    ledger.settle(ticket, 25, 15)
    models = _LocalModels([{'issues': []}, payload(4, 2)])
    refine_public_answer(public, result, models, ledger, cfg)
    assert len(models.responses.requests) == 2
    assert all(shared is ledger for _, _, shared, _ in models.created)
    assert ledger.snapshot()['model_calls'] == 3 and ledger.snapshot()['tokens'] == 140
    assert ledger.snapshot()['reserved_tokens'] == 0
    request = models.responses.requests[-1]
    requested = request['kwargs']['response_format']
    assert requested['type'] == 'json_schema' and requested['json_schema']['strict'] is True
    schema = requested['json_schema']['schema']
    number_container = resolve(schema, schema['properties']['number_slots'])
    assert number_container['type'] == 'object'
    assert number_container['required'] == ['slot_1', 'slot_2', 'slot_3', 'slot_4']
    assert 'answer' not in schema['properties']
    assert request['kwargs']['max_tokens'] == 4096
    system = request['messages'][0]['content']
    assert 'Return the complete finished artifact in answer.' not in system
    assert 'Keep all process commentary, internal IDs and review notes outside answer.' not in system
    assert 'Return the complete finished artifact using only the specified construction fields.' in system
    assert 'Keep all process commentary, internal IDs and review notes outside the rendered artifact.' in system
    assert 'PRIVATE_ROUTE_CANARY' not in json.dumps(request['messages'])
    assert 'PRIVATE_FEEDBACK_CANARY' not in json.dumps(request['messages'])
    audit = result.metadata['public_refinement']
    assert audit['status'] == 'completed' and len(audit['budget_records']) == 2
    assert audit['public_numeric_construction']['layout'] == 'named_objects'
    assert audit['revision_public_diagnostics']['numeric_token_diagnostics']['total_numeric_tokens'] == 4
    assert audit['revision'] == payload(4, 2)


@pytest.mark.parametrize('bad', ['missing_slot', 'wrong_array', 'bare_extra_field', 'second_root', 'illegal_number'])
def test_real_refinement_named_failure_is_charged_without_retry_prefix_selection_or_draft_submission(bad):
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    public, result, cfg = _refiner_setup()
    original = result.answer
    data = payload(4, 2)
    if bad == 'missing_slot':
        del data['number_slots']['slot_3']
    elif bad == 'wrong_array':
        data['number_slots'] = list(data['number_slots'].values())
    elif bad == 'illegal_number':
        data['number_slots']['slot_3']['number'] = '-'
    else:
        root = json.dumps(data)
        data = root + (' "number_slots": {}' if bad == 'bare_extra_field' else root)
    ledger = BudgetLedger(None, 2_000_000, 0)
    models = _LocalModels([{'issues': []}, data, payload(4, 2)])
    with pytest.raises(ValueError):
        refine_public_answer(public, result, models, ledger, cfg)
    assert len(models.responses.requests) == 2
    assert ledger.snapshot()['model_calls'] == 2 and ledger.snapshot()['tokens'] == 100
    assert ledger.snapshot()['reserved_tokens'] == 0
    assert result.answer == original  # exception prevents submission; never a fallback
    audit = result.metadata['public_refinement']
    assert audit['status'] == 'failed' and audit['revision'] is None
    assert audit['calls'][-1]['status'] == 'failed'


def test_real_refinement_named_revision_budget_block_happens_before_provider_dispatch():
    from jit_mas.budget import BudgetExceeded, BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    public, result, cfg = _refiner_setup()
    ledger = BudgetLedger(1, 2_000_000, 0)
    models = _LocalModels([{'issues': []}, payload(4, 2)])
    with pytest.raises(BudgetExceeded):
        refine_public_answer(public, result, models, ledger, cfg)
    assert len(models.responses.requests) == ledger.snapshot()['model_calls'] == 1


@pytest.mark.parametrize('numeric,positional', [(True, False), (False, True), (True, True)])
def test_real_refinement_mixed_family_guards_prevent_either_compiler_activation(numeric, positional):
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    public = task(4)
    public['question'] += ' Include keyword moss in the third sentence as the sixth word.'
    public, result, cfg = _refiner_setup(public=public,
        numeric=numeric, positional=positional)
    models = _LocalModels([{'issues': []}, {'answer': 'A complete ordinary fictional artifact.'}])
    refine_public_answer(public, result, models, BudgetLedger(None, 2_000_000, 0), cfg)
    assert models.responses.requests[-1]['kwargs']['response_format'] == {'type': 'json_object'}
    audit = result.metadata['public_refinement']
    for name, enabled in [('public_numeric_construction', numeric), ('public_positional_construction', positional)]:
        if enabled:
            assert audit[name]['active'] is False
            assert audit[name]['reason'] == 'Mixed public numeric and positional rules'


@pytest.mark.parametrize('branch', ['named_objects', 'array', 'positional'])
def test_real_construction_requests_use_one_schema_protocol_without_ordinary_answer_requirement(branch):
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    if branch == 'positional':
        public = {'question': 'Write a fictional woodland account. Include keyword moss in the third sentence, as the sixth word of that sentence.',
                  'constraints': []}
        public, result, cfg = _refiner_setup(public=public, numeric=False, positional=True)
        revision = {'preceding_sentences': ['Birds crossed the canopy.', 'We followed the path.'],
                    'prefix_words': ['The', 'path', 'was', 'still', 'and'], 'keyword': 'moss',
                    'suffix_words': ['covered', 'the', 'stones'], 'following_sentences': []}
    else:
        public, result, cfg = _refiner_setup()
        cfg.public_numeric_construction_layout = branch
        revision = payload(4, 2)
        if branch == 'array':
            revision['number_slots'] = list(revision['number_slots'].values())
            revision['conjunction_clauses'] = list(revision['conjunction_clauses'].values())
    models = _LocalModels([{'issues': []}, revision])
    refine_public_answer(public, result, models, BudgetLedger(None, 2_000_000, 0), cfg)
    request = models.responses.requests[-1]
    system = request['messages'][0]['content']
    assert 'Return the complete finished artifact in answer.' not in system
    assert 'Keep all process commentary, internal IDs and review notes outside answer.' not in system
    assert 'Return the complete finished artifact using only the specified construction fields.' in system
    assert request['kwargs']['response_format']['type'] == 'json_schema'
    schema = request['kwargs']['response_format']['json_schema']['schema']
    assert schema['type'] == 'object' and schema['additionalProperties'] is False
    assert 'answer' not in schema['properties']
    assert result.metadata['public_refinement']['status'] == 'completed'
    assert len(models.responses.requests) == 2


def test_ordinary_revision_keeps_original_answer_protocol_when_named_layout_is_configured_but_disabled():
    from jit_mas.budget import BudgetLedger
    from jit_mas.config import MASConfig
    from jit_mas.public_refinement import REVISION_PROMPT, refine_public_answer
    assert MASConfig().public_numeric_construction is False
    assert MASConfig().public_numeric_construction_layout == 'array'
    public, result, cfg = _refiner_setup(public={
        'question': 'Write a short fictional woodland account.', 'constraints': []}, numeric=False)
    assert cfg.public_numeric_construction_layout == 'named_objects'
    models = _LocalModels([{'issues': []}, {'answer': 'The walkers returned from the quiet grove.'}])
    refine_public_answer(public, result, models, BudgetLedger(None, 2_000_000, 0), cfg)
    request = models.responses.requests[-1]
    system = request['messages'][0]['content']
    assert system.startswith(REVISION_PROMPT)
    assert 'Return the complete finished artifact in answer.' in system
    assert 'Keep all process commentary, internal IDs and review notes outside answer.' in system
    assert 'Return the complete finished artifact using only the specified construction fields.' not in system
    assert request['kwargs']['response_format'] == {'type': 'json_object'}
    assert 'public_numeric_construction' not in result.metadata['public_refinement']


@pytest.mark.parametrize('invalid_layout', ['object', 'named', '', None])
def test_layout_configuration_rejects_invalid_names_without_enabling_construction(invalid_layout):
    from jit_mas.config import MASConfig
    with pytest.raises(ValidationError):
        MASConfig(backend='scripted', public_numeric_construction_layout=invalid_layout)
