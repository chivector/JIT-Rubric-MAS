"""Generic offline template construction; no benchmark/checker/client calls."""
import copy
import json
import re

import jsonschema
import pytest
from pydantic import ValidationError


from jit_mas import public_numeric_template


@pytest.fixture(scope='module')
def templates():
    return public_numeric_template


CLAUSES = [
    'The grove was still, and the walkers listened.',
    'The sky darkened, but the walkers continued.',
    'They could cross the stream, or they could follow the bank.',
    'Mist covered the trail, so they rested.',
    'The journey was long, yet they kept moving.',
    'They paused, for the bridge was unsafe.',
    'They did not disturb the birds, nor did they break branches.',
]


def public(n=4, k=None, **extras):
    question = f'Write a fictional woodland walk. Include exactly {n} numbers.'
    if k is not None:
        question += f' Use at least {k} distinct coordinating conjunctions.'
    return {'question': question, 'constraints': [], **extras}


def data(plan):
    prose = ' '.join(f'They counted <{key}> stones beside the trail.' for key in plan.marker_keys)
    if plan.conjunction_count:
        prose += ' ' + ' '.join(CLAUSES[:plan.conjunction_count])
    return {'answer_template': prose or 'The walkers returned to their quiet home.',
            'numeric_values': {key: str(i) for i, key in enumerate(plan.marker_keys, 1)}}


def resolve(schema, prop):
    return schema['$defs'][prop['$ref'].split('/')[-1]] if '$ref' in prop else prop


@pytest.mark.parametrize('n', [0, 1, 4, 26, 27, 128])
@pytest.mark.parametrize('k', [None, 1, 7])
def test_full_template_boundary_counts_schema_and_natural_conjunctions(templates, n, k):
    plan = templates.template_plan(public(n, k))
    assert plan.number_count == n and plan.layout == plan.audit()['layout'] == 'template'
    values = resolve(plan.schema, plan.schema['properties']['numeric_values'])
    assert values['type'] == 'object' and values['additionalProperties'] is False
    assert list(values['properties']) == list(plan.marker_keys)
    assert set(values.get('required', [])) == set(plan.marker_keys)
    assert not any(char.isdecimal() for key in plan.marker_keys for char in key)
    payload = data(plan)
    jsonschema.Draft202012Validator.check_schema(plan.schema)
    jsonschema.validate(payload, plan.schema)
    answer = templates.render(plan.model.model_validate(payload))
    from jit_mas.public_output_metrics import public_output_metrics
    metrics = public_output_metrics(public(n, k), answer)
    assert metrics['numeric_token_diagnostics']['total_numeric_tokens'] == n
    if k:
        assert metrics['fanboys_lexical_diagnostics']['different_lexical_types_present'] >= k
        assert all(clause in answer for clause in CLAUSES[:k])
    assert '<NUM_' not in answer


def test_alphabetic_marker_suffixes_do_not_have_prefix_collision(templates):
    plan = templates.template_plan(public(27))
    assert plan.marker_keys[0] == 'NUM_A'
    assert plan.marker_keys[25] == 'NUM_Z'
    assert plan.marker_keys[26] == 'NUM_AA'
    value = data(plan)
    assert value['answer_template'].count('<NUM_A>') == 1
    assert value['answer_template'].count('<NUM_AA>') == 1
    assert templates.render(plan.model.model_validate(value)).count('stones beside the trail.') == 27


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unknown', 'lowercase', 'space', 'unclosed', 'nested'])
def test_missing_repeated_unknown_and_malformed_markers_reject(templates, kind):
    plan = templates.template_plan(public(4))
    value = data(plan)
    if kind == 'missing':
        value['answer_template'] = value['answer_template'].replace('<NUM_C>', '')
    elif kind == 'duplicate':
        value['answer_template'] += ' <NUM_C>'
    else:
        bad = {'unknown': '<NUM_Z>', 'lowercase': '<num_a>', 'space': '<NUM_ A>',
               'unclosed': '<NUM_X', 'nested': '<NUM_<NUM_A>'}[kind]
        value['answer_template'] += ' ' + bad
    with pytest.raises(ValidationError):
        plan.model.model_validate(value)


@pytest.mark.parametrize('digit', ['8', '\u0664', '\uff12', '\U0001d7dc'])
def test_no_unicode_decimal_digits_anywhere_in_full_template(templates, digit):
    plan = templates.template_plan(public(4))
    value = data(plan)
    value['answer_template'] += ' An accidental ' + digit + ' escaped the values map.'
    with pytest.raises(ValidationError):
        plan.model.model_validate(value)


@pytest.mark.parametrize('value', ['-', '+', '01', '2.7', '2e4', '\u0664', '4\n', 4, None])
def test_numeric_values_are_locally_strict_canonical_ascii_integer_strings(templates, value):
    plan = templates.template_plan(public(4))
    payload = data(plan)
    payload['numeric_values']['NUM_B'] = value
    with pytest.raises(ValidationError):
        plan.model.model_validate(payload)


@pytest.mark.parametrize('change', ['missing', 'extra', 'array', 'null', 'root_extra'])
def test_values_map_and_root_have_only_complete_required_fields(templates, change):
    plan = templates.template_plan(public(4))
    payload = data(plan)
    if change == 'missing':
        del payload['numeric_values']['NUM_B']
    elif change == 'extra':
        payload['numeric_values']['NUM_Z'] = '9'
    elif change == 'array':
        payload['numeric_values'] = list(payload['numeric_values'].values())
    elif change == 'null':
        payload['numeric_values'] = None
    else:
        payload['answer'] = 'A forbidden second representation.'
    with pytest.raises(ValidationError):
        plan.model.model_validate(payload)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, plan.schema)


@pytest.mark.parametrize('n,k', [(0, None), (0, 2), (4, None), (128, 7)])
def test_sdk_strict_schema_keeps_fixed_value_object_keys(templates, n, k):
    from openai.lib._pydantic import to_strict_json_schema
    plan = templates.template_plan(public(n, k))
    schema = to_strict_json_schema(plan.model)
    values = resolve(schema, schema['properties']['numeric_values'])
    assert values['required'] == list(plan.marker_keys)
    assert values['additionalProperties'] is False
    jsonschema.validate(data(plan), schema)


@pytest.mark.parametrize('template', ['', '  ', '\n\t'])
def test_zero_numbers_still_requires_nonblank_complete_template(templates, template):
    plan = templates.template_plan(public(0))
    with pytest.raises(ValidationError):
        plan.model.model_validate({'answer_template': template, 'numeric_values': {}})


def test_zero_numbers_rejects_unknown_markers_or_extra_value(templates):
    plan = templates.template_plan(public(0))
    for value in [
        {'answer_template': 'An unknown <NUM_A> appears.', 'numeric_values': {}},
        {'answer_template': 'A quiet walk.', 'numeric_values': {'NUM_A': '4'}},
    ]:
        with pytest.raises(ValidationError):
            plan.model.model_validate(value)


def test_adjacent_markers_and_words_gain_numeric_literal_boundaries_without_appended_prose(templates):
    plan = templates.template_plan(public(2))
    value = {'answer_template': 'count<NUM_A><NUM_B>stones.',
             'numeric_values': {'NUM_A': '+8', 'NUM_B': '-4'}}
    answer = templates.render(plan.model.model_validate(value))
    assert answer == 'count +8  -4 stones.'
    assert re.findall(r'[+-]?\d+', answer) == ['+8', '-4']


def test_value_mapping_order_cannot_change_where_values_are_inserted(templates):
    plan = templates.template_plan(public(4))
    value = data(plan)
    value['numeric_values'] = dict(reversed(list(value['numeric_values'].items())))
    answer = templates.render(plan.model.model_validate(value))
    assert re.findall(r'\b\d+\b', answer) == ['1', '2', '3', '4']


@pytest.mark.parametrize('template', [
    'They counted <NUM_A> stones and admired trees.',
    'They counted <NUM_A> stones near a sandy border of candy.',
    'They counted <NUM_A> stones with andromeda overhead.',
])
def test_explicit_distinct_conjunction_minimum_is_lexical_and_not_substring_counting(templates, template):
    plan = templates.template_plan(public(1, 2))
    with pytest.raises(ValidationError, match='distinct conjunction'):
        plan.model.model_validate({'answer_template': template, 'numeric_values': {'NUM_A': '4'}})


def test_conjunction_lexical_presence_does_not_claim_grammatical_proof(templates):
    plan = templates.template_plan(public(1, 2))
    # Local checks deliberately cannot prove grammatical use; prompts/review must.
    value = {'answer_template': 'They counted <NUM_A> stones. and but', 'numeric_values': {'NUM_A': '4'}}
    assert templates.render(plan.model.model_validate(value)).endswith('and but')
    assert 'lexical evidence only' in plan.audit()['grammar_limit']
    assert 'genuine grammatical coordination' in templates.prepare_prompt_instruction(plan)


def test_any_requested_number_of_registry_types_is_accepted_without_fixed_first_types(templates):
    plan = templates.template_plan(public(1, 2))
    value = {'answer_template': 'They counted <NUM_A> stones. ' + ' '.join(CLAUSES[-2:]),
             'numeric_values': {'NUM_A': '4'}}
    assert templates.render(plan.model.model_validate(value)).endswith(CLAUSES[-1])
    audit = plan.audit()
    assert 'inserted_conjunction_literals' not in audit
    assert set(audit['allowed_conjunction_literals']) == {'and', 'but', 'or', 'so', 'yet', 'for', 'nor'}


@pytest.mark.parametrize('mutation', ['missing_value', 'illegal_number', 'duplicate_marker', 'digit_template', 'conjunction_removed'])
def test_render_revalidates_recursively_mutated_instances(templates, mutation):
    plan = templates.template_plan(public(4, 2))
    parsed = plan.model.model_validate(data(plan))
    if mutation == 'missing_value':
        parsed.numeric_values.__dict__.pop('NUM_C')
    elif mutation == 'illegal_number':
        parsed.numeric_values.__dict__['NUM_B'] = '4.5'
    elif mutation == 'duplicate_marker':
        parsed.__dict__['answer_template'] += ' <NUM_C>'
    elif mutation == 'digit_template':
        parsed.__dict__['answer_template'] += ' 9'
    else:
        parsed.__dict__['answer_template'] = ' '.join(f'<{key}>' for key in plan.marker_keys)
    with pytest.raises(ValidationError):
        templates.render(parsed)


def test_bypassed_construct_does_not_avoid_strict_render_validation(templates):
    plan = templates.template_plan(public(0))
    parsed = plan.model.model_construct(answer_template='', numeric_values={})
    with pytest.warns(UserWarning, match='Pydantic serializer warnings'):
        with pytest.raises(ValidationError):
            templates.render(parsed)


@pytest.mark.parametrize('rule', [
    'Include exactly 129 numbers.',
    'Include exactly 2 positive numbers.',
    'Include exactly 2 numbers. Include exactly 4 numbers.',
    'Example: "Include exactly 2 numbers."',
    'Include exactly 2 numbers. Include keyword moss in the third sentence as the sixth word.',
    'Include exactly 2 numbers. Print the literal <NUM_A>.',
])
def test_public_unsupported_and_reserved_marker_rules_decline(templates, rule):
    assert templates.template_plan({'question': rule, 'constraints': []}) is None


def test_source_metadata_and_reference_canaries_never_enter_template_schema_or_instruction(templates):
    source = public(4, 2, task_id='PRIVATE_ROUTE', metadata={'secret': 'PRIVATE_REFERENCE'},
                    reference_answer='<NUM_UNKNOWN>', evaluator_kwargs={'count': 99})
    plan = templates.template_plan(source)
    assert plan.number_count == 4 and plan.conjunction_count == 2
    for span in plan.source_plan.matched_public_spans:
        assert source[span.source][span.start:span.end] == span.text
    surface = json.dumps(plan.schema) + json.dumps(plan.audit()) + templates.prepare_prompt_instruction(plan)
    assert 'PRIVATE_ROUTE' not in surface and 'PRIVATE_REFERENCE' not in surface
    assert 'NUM_UNKNOWN' not in surface and 'evaluator_kwargs' not in surface


@pytest.mark.parametrize('suffix', [' "numeric_values": {}', '{}'])
def test_strict_json_rejects_extra_roots_or_bare_properties_without_prefix_recovery(templates, suffix):
    from jit_mas.public_refinement import _strict_json
    plan = templates.template_plan(public(4, 2))
    with pytest.raises(json.JSONDecodeError):
        _strict_json(json.dumps(data(plan)) + suffix, plan.model)


def test_template_preserves_complete_prose_instead_of_requiring_fragment_contexts(templates):
    plan = templates.template_plan(public(1, 1))
    value = {'answer_template': 'We saw <NUM_A> birds, and their calls followed us home.',
             'numeric_values': {'NUM_A': '8'}}
    answer = templates.render(plan.model.model_validate(value))
    assert answer == 'We saw  8  birds, and their calls followed us home.'
    assert 'before' not in plan.schema['properties'] and 'after' not in plan.schema['properties']
    assert set(plan.schema['properties']) == {'answer_template', 'numeric_values'}


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


def _refiner_inputs(mode='json_schema_review', transport='json_object',
                    numeric=True, positional=False, question=None):
    from jit_mas.config import MASConfig, ModelConfig
    from jit_mas.schemas import PublicTask
    from scripts.kernel.types import RunResult
    task = PublicTask(task_id='PRIVATE_ROUTE_CANARY',
        question=question or public(4, 2)['question'])
    result = RunResult(answer='An earlier fictional woodland draft.', terminated_reason='final_answer',
        metadata={'private_feedback': 'PRIVATE_FEEDBACK_CANARY', 'call_counts': {'writer': 1},
                  'model_calls_used': 1})
    config = MASConfig(backend='scripted', public_refinement=True,
        public_refinement_response_format=mode, public_numeric_construction=numeric,
        public_numeric_construction_layout='template', public_positional_construction=positional,
        public_construction_response_format=transport,
        models={'exec': ModelConfig(max_tokens=4096), 'global': ModelConfig(max_tokens=16000)})
    return task, result, config


def _revision_payload():
    return data(public_numeric_template.template_plan(public(4, 2)))


@pytest.mark.parametrize('mode', ['json_schema', 'json_schema_review'])
@pytest.mark.parametrize('transport', ['json_object', 'json_schema'])
def test_real_template_refinement_two_global_calls_share_budget_and_render_complete_public_artifact(mode, transport):
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    task, result, config = _refiner_inputs(mode, transport)
    ledger = BudgetLedger(None, 2_000_000, 0)
    ticket = ledger.reserve('inference', 'writer', 25, 15)
    ledger.settle(ticket, 25, 15)
    revision = _revision_payload()
    models = _LocalModels([{'issues': []}, revision])
    refine_public_answer(task, result, models, ledger, config)
    assert len(models.responses.requests) == 2
    assert all(shared is ledger for _, _, shared, _ in models.created)
    assert [agent for _, agent, _, _ in models.created] == ['public-review', 'public-revision']
    assert ledger.snapshot()['model_calls'] == 3 and ledger.snapshot()['tokens'] == 140
    assert ledger.snapshot()['reserved_tokens'] == 0
    assert result.metadata['call_counts'] == {'writer': 1} and result.metadata['model_calls_used'] == 1
    review_request, revision_request = models.responses.requests
    assert review_request['kwargs']['response_format']['type'] == 'json_schema'
    assert revision_request['kwargs']['response_format']['type'] == transport
    assert revision_request['kwargs']['max_tokens'] == 4096
    assert 'frequency_penalty' not in revision_request['kwargs']
    system = revision_request['messages'][0]['content']
    assert 'Return the complete finished artifact in answer.' not in system
    assert 'Return exactly one object with only answer_template and numeric_values.' in system
    schema = json.loads(system.rsplit('\n', 1)[-1])
    assert set(schema['properties']) == {'answer_template', 'numeric_values'}
    assert schema['additionalProperties'] is False
    assert resolve(schema, schema['properties']['numeric_values'])['required'] == ['NUM_A', 'NUM_B', 'NUM_C', 'NUM_D']
    for request in models.responses.requests:
        assert 'PRIVATE_ROUTE_CANARY' not in json.dumps(request['messages'])
        assert 'PRIVATE_FEEDBACK_CANARY' not in json.dumps(request['messages'])
    audit = result.metadata['public_refinement']
    assert audit['status'] == 'completed' and len(audit['budget_records']) == 2
    assert audit['public_numeric_construction']['layout'] == 'template'
    assert audit['revision'] == revision
    assert audit['revision_public_diagnostics']['numeric_token_diagnostics']['total_numeric_tokens'] == 4
    assert audit['revision_public_diagnostics']['fanboys_lexical_diagnostics']['different_lexical_types_present'] >= 2
    assert result.answer == public_numeric_template.render(
        public_numeric_template.template_plan(task).model.model_validate(revision))
    assert '<NUM_' not in result.answer and 'NUM_A' not in result.answer
    assert not result.answer.endswith('and but')


@pytest.mark.parametrize('transport', ['json_object', 'json_schema'])
@pytest.mark.parametrize('invalid', ['digit_prose', 'duplicate_marker', 'unknown_marker', 'missing_value',
    'too_few_conjunction_types', 'illegal_integer', 'bare_field_suffix', 'second_root'])
def test_real_template_invalid_revision_charges_two_calls_and_fails_without_relaxation_retry_or_prefix_selection(transport, invalid):
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    task, result, config = _refiner_inputs(transport=transport)
    original = result.answer
    value = _revision_payload()
    if invalid == 'digit_prose':
        value['answer_template'] += ' There were 4 extra birds.'
    elif invalid == 'duplicate_marker':
        value['answer_template'] += ' <NUM_A>'
    elif invalid == 'unknown_marker':
        value['answer_template'] += ' <NUM_Z>'
    elif invalid == 'missing_value':
        del value['numeric_values']['NUM_C']
    elif invalid == 'too_few_conjunction_types':
        value['answer_template'] = ' '.join(f'<{key}>' for key in value['numeric_values'])
    elif invalid == 'illegal_integer':
        value['numeric_values']['NUM_A'] = '4.5'
    else:
        root = json.dumps(value)
        value = root + (' "numeric_values": {}' if invalid == 'bare_field_suffix' else root)
    ledger = BudgetLedger(None, 2_000_000, 0)
    models = _LocalModels([{'issues': []}, value, _revision_payload()])
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.responses.requests) == 2
    assert ledger.snapshot()['model_calls'] == 2 and ledger.snapshot()['tokens'] == 100
    assert ledger.snapshot()['reserved_tokens'] == 0
    assert result.answer == original  # exception prevents submission; no draft fallback
    audit = result.metadata['public_refinement']
    assert audit['status'] == 'failed' and audit['revision'] is None
    assert audit['calls'][-1]['status'] == 'failed'
    assert audit['calls'][-1]['response_format']['type'] == transport


@pytest.mark.parametrize('numeric,positional', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('unsupported', [False, True])
@pytest.mark.parametrize('transport', ['json_object', 'json_schema'])
def test_real_template_mixed_rules_decline_both_construction_families_before_compilation(numeric, positional, unsupported, transport):
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    numbers = 'Include exactly 4 positive numbers.' if unsupported else 'Include exactly 4 numbers.'
    task, result, config = _refiner_inputs(transport=transport, numeric=numeric, positional=positional,
        question='Write a fictional woodland walk. ' + numbers +
            ' Include keyword moss in the third sentence, as the sixth word of that sentence.')
    models = _LocalModels([{'issues': []}, {'answer': 'A complete ordinary fictional artifact.'}])
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), config)
    request = models.responses.requests[-1]
    assert request['kwargs']['response_format'] == {'type': 'json_object'}
    assert 'Return the complete finished artifact in answer.' in request['messages'][0]['content']
    audit = result.metadata['public_refinement']
    assert len(models.responses.requests) == 2
    for name, enabled in [('public_numeric_construction', numeric), ('public_positional_construction', positional)]:
        if enabled:
            assert audit[name] == {'active': False, 'reason': 'Mixed public numeric and positional rules'}


def test_real_template_budget_guard_blocks_revision_before_dispatch():
    from jit_mas.budget import BudgetExceeded, BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    task, result, config = _refiner_inputs()
    ledger = BudgetLedger(1, 2_000_000, 0)
    models = _LocalModels([{'issues': []}, _revision_payload()])
    with pytest.raises(BudgetExceeded):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.responses.requests) == 1
    assert ledger.snapshot()['model_calls'] == 1 and ledger.snapshot()['tokens'] == 50


def test_real_template_transport_failure_is_metered_without_switching_or_extra_retry():
    from jit_mas.budget import BudgetLedger
    from jit_mas.public_refinement import refine_public_answer
    task, result, config = _refiner_inputs()
    ledger = BudgetLedger(None, 2_000_000, 0)
    models = _LocalModels([{'issues': []}, RuntimeError('synthetic template transport failure'), _revision_payload()])
    with pytest.raises(RuntimeError, match='synthetic template'):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.responses.requests) == 2
    assert ledger.snapshot()['model_calls'] == 2 and ledger.snapshot()['records'][-1]['estimated'] is True
    assert ledger.snapshot()['reserved_tokens'] == 0
    assert models.responses.requests[-1]['kwargs']['response_format'] == {'type': 'json_object'}


@pytest.mark.parametrize('transport', ['json_object', 'json_schema'])
def test_disabled_template_configuration_preserves_ordinary_answer_protocol_and_defaults(transport):
    from jit_mas.budget import BudgetLedger
    from jit_mas.config import MASConfig
    from jit_mas.public_refinement import REVISION_PROMPT, refine_public_answer
    defaults = MASConfig()
    assert defaults.public_numeric_construction is False
    assert defaults.public_numeric_construction_layout == 'array'
    assert defaults.public_construction_response_format == 'json_schema'
    task, result, config = _refiner_inputs(transport=transport, numeric=False,
        question='Write a short fictional woodland walk.')
    models = _LocalModels([{'issues': []}, {'answer': 'The walkers followed the quiet path home.'}])
    refine_public_answer(task, result, models, BudgetLedger(None, 2_000_000, 0), config)
    request = models.responses.requests[-1]
    assert request['kwargs']['response_format'] == {'type': 'json_object'}
    assert request['messages'][0]['content'].startswith(REVISION_PROMPT)
    assert 'Return the complete finished artifact in answer.' in request['messages'][0]['content']
    assert 'public_numeric_construction' not in result.metadata['public_refinement']
