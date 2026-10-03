"""Synthetic actual planning stages without graph or budget relaxation."""
import copy
import json
from types import SimpleNamespace

import pytest

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.planning import GlobalAnalyzer, public_planning_context
from jit_mas.schemas import AgentSpec, Prediction, PublicTask, RubricGraph

QUESTION = ('Write a concise mountain journey. Put the word Juniper in the third '
            'sentence as the fourth word of that sentence.')


def config(**changes):
    values = dict(public_refinement=True, public_positional_construction=True,
                  public_positional_draft_guidance=True,
                  public_positional_draft_projection=True,
                  reference_answers='SYNTHETIC_PRIVATE_CONFIG_CANARY')
    values.update(changes)
    return SimpleNamespace(**values)


def organization(mode, role='Evidence analyst'):
    cap = None if mode == 'iterative_shared_ledger' else 1
    graph = RubricGraph(rubrics=[{'rubric_id': 'completeness',
                                'requirement': 'Preserve the complete public request.'}])
    agents = [AgentSpec(agent_id='engineer', role=role, capability='public analysis',
                        rubric_ids=['completeness'], max_tokens=128, max_calls=cap),
              AgentSpec(agent_id='editor', role='Final author', capability='composition',
                        rubric_ids=['completeness'], depends_on=['engineer'],
                        max_tokens=256, max_calls=cap)]
    prediction = Prediction(graph=graph, candidates=agents)
    response = {'graph': graph.model_dump(mode='json'), 'team': {
        'execution_mode': mode, 'agents': [a.model_dump(mode='json') for a in agents],
        'synthesizer_id': 'editor', 'coverage': {'completeness': ['engineer', 'editor']},
        'primary': {'completeness': 'engineer'}, 'reviewers': {},
        'total_max_calls': None if cap is None else 2,
        'budget_plan': {'agents': [{'agent_id': a.agent_id, 'expected_model_calls': 1,
                                  'expected_input_tokens': 50,
                                  'expected_output_tokens': a.max_tokens,
                                  'rationale': 'One compact initial contribution.'} for a in agents],
                        'quality_cost_tradeoff': 'Keep useful analysis before complete synthesis.',
                        'stopping_policy': 'Complete within shared resource ceilings.'}}}
    return prediction, response


class Capture:
    def __init__(self, prediction, response):
        self.prediction, self.response = prediction, response
        self.requests = []

    def __call__(self, messages, **kwargs):
        payload = json.loads(messages[1]['content'])
        self.requests.append(copy.deepcopy({'messages': messages, 'payload': payload,
                                           'kwargs': kwargs}))
        if payload['phase'] == 'predict':
            return self.prediction.model_dump_json()
        if payload['phase'] == 'local_plan':
            c = payload['candidate']
            return json.dumps({key: c[key] for key in
                               ('agent_id', 'capability', 'rubric_ids', 'depends_on', 'max_calls')})
        return json.dumps(self.response)

    def get_token_counts(self):
        return {'input_token_count': 12, 'output_token_count': 8}


def analyzer_for(raw, ledger, mode):
    bound = MeteredModel(raw, ledger, 'inference', 'planning', max_tokens=1024)
    return GlobalAnalyzer(bound, execution_mode=mode,
                          total_max_calls=None if mode == 'iterative_shared_ledger' else 2,
                          budget_context=ledger.resource_context)


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
@pytest.mark.parametrize('role', ['Evidence analyst', 'Analytical critic'])
def test_conditional_stages_reach_every_actual_planning_phase_without_task_edits(mode, role):
    task = PublicTask(task_id='synthetic-stage', question=QUESTION,
                      constraints=['Use plain prose and retain the destination.'])
    original = task.model_dump(mode='json')
    prediction, response = organization(mode, role)
    before_prediction = prediction.model_dump(mode='json')
    raw = Capture(prediction, response)
    ledger = BudgetLedger(max_calls=None, max_tokens=2_000_000)
    analyzer = analyzer_for(raw, ledger, mode)
    context = public_planning_context(task, config())
    analyzer.public_planning_context = context
    result = analyzer.build(task)
    assert len(raw.requests) == 4
    assert {r['payload']['phase'] for r in raw.requests} == {'predict', 'local_plan', 'reconcile'}
    for request in raw.requests:
        payload = request['payload']
        assert payload['task'] == original
        assert payload['public_planning_context'] == context
        assert 'SYNTHETIC_PRIVATE_CONFIG_CANARY' not in json.dumps(payload)
    assert context['positional_draft_stage']['active'] is True
    assert 'Juniper' not in json.dumps(context)
    assert task.model_dump(mode='json') == original
    assert prediction.model_dump(mode='json') == before_prediction
    assert analyzer.last_prediction.model_dump(mode='json') == before_prediction
    assert result.team.synthesizer_id == 'editor'
    assert result.team.agents[1].depends_on == ['engineer']
    assert result.team.reviewers == {}
    assert result.team.agents[0].role == role
    assert result.team.agents[0].max_calls is (None if mode == 'iterative_shared_ledger' else 1)
    assert ledger.snapshot()['model_calls'] == 4
    assert ledger.snapshot()['tokens'] == 80
    assert ledger.snapshot()['reserved_tokens'] == 0


@pytest.mark.parametrize('flag', ['public_refinement', 'public_positional_construction',
                                 'public_positional_draft_guidance',
                                 'public_positional_draft_projection'])
def test_each_stage_flag_is_required_for_position_deferral(flag):
    task = PublicTask(task_id='synthetic-flags', question=QUESTION)
    context = public_planning_context(task, config(**{flag: False}))
    assert 'positional_draft_stage' not in context
    assert bool(context) is (flag != 'public_refinement')


@pytest.mark.parametrize('question', [
    'Write a compact mountain journey with no exact positional rule.',
    'Only if useful, put the word Juniper in the third sentence as the fourth word of that sentence.',
    'Reproduce the following passage verbatim. Put the word Juniper in the third sentence as the fourth word of that sentence.',
    'Write a journey. Put the word Juniper in the third sentence as the fourth word of that sentence. Include exactly five numbers.',
])
def test_unsupported_qualified_quoted_or_conflicting_tasks_do_not_defer_positions(question):
    task = PublicTask(task_id='synthetic-unsupported', question=question)
    original = task.model_dump(mode='json')
    context = public_planning_context(task, config())
    assert context['initial_team_reviewers'] == 'empty'
    assert 'positional_draft_stage' not in context
    assert task.model_dump(mode='json') == original


def test_supported_rule_in_constraints_is_detected_without_changing_any_public_field():
    task = PublicTask(task_id='synthetic-constraint', question='Write a compact mountain journey.',
        constraints=[QUESTION.split('. ', 1)[1], 'Preserve a hopeful tone.'])
    before = task.model_dump(mode='json')
    context = public_planning_context(task, config())
    assert context['positional_draft_stage']['active'] is True
    assert task.model_dump(mode='json') == before


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_refinement_off_has_identical_actual_default_requests(mode):
    task = PublicTask(task_id='synthetic-default', question=QUESTION)
    prediction, response = organization(mode)
    before, after = Capture(prediction, response), Capture(prediction, response)
    kwargs = {'execution_mode': mode, 'total_max_calls': response['team']['total_max_calls']}
    old, new = GlobalAnalyzer(before, **kwargs), GlobalAnalyzer(after, **kwargs)
    new.public_planning_context = public_planning_context(task, config(public_refinement=False))
    old_result, new_result = old.build(task), new.build(task)
    project = lambda rows: sorted((json.dumps(r, sort_keys=True) for r in rows))
    assert project(before.requests) == project(after.requests)
    assert old_result == new_result
    assert all('public_planning_context' not in r['payload'] for r in after.requests)


@pytest.mark.parametrize('error_kind', ['nonterminal_synthesizer', 'assignment_mismatch', 'cycle'])
def test_invalid_generated_graph_is_not_cleaned_or_accepted_and_both_attempts_are_charged(error_kind):
    task = PublicTask(task_id='synthetic-invalid', question=QUESTION)
    prediction, response = organization('iterative_shared_ledger')
    bad = copy.deepcopy(response)
    if error_kind == 'nonterminal_synthesizer':
        bad['team']['synthesizer_id'] = 'engineer'
    elif error_kind == 'assignment_mismatch':
        bad['team']['agents'][1]['rubric_ids'] = []
    else:
        bad['team']['agents'][0]['depends_on'] = ['editor']
    unchanged = copy.deepcopy(bad)
    raw = Capture(prediction, bad)
    ledger = BudgetLedger(max_calls=None, max_tokens=2_000_000)
    analyzer = analyzer_for(raw, ledger, 'iterative_shared_ledger')
    context = public_planning_context(task, config())
    analyzer.public_planning_context = context
    with pytest.raises(ValueError):
        analyzer.reconcile(task, prediction, [])
    assert len(raw.requests) == len(analyzer.call_records) == 2
    assert all(r['validation_errors'] for r in analyzer.call_records)
    assert raw.requests[1]['payload']['response_correction']['assignment_audit']
    assert all(r['payload']['task'] == task.model_dump(mode='json') for r in raw.requests)
    assert all(r['payload']['public_planning_context'] == context for r in raw.requests)
    assert bad == unchanged
    assert ledger.snapshot()['model_calls'] == 2
    assert ledger.snapshot()['tokens'] == 40
    assert ledger.snapshot()['reserved_tokens'] == 0


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_positional_stage_does_not_create_rubrics_when_explicit_prediction_is_disabled(mode):
    task = PublicTask(task_id='synthetic-no-rubrics', question=QUESTION)
    prediction, response = organization(mode)
    prediction.graph = RubricGraph(rubrics=[])
    for agent in prediction.candidates:
        agent.rubric_ids = []
    response['graph'] = prediction.graph.model_dump(mode='json')
    for agent in response['team']['agents']:
        agent['rubric_ids'] = []
    for field in ('coverage', 'primary', 'reviewers'):
        response['team'][field] = {}
    raw = Capture(prediction, response)
    ledger = BudgetLedger(max_calls=None, max_tokens=2_000_000)
    analyzer = analyzer_for(raw, ledger, mode)
    analyzer.explicit_rubrics = False
    analyzer.public_planning_context = public_planning_context(task, config())
    result = analyzer.build(task)
    assert result.graph.rubrics == []
    assert all(agent.rubric_ids == [] for agent in result.team.agents)
    assert all(r['payload']['task'] == task.model_dump(mode='json') for r in raw.requests)
    assert all(r['payload']['public_planning_context']['positional_draft_stage']['active']
               for r in raw.requests)
    assert ledger.snapshot()['model_calls'] == 4


@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_existing_sole_correction_can_fix_final_sink_without_rewriting_original_records(mode):
    task = PublicTask(task_id='synthetic-correction', question=QUESTION)
    prediction, response = organization(mode)
    bad = copy.deepcopy(response)
    bad['team']['synthesizer_id'] = 'engineer'
    class CorrectingCapture(Capture):
        def __call__(self, messages, **kwargs):
            payload = json.loads(messages[1]['content'])
            self.response = response if 'response_correction' in payload else bad
            return super().__call__(messages, **kwargs)

    raw = CorrectingCapture(prediction, bad)
    ledger = BudgetLedger(max_calls=None, max_tokens=2_000_000)
    analyzer = analyzer_for(raw, ledger, mode)
    context = public_planning_context(task, config())
    analyzer.public_planning_context = context
    result = analyzer.reconcile(task, prediction, [])
    assert result.team.synthesizer_id == 'editor'
    assert result.team.agents[1].depends_on == ['engineer']
    assert len(raw.requests) == 2
    assert all(r['payload']['public_planning_context'] == context for r in raw.requests)
    assert all(r['payload']['task'] == task.model_dump(mode='json') for r in raw.requests)
    assert analyzer.call_records[0]['validation_errors']
    assert not analyzer.call_records[1].get('validation_errors')
    assert json.loads(analyzer.call_records[0]['response'])['team']['synthesizer_id'] == 'engineer'
    assert bad['team']['synthesizer_id'] == 'engineer'
    assert ledger.snapshot()['model_calls'] == 2
    assert ledger.snapshot()['tokens'] == 40
