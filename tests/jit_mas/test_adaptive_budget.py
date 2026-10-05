"""Roster size stays a model decision within configured and live budgets."""

import json

import pytest
from pydantic import ValidationError

from jit_mas.agent_pool import seed_pool
from jit_mas.budget import BudgetLedger
from jit_mas.config import MASConfig
from jit_mas.planning import GlobalAnalyzer
from jit_mas.schemas import AgentSpec, PublicTask


def candidates(count, mode='single_pass'):
    roles = ['analyst', 'critic', 'writer'][-count:]
    return [AgentSpec(agent_id=role, role=role.title(), capability='reasoning',
                      pool_agent_id=role, pool_agent_version=1,
                      max_calls=None if mode == 'iterative_shared_ledger' else 1,
                      max_tokens=256).model_dump(mode='json') for role in roles]


@pytest.mark.parametrize('question,count', [
    ('Prove it.', 3),
    ('Compare the alternatives and explain the result.', 3),
    ('Summarize these repeated notes. ' * 50, 1),
])
@pytest.mark.parametrize('mode', ['single_pass', 'iterative_shared_ledger'])
def test_roster_is_not_determined_by_question_length_or_keywords(question, count, mode):
    agents = candidates(count, mode)
    graph = {'rubrics': []}
    calls = []
    budget = BudgetLedger(max_calls=None, max_tokens=5000)
    planned_calls = 2 if mode == 'iterative_shared_ledger' else 1

    def model(messages, **kwargs):
        payload = json.loads(messages[1]['content'])
        calls.append((payload, kwargs))
        if payload['phase'] == 'predict':
            return json.dumps({'graph': graph, 'candidates': agents})
        team_agents = [dict(a) for a in agents]
        team_agents[-1]['depends_on'] = [a['agent_id'] for a in agents[:-1]]
        return json.dumps({'graph': graph, 'team': {
            'agents': team_agents, 'synthesizer_id': agents[-1]['agent_id'],
            'max_parallel': 2, 'total_max_calls': None, 'execution_mode': mode,
            'budget_plan': {
                'agents': [
                    {'agent_id': a['agent_id'], 'expected_model_calls': planned_calls,
                     'expected_input_tokens': 400, 'expected_output_tokens': 200,
                     'rationale': 'Independent reasoning and verification for this task.'}
                    for a in agents],
                'reserved_future_tokens': 500, 'reserved_future_model_calls': 1,
                'quality_cost_tradeoff': 'Distinct contributions justify their expected token cost.',
                'stopping_policy': 'Finish once the deliverable and consequential checks are complete.',
            },
        }})

    analyzer = GlobalAnalyzer(model, max_agents=3, agent_pool=seed_pool(),
                              execution_mode=mode, total_max_calls=None,
                              budget_context=budget.resource_context, max_corrections=0)
    analyzer.planning_response_format = 'json_schema'
    team = analyzer.build(PublicTask(task_id='roster', question=question),
                          local_planning=False).team
    assert len(team.agents) == count
    assert team.total_max_calls is None
    assert team.budget_plan.totals()['model_calls'] == count * planned_calls
    assert len(calls) == 2
    for payload, kwargs in calls:
        assert payload['limits']['max_agents'] == 3
        assert 'adaptive_budget' not in payload['limits']
        assert payload['limits']['resource_budget']['remaining_tokens'] == 5000
    predict_schema = calls[0][1]['response_format']['json_schema']['schema']
    assert predict_schema['properties']['candidates']['maxItems'] == 3


@pytest.mark.parametrize('max_agents,remaining_calls,error', [
    (2, 10, 'Candidate count exceeds limit=2'),
    (3, 2, 'remaining shared model-call budget'),
])
def test_real_resource_limits_still_reject_unaffordable_rosters(max_agents, remaining_calls, error):
    response = {'graph': {'rubrics': []}, 'candidates': candidates(3)}
    budget = BudgetLedger(max_calls=remaining_calls, max_tokens=5000)
    analyzer = GlobalAnalyzer(lambda _: json.dumps(response), max_agents=max_agents,
                              agent_pool=seed_pool(), max_corrections=0,
                              budget_context=budget.resource_context)
    with pytest.raises(ValueError, match=error):
        analyzer.predict(PublicTask(task_id='budget', question='Prove it.'))


def test_old_disabled_config_loads_but_difficulty_caps_cannot_be_reenabled():
    assert MASConfig(adaptive_budget_enforcement=False).adaptive_budget_enforcement is False
    with pytest.raises(ValidationError, match='adaptive_budget_enforcement'):
        MASConfig(adaptive_budget_enforcement=True)
    with pytest.raises(ValueError, match='Difficulty-based roster enforcement has been removed'):
        GlobalAnalyzer(lambda _: '', adaptive_budget_enforcement=True)
