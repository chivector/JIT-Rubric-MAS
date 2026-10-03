"""Stage-local option transport, audit and failure charging; synthetic only."""
import copy
import json

import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_refinement import refine_public_answer
from jit_mas.schemas import PublicTask, digest
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage


class Models:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def create(self, role, agent_id, ledger, stage):
        assert role == 'global' and stage == 'inference'
        return MeteredModel(self, ledger, stage, agent_id, 16000)

    def __call__(self, messages, **kwargs):
        self.requests.append({'messages': copy.deepcopy(messages), 'kwargs': copy.deepcopy(kwargs)})
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return ChatMessage(role='assistant', content=response if isinstance(response, str) else json.dumps(response))

    def get_token_counts(self):
        return {'input_token_count': 30, 'output_token_count': 20}


def inputs(positional, penalty):
    question = 'Write a complete report from the supplied public notes.'
    if positional:
        question = 'Write a scene. Include keyword brightly in the second sentence, as the third word of that sentence.'
    task = PublicTask(task_id='synthetic-only', question=question)
    result = RunResult(answer='Original draft.', terminated_reason='final_answer', metadata={})
    config = MASConfig(backend='scripted', public_refinement=True, public_refinement_response_format='json_schema_review',
        public_positional_construction=positional, public_revision_frequency_penalty=penalty,
        models={'global': ModelConfig(max_tokens=16000), 'exec': ModelConfig(max_tokens=8192)})
    ledger = BudgetLedger(None, 2_000_000, None, timeout_seconds=900)
    return task, result, config, ledger


def revision(positional):
    if positional:
        return {'preceding_sentences': ['The curtain rose.'], 'prefix_words': ['She', 'smiled'],
            'keyword': 'brightly', 'suffix_words': ['as', 'the', 'crowd', 'cheered'], 'following_sentences': []}
    return {'answer': 'Complete report with supported details.'}


@pytest.mark.parametrize('positional', [False, True])
@pytest.mark.parametrize('penalty', [None, .25, -2, 2])
def test_only_actual_revision_call_receives_explicit_option_and_exact_audit(positional, penalty):
    task, result, config, ledger = inputs(positional, penalty)
    models = Models([{'issues': []}, revision(positional)])
    refine_public_answer(task, result, models, ledger, config)
    assert len(models.requests) == ledger.snapshot()['model_calls'] == 2
    first, second = [request['kwargs'] for request in models.requests]
    assert 'frequency_penalty' not in first
    if penalty is None:
        assert 'frequency_penalty' not in second
    else:
        assert second['frequency_penalty'] == penalty
    assert first['response_format']['type'] == 'json_schema'
    assert second['response_format']['type'] == ('json_schema' if positional else 'json_object')
    audit = result.metadata['public_refinement']
    for call, request in zip(audit['calls'], models.requests):
        assert call['model_call_options'] == request['kwargs']
        assert call['model_call_options_hash'] == digest(request['kwargs'])
    assert audit['status'] == 'completed' and audit['draft'] == 'Original draft.'
    assert ledger.snapshot()['tokens'] == 100 and ledger.snapshot()['reserved_tokens'] == 0
    assert config.models['global'].frequency_penalty is None


@pytest.mark.parametrize('positional', [False, True])
def test_unsupported_revision_option_fails_once_without_fallback_and_retains_cost(positional):
    task, result, config, ledger = inputs(positional, .25)
    models = Models([{'issues': []}, RuntimeError('synthetic unsupported option'), revision(positional)])
    with pytest.raises(RuntimeError, match='synthetic unsupported option'):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.requests) == ledger.snapshot()['model_calls'] == 2
    assert 'frequency_penalty' not in models.requests[0]['kwargs']
    assert models.requests[1]['kwargs']['frequency_penalty'] == .25
    audit = result.metadata['public_refinement']
    assert result.answer == audit['draft'] == 'Original draft.' and audit['status'] == 'failed'
    assert audit['calls'][-1]['model_call_options'] == models.requests[-1]['kwargs']
    assert ledger.snapshot()['tokens'] > 50 and ledger.snapshot()['reserved_tokens'] == 0
    assert ledger.snapshot()['records'][-1]['estimated'] and ledger.snapshot()['records'][-1]['error'] == 'RuntimeError'


@pytest.mark.parametrize('positional', [False, True])
def test_malformed_revision_is_not_retried_or_selected_as_fallback(positional):
    task, result, config, ledger = inputs(positional, .25)
    models = Models([{'issues': []}, '{"answer":"unfinished"', revision(positional)])
    with pytest.raises(ValueError):
        refine_public_answer(task, result, models, ledger, config)
    assert len(models.requests) == ledger.snapshot()['model_calls'] == 2
    audit = result.metadata['public_refinement']
    assert result.answer == audit['draft'] == 'Original draft.' and audit['status'] == 'failed'
    assert audit['calls'][-1]['response'] == '{"answer":"unfinished"'
    assert audit['calls'][-1]['model_call_options_hash'] == digest(models.requests[-1]['kwargs'])
    assert ledger.snapshot()['tokens'] == 100 and ledger.snapshot()['reserved_tokens'] == 0


@pytest.mark.parametrize('value', [-2.01, 2.01, float('nan'), float('inf'), -float('inf')])
def test_revision_penalty_range_and_nonfinite_rejected(value):
    with pytest.raises(ValidationError):
        MASConfig(public_revision_frequency_penalty=value)


def test_revision_penalty_is_off_by_default():
    assert MASConfig().public_revision_frequency_penalty is None
