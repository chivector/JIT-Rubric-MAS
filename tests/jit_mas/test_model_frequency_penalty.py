"""Offline real wrapper transmission and rejection; never reaches a network."""
from types import SimpleNamespace

import openai
import pytest
from pydantic import ValidationError

from jit_mas.budget import BudgetLedger
from jit_mas.config import MASConfig, ModelConfig, NativeModels


@pytest.mark.parametrize('value', [-2.01, 2.01, float('nan'), float('inf'), -float('inf')])
def test_penalty_rejects_out_of_range_and_nonfinite(value):
    with pytest.raises(ValidationError):
        ModelConfig(frequency_penalty=value)


@pytest.fixture
def provider(monkeypatch):
    sent = []
    fail = [False]

    class Message:
        content = '{"answer":"synthetic"}'
        tool_calls = reasoning = reasoning_content = None

        def model_dump(self, include=None):
            return {'role': 'assistant', 'content': self.content, 'tool_calls': None}

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            sent.append(kwargs)
            if fail[0]:
                raise RuntimeError('synthetic unsupported option')
            return SimpleNamespace(model='synthetic-model', id='synthetic-response', system_fingerprint='synthetic',
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
                choices=[SimpleNamespace(message=Message(), finish_reason='stop')])

    monkeypatch.setattr(openai, 'OpenAI', Client)
    monkeypatch.setenv('SYNTHETIC_MODEL_KEY', 'not-a-real-key')
    monkeypatch.setenv('JIT_MAS_MODEL_ATTEMPTS', '1')
    monkeypatch.delenv('EXEC_NATIVE_TOOLCALL', raising=False)
    monkeypatch.delenv('EXEC_FORMAT_REMINDER', raising=False)
    return sent, fail


def native_models(penalty):
    roles = ('meta', 'global', 'local', 'exec', 'judge')
    cfg = ModelConfig(model='synthetic-model', endpoint='http://synthetic.invalid/v1',
        key_env='SYNTHETIC_MODEL_KEY', max_tokens=64, frequency_penalty=penalty,
        expected_response_model='synthetic-model', thinking='disabled', reasoning_effort='none')
    return NativeModels(MASConfig(unsafe_local=True, models={role: cfg for role in roles}))


@pytest.mark.parametrize('role', ['meta', 'global', 'local', 'exec', 'judge'])
@pytest.mark.parametrize('penalty', [None, .25, -2, 2])
def test_actual_completion_kwargs_preserve_opt_in_and_shared_accounting(provider, role, penalty):
    sent, _ = provider
    ledger = BudgetLedger(max_calls=None, timeout_seconds=10)
    model = native_models(penalty).create(role, 'synthetic', ledger, 'inference')
    response = model([{'role': 'user', 'content': 'Synthetic transport test.'}])
    assert response.content == '{"answer":"synthetic"}'
    assert len(sent) == 1
    if penalty is None:
        assert 'frequency_penalty' not in sent[0]
    else:
        assert sent[0]['frequency_penalty'] == penalty
    assert sent[0]['temperature'] == 0
    assert sent[0]['max_tokens'] == 64
    assert sent[0]['reasoning_effort'] == 'none'
    assert sent[0]['extra_body'] == {'thinking': {'type': 'disabled'}}
    if role in {'global', 'local'}:
        assert sent[0]['response_format'] == {'type': 'json_object'}
    else:
        assert 'response_format' not in sent[0]
    budget = ledger.snapshot()
    assert budget['model_calls'] == 1 and budget['tokens'] == 5 and budget['reserved_tokens'] == 0


def test_unsupported_option_fails_once_and_is_fully_metered_without_fallback(provider):
    sent, fail = provider
    fail[0] = True
    ledger = BudgetLedger(max_calls=None, timeout_seconds=10)
    model = native_models(.25).create('exec', 'synthetic', ledger, 'inference')
    with pytest.raises(RuntimeError, match='synthetic unsupported option'):
        model([{'role': 'user', 'content': 'Synthetic rejection test.'}])
    assert len(sent) == 1 and sent[0]['frequency_penalty'] == .25
    budget = ledger.snapshot()
    assert budget['model_calls'] == 1 and budget['tokens'] > 0 and budget['reserved_tokens'] == 0
    assert budget['records'][0]['estimated'] and budget['records'][0]['error'] == 'RuntimeError'


def test_model_default_keeps_penalty_absent():
    assert ModelConfig().frequency_penalty is None
