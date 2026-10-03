"""Real public numeric refinement: shared metering, strict failure and flag guards.

Providers are local fixtures; no network or model API is used.
"""
import copy
import json
import builtins
import pytest

from jit_mas.budget import BudgetExceeded, BudgetLedger, MeteredModel
from jit_mas.config import MASConfig, ModelConfig
from jit_mas.public_refinement import refine_public_answer
from jit_mas.schemas import PublicTask
from scripts.kernel.types import RunResult
from scripts.models.base import ChatMessage



class Responses:
    def __init__(self,replies):
        self.replies = iter(replies)
        self.requests = []

    def __call__(self,messages,**kwargs):
        self.requests.append(dict(messages=copy.deepcopy(messages),kwargs=copy.deepcopy(kwargs)))
        response = next(self.replies)
        if isinstance(response,BaseException):
            raise response
        return ChatMessage(role='assistant',content=response if isinstance(response,str) else json.dumps(response))

    def get_token_counts(self):
        return dict(input_token_count=30,output_token_count=20)


class Models:
    def __init__(self,replies):
        self.responses = Responses(replies)
        self.created = []

    def create(self,role,agent_id,ledger,stage):
        self.created.append((role,agent_id,ledger,stage))
        return MeteredModel(self.responses,ledger,stage,agent_id,16000)


def inputs(mode='json_schema_review',numeric=True,positional=False):
    task = PublicTask(task_id='PRIVATE_ROUTING_CANARY',
        question='Write a fictional woodland account. Include exactly 3 numbers. '
                 'Use at least 2 different coordinating conjunctions.')
    result = RunResult(answer='We saw 7 birds by a tree.',terminated_reason='final_answer',
        metadata={'private_feedback':'PRIVATE_METADATA_CANARY',
                  'call_counts':{'writer':1},'model_calls_used':1})
    config = MASConfig(backend='scripted',public_refinement=True,
        public_refinement_response_format=mode,public_numeric_construction=numeric,
        public_positional_construction=positional,
        models={'exec':ModelConfig(max_tokens=4096),'global':ModelConfig(max_tokens=16000)})
    return task,result,config


def numeric_payload():
    return dict(opening='The woodland path was quiet.',
        number_slots=[{'before':'We counted','number':n,'after':'birds near the stream.'}
                      for n in ['-8','+42','109']],
        conjunction_clauses=[{'before':'The trees swayed','after':'the walkers listened.'},
                             {'before':'Rain began','after':'the walkers kept going.'}],
        closing='They returned home together.')


@pytest.mark.parametrize('mode',['json_schema','json_schema_review'])
def test_numeric_revision_uses_two_shared_metered_calls_and_strict_schema(mode):
    task,result,config = inputs(mode)
    ledger = BudgetLedger(None,2_000_000,0)
    earlier = ledger.reserve('inference','writer',25,15)
    ledger.settle(earlier,25,15)
    models = Models([{'issues':[]},numeric_payload()])
    refine_public_answer(task,result,models,ledger,config)
    assert len(models.responses.requests)==2
    assert all(shared is ledger for _,_,shared,_ in models.created)
    assert [agent for _,agent,_,_ in models.created]==['public-review','public-revision']
    assert ledger.snapshot()['model_calls']==3 and ledger.snapshot()['tokens']==140
    assert ledger.snapshot()['reserved_tokens']==0
    assert result.metadata['call_counts']=={'writer':1}
    assert result.metadata['model_calls_used']==1
    for request in models.responses.requests:
        assert request['kwargs']['max_tokens']==4096
        assert request['kwargs']['response_format']['type']=='json_schema'
        assert request['kwargs']['response_format']['json_schema']['strict'] is True
        assert 'frequency_penalty' not in request['kwargs']
        assert 'PRIVATE_METADATA_CANARY' not in json.dumps(request['messages'])
        assert 'PRIVATE_ROUTING_CANARY' not in json.dumps(request['messages'])
    schema = models.responses.requests[1]['kwargs']['response_format']['json_schema']['schema']
    assert schema['properties']['number_slots']['minItems']==3
    assert schema['properties']['number_slots']['maxItems']==3
    assert schema['properties']['conjunction_clauses']['minItems']==2
    assert 'answer' not in schema['properties']
    assert result.answer.startswith(numeric_payload()['opening'])
    assert result.answer.endswith(numeric_payload()['closing'])
    audit = result.metadata['public_refinement']
    assert audit['status']=='completed'
    assert audit['public_numeric_construction']['number_count']==3
    assert audit['revision']==numeric_payload()
    assert audit['revision_public_diagnostics']['numeric_token_diagnostics']['total_numeric_tokens']==3
    assert len(audit['budget_records'])==2


@pytest.mark.parametrize('invalid',[
    {'opening':'A walk.','number_slots':[],'conjunction_clauses':[],'closing':'Home.'},
    {'answer':'An ordinary answer cannot replace active strict slots.'},
    'not json',
    '{"opening":"A walk.","opening":"A second opening."}',
])
def test_invalid_numeric_revision_fails_after_two_charged_calls_without_retry_or_draft_selection(invalid):
    task,result,config = inputs()
    original = result.answer
    ledger = BudgetLedger(None,2_000_000,0)
    models = Models([{'issues':[]},invalid,{'answer':'Forbidden extra retry.'}])
    with pytest.raises(ValueError):
        refine_public_answer(task,result,models,ledger,config)
    assert result.answer==original  # exception propagates; draft is not submitted
    assert len(models.responses.requests)==2
    assert ledger.snapshot()['model_calls']==2 and ledger.snapshot()['tokens']==100
    assert ledger.snapshot()['reserved_tokens']==0
    audit = result.metadata['public_refinement']
    assert audit['status']=='failed' and audit['review']=={'issues':[]}
    assert audit['revision'] is None and audit['calls'][-1]['status']=='failed'


def test_common_call_budget_blocks_numeric_revision_before_provider_dispatch():
    task,result,config = inputs()
    ledger = BudgetLedger(1,2_000_000,0)
    models = Models([{'issues':[]},numeric_payload()])
    with pytest.raises(BudgetExceeded):
        refine_public_answer(task,result,models,ledger,config)
    assert len(models.responses.requests)==1
    assert ledger.snapshot()['model_calls']==1 and ledger.snapshot()['tokens']==50
    assert result.metadata['public_refinement']['status']=='failed'


def test_schema_transport_failure_keeps_charge_and_does_not_switch_formats():
    task,result,config = inputs()
    ledger = BudgetLedger(None,2_000_000,0)
    models = Models([{'issues':[]},RuntimeError('synthetic schema failure'),numeric_payload()])
    with pytest.raises(RuntimeError,match='synthetic'):
        refine_public_answer(task,result,models,ledger,config)
    assert len(models.responses.requests)==2 and ledger.snapshot()['model_calls']==2
    assert ledger.snapshot()['records'][-1]['estimated'] is True
    assert ledger.snapshot()['reserved_tokens']==0
    assert models.responses.requests[-1]['kwargs']['response_format']['type']=='json_schema'


def test_default_numeric_flag_stays_off_and_ordinary_revision_remains_ordinary():
    assert MASConfig().public_numeric_construction is False
    task,result,config = inputs(numeric=False)
    models = Models([{'issues':[]},{'answer':'The complete ordinary artifact.'}])
    refine_public_answer(task,result,models,BudgetLedger(None,2_000_000,0),config)
    assert result.answer=='The complete ordinary artifact.'
    assert models.responses.requests[-1]['kwargs']['response_format']=={'type':'json_object'}


def test_both_default_off_flags_avoid_importing_construction_helpers(monkeypatch):
    task,result,config = inputs(numeric=False,positional=False)
    original_import = builtins.__import__

    def ordinary_import(name,globals=None,locals=None,fromlist=(),level=0):
        if 'public_numeric_slots' in name or 'public_word_slots' in name:
            pytest.fail('Disabled construction branch imported a slot compiler')
        return original_import(name,globals,locals,fromlist,level)

    monkeypatch.setattr(builtins,'__import__',ordinary_import)
    models = Models([{'issues':[]},{'answer':'A complete ordinary artifact.'}])
    refine_public_answer(task,result,models,BudgetLedger(None,2_000_000,0),config)
    assert len(models.responses.requests)==2
    assert result.answer=='A complete ordinary artifact.'


def test_numeric_only_revision_has_no_unspecified_conjunction_field():
    task,result,config = inputs()
    task.question='Write a fictional woodland account. Include exactly 3 numbers.'
    payload=numeric_payload()
    del payload['conjunction_clauses']
    models=Models([{'issues':[]},payload])
    refine_public_answer(task,result,models,BudgetLedger(None,2_000_000,0),config)
    schema=models.responses.requests[-1]['kwargs']['response_format']['json_schema']['schema']
    assert 'conjunction_clauses' not in schema['properties']
    audit=result.metadata['public_refinement']
    assert audit['public_numeric_construction']['minimum_distinct_conjunction_count'] is None
    assert audit['revision']==payload
    assert audit['revision_public_diagnostics']['numeric_token_diagnostics']['total_numeric_tokens']==3


@pytest.mark.parametrize('options',[
    {'public_refinement':False,'public_refinement_response_format':'json_schema'},
    {'public_refinement':True,'public_refinement_response_format':'json_object'},
    {'public_refinement':False,'public_refinement_response_format':'json_schema_review'},
])
def test_numeric_flag_requires_active_schema_refinement(options):
    with pytest.raises(ValueError):
        MASConfig(public_numeric_construction=True,**options)


@pytest.mark.parametrize('mode',['json_schema','json_schema_review'])
@pytest.mark.parametrize('numeric,positional',[(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('number_rule',['Include exactly 3 numbers.','Include exactly three distinct numbers.'])
def test_mixed_families_fall_back_before_either_construction_even_when_one_flag_is_off(mode,numeric,positional,number_rule):
    task,result,config = inputs(mode,numeric=numeric,positional=positional)
    task.question=number_rule
    task.constraints=['Put the word Birch in the second sentence as the third word of that sentence.']
    models = Models([{'issues':[]},{'answer':'The complete ordinary artifact.'}])
    refine_public_answer(task,result,models,BudgetLedger(None,2_000_000,0),config)
    assert result.answer=='The complete ordinary artifact.'
    assert len(models.responses.requests)==2
    expected_type='json_schema' if mode=='json_schema' else 'json_object'
    assert models.responses.requests[-1]['kwargs']['response_format']['type']==expected_type
    audit = result.metadata['public_refinement']
    if numeric:
        assert audit['public_numeric_construction']['active'] is False
    if positional:
        assert audit['public_positional_construction']['active'] is False
    payload = json.loads(models.responses.requests[-1]['messages'][1]['content'])
    assert 'number_slots' not in payload and 'prefix_words' not in payload
