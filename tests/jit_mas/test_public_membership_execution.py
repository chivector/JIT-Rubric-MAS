"""Opt-in writer context forwards public conditions without routing or private fields."""
import json
from types import SimpleNamespace

from jit_mas.execution import _apply_public_membership_guidance
from jit_mas.schemas import PublicTask
from jit_mas.schemas import digest

def test_default_off_preserves_instruction_and_system_without_observations():
    instruction = {"public_task": {"question": "Write a scene."}, "shared_ledger": {}}
    system = _apply_public_membership_guidance('System', instruction, SimpleNamespace())
    assert system == 'System' and 'public_membership_observations' not in instruction

def test_existing_writer_context_only_projects_public_ledger_claims_and_original_conditions():
    task = PublicTask(task_id='OPAQUE_ROUTE_DO_NOT_EMIT', question='List providers with score at least 2.8 points in 2024.')
    services = SimpleNamespace(public_task=task, public_membership_observations_requested=True)
    instruction = {"shared_ledger": {"private_feedback": 'PRIVATE_CANARY', 'contributions': [
        {'answer': 'Cedar PASS', 'private_judge': 'PRIVATE_CANARY', 'ledger': {
            'outline': ['Cedar score is 2.7: PASS'], 'private_rubric': 'PRIVATE_CANARY',
            'source_references': [], 'evidence_spans': [], 'requirements': []}}]}}
    system = _apply_public_membership_guidance('System', instruction, services)
    observations = instruction['public_membership_observations']
    assert observations['original_request']['text'] == task.question
    assert observations['membership_status'] == 'unknown' and not observations['independently_verified']
    assert observations['upstream_claims']
    assert 'PRIVATE_CANARY' not in json.dumps(observations)
    assert 'OPAQUE_ROUTE_DO_NOT_EMIT' not in json.dumps(observations)
    assert 'UNKNOWN is missing verification' in system


def test_compact_input_persists_complete_basis_before_a_request():
    task = PublicTask(task_id='PRIVATE_ROUTING_CANARY',
                      question='For 2023, list entities whose score is at least 2.8 points.')
    snapshots = []
    services = SimpleNamespace(public_task=task, public_membership_observations_requested=True,
                               public_membership_input_format='compact',
                               public_membership_audit_writer=snapshots.append)
    instruction = {'shared_ledger': {}}
    _apply_public_membership_guidance('System', instruction, services)
    projected = instruction['public_membership_observations']
    assert len(snapshots) == 1 and len(snapshots[0]) == 1
    basis = snapshots[0][0]
    assert basis['full_observation_hash'] == digest(basis['observations'])
    assert projected['full_observation_hash'] == basis['full_observation_hash']
    assert basis['model_input_hash'] == digest(projected)
    assert basis['observations']['original_request']['text'] == task.question
    assert 'observations' not in projected
    assert 'PRIVATE_ROUTING_CANARY' not in json.dumps(snapshots)
    assert snapshots[0][0]['observations']['membership_status'] == 'unknown'


def test_compact_execution_audit_deduplicates_identical_bases_and_keeps_changes():
    task = PublicTask(task_id='opaque', question='List entities whose score is at least 2.8 points.')
    snapshots = []
    services = SimpleNamespace(public_task=task, public_membership_observations_requested=True,
                               public_membership_input_format='compact',
                               public_membership_audit_writer=snapshots.append)
    for _ in range(2):
        _apply_public_membership_guidance('System', {'shared_ledger': {}}, services)
    assert len(services.public_membership_execution_audits) == 1
    assert len(snapshots[0]) == len(snapshots[1]) == 1
    task.question = 'List entities whose score is at least 3.0 points.'
    _apply_public_membership_guidance('System', {'shared_ledger': {}}, services)
    assert len(services.public_membership_execution_audits) == 2
    assert len(snapshots[-1]) == 2 and len(snapshots[0]) == 1


def test_full_execution_format_keeps_model_payload_and_never_creates_compact_audit():
    from jit_mas.public_membership import build_public_membership_observations

    task = PublicTask(task_id='opaque', question='List entities with score at least 2.8 points.')
    snapshots = []
    services = SimpleNamespace(public_task=task, public_membership_observations_requested=True,
                               public_membership_input_format='full',
                               public_membership_audit_writer=snapshots.append)
    instruction = {'shared_ledger': {}}
    _apply_public_membership_guidance('System', instruction, services)
    assert instruction['public_membership_observations'] == build_public_membership_observations(task, {})
    assert not snapshots and not hasattr(services, 'public_membership_execution_audits')


def test_pipeline_failure_retains_rich_execution_basis_on_disk_before_provider_call(tmp_path, monkeypatch):
    import pytest
    from jit_mas.config import MASConfig
    from jit_mas.experience import ExperienceStore
    from jit_mas.offline import FixtureModel, FixtureModels
    from scripts.models.openai_server import OpenAIServerModel
    from scripts.run_jit_mas import make_pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError('Offline failure integration must not issue real model requests')

    monkeypatch.setattr(OpenAIServerModel, '__call__', forbidden)
    original_call = FixtureModel.__call__
    requests = []
    output = tmp_path / 'runs'

    def fail_executor(self, messages, **kwargs):
        if self.role != 'exec':
            return original_call(self, messages, **kwargs)
        payload = json.loads(messages[1]['content'])
        projected = payload['public_membership_observations']
        paths = list(output.rglob('public_membership_execution_audit.json'))
        assert len(paths) == 1  # This callback runs inside the real executor request.
        persisted = json.loads(paths[0].read_text(encoding='utf-8'))['records']
        assert any(row['full_observation_hash'] == projected['full_observation_hash']
                   and row['model_input_hash'] == digest(projected) for row in persisted)
        assert 'observations' not in projected
        requests.append(payload)
        raise RuntimeError('synthetic executor provider failure')

    monkeypatch.setattr(FixtureModel, '__call__', fail_executor)
    store = ExperienceStore(tmp_path / 'state.sqlite')
    try:
        pipeline = make_pipeline(MASConfig(backend='scripted', max_repairs=0,
            public_membership_observations=True, public_membership_input_format='compact'),
            store, output, fixture_models=FixtureModels())
        with pytest.raises(RuntimeError):
            pipeline.run('evaluate', limit=1)
    finally:
        store.close()
    assert requests
    paths = list(output.rglob('public_membership_execution_audit.json'))
    assert len(paths) == 1
    persisted = json.loads(paths[0].read_text(encoding='utf-8'))['records']
    assert persisted and all(row['full_observation_hash'] == digest(row['observations']) for row in persisted)
    assert all('PRIVATE_CANARY' not in json.dumps(row) for row in persisted)
    assert not list(output.rglob('submission.json')) and not list(output.rglob('evaluation.json'))
