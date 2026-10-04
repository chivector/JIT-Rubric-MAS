"""Opt-in writer context forwards public conditions without routing or private fields."""
import json
from types import SimpleNamespace

from jit_mas.execution import _apply_public_membership_guidance
from jit_mas.schemas import PublicTask

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
