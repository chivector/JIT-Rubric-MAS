from pathlib import Path
from types import SimpleNamespace
import json, sys
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from jit_mas.experience import ExperienceStore
from jit_mas.schemas import SplitManifest, digest
from scripts.run_rr_two_arm_pilot import _config, _manifest, _pipeline, _safe_error, write_json
root = Path.cwd(); output = root / sys.argv[2]; output.mkdir(parents=True, exist_ok=False)
original = _manifest(root / "paper/experiments/joint_task_splits_v5.json")
public_validation_id = original.validation[0]
manifest = SplitManifest(test=[public_validation_id]); split_path = output / "preflight_split.json"; write_json(split_path, manifest)
args = SimpleNamespace(data=root / "outputs/researchrubrics_live_20260930/processed_data.jsonl", split_path=split_path,
    exec_model="deepseek-v4-flash-vision", exec_endpoint="https://composure-presuming-comrade.ngrok-free.dev/v1",
    judge_model="deepseek-v4-flash-vision", judge_endpoint="https://composure-presuming-comrade.ngrok-free.dev/v1", timeout=180,
    judge_max_tokens=4096, judge_attempts=2, judge_parallel=1, max_inflight_requests=1, closed_book=True,
    structured_output="json_object", planning_string_max_length=2048, planning_communication_max_length=1024,
    planning_array_max_items=64)
config = _config(args); store = ExperienceStore(output / "state.sqlite")
try:
    pipeline = _pipeline(config, store, output / "runs", args, manifest, None); initial = digest(store.snapshot())
    write_json(output / "metadata.json", {"purpose": "Unscored engineering preflight on public VAL input; excluded from formal checkpoints and TEST.",
        "original_split": "validation", "task_id": public_validation_id, "knowledge_policy": pipeline.knowledge_policy,
        "code_hash": pipeline.code_hash, "state_updates_allowed": False, "judge_requests": 0, "status": "running",
        "execution_model": args.exec_model, "judge_model": args.judge_model, "judge_parallel": 1, "max_inflight_requests": 1})
    result = pipeline.run_task(public_validation_id, store.snapshot(), mode="evaluate", resume=False, defer_evaluation=True)
    assert result["evaluation"] is None; assert digest(store.snapshot()) == initial
    write_json(output / "summary.json", {"status": "passed", "formal_score": None, "judge_requests": 0,
        "task_id": public_validation_id, "run_dir": result["run_dir"], "answer_hash": result["answer_hash"], "budget": result["budget"]})
    print(json.dumps({"status": "passed", "output": str(output), "judge_requests": 0}), flush=True)
except Exception as error:
    write_json(output / "summary.json", {"status": "failed", "formal_score": None, "judge_requests": 0,
        "error_type": type(error).__name__, "error": _safe_error(error), "failure": getattr(error, "jit_mas_run_failure", None)})
    print(json.dumps({"status": "failed", "error_type": type(error).__name__, "output": str(output)}), flush=True); raise
finally: store.close()

