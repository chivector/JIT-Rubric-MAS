"""Run the independent v5 software preflight without contacting any model API.

The report is intentionally marked synthetic and cannot authorize formal
execution. A later provider connectivity preflight must replace its serving
identities before ``run_independent_experiment.py --mode run`` is allowed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from jit_mas.config import MASConfig
from jit_mas.pipeline import code_fingerprint
from jit_mas.schemas import digest
from scripts.env_config import load_yaml_config
from scripts.run_jit_mas import make_pipeline
from jit_mas.experience import ExperienceStore


ROLES = ("meta", "global", "local", "exec", "judge")


def _validate_config(config: MASConfig) -> None:
    if config.backend != "native_jit" or config.execution_mode != "iterative_shared_ledger":
        raise ValueError("Independent preflight requires native iterative shared-ledger configuration")
    if any(value is not None for value in (config.team_max_calls, config.max_model_calls, config.max_tool_calls)):
        raise ValueError("Independent preflight requires uncapped model/team/tool calls")
    if config.available_tools or not config.unsafe_local:
        raise ValueError("Independent preflight requires unsafe_local and an empty frozen actor tool list")
    if not all((config.persistent_experience, config.evolving_agent_pool,
                config.explicit_rubrics, config.local_planning, config.local_attribution)):
        raise ValueError("Independent preflight requires both evolution layers and local planning/attribution")
    if set(config.models) != set(ROLES):
        raise ValueError("Independent preflight requires all five model roles")
    for role in ROLES:
        model = config.models[role]
        if not model.model or not model.endpoint or "${" in model.endpoint:
            raise ValueError(f"Missing expanded endpoint/model for {role}")
        if not model.endpoint.startswith(("http://", "https://")):
            raise ValueError(f"Invalid endpoint URL for {role}")


def run_preflight(config_path: Path, output_path: Path) -> dict:
    config = MASConfig.model_validate(load_yaml_config(config_path))
    _validate_config(config)
    # Exercise the full offline fixture pipeline with the registered execution
    # mode, while keeping all requests and benchmark records synthetic.
    with tempfile.TemporaryDirectory(prefix="jit-mas-independent-preflight-") as temporary:
        root = Path(temporary)
        store = ExperienceStore(root / "experience.sqlite")
        try:
            # The labelled fixture provider implements the legacy answer envelope;
            # use it only to verify persistence and pool plumbing. Iterative
            # scheduling itself is covered by the dedicated synthetic protocol
            # tests and is still required in the native config above.
            software_config = config.model_copy(update={"backend": "scripted", "models": {},
                                                        "execution_mode": "single_pass"})
            pipeline = make_pipeline(software_config, store, root / "runs")
            evolved = pipeline.run("evolve", limit=1)
            heldout = pipeline.run("evaluate", limit=1)
            if len(evolved) != 1 or len(heldout) != 1:
                raise ValueError("Synthetic independent pipeline did not complete one EVO and one held-out task")
            snapshot = store.snapshot()
            if snapshot.version < 1 or not snapshot.agent_pool.profiles:
                raise ValueError("Synthetic preflight did not update the evolving agent pool")
            software = {"evolution_tasks": len(evolved), "heldout_tasks": len(heldout),
                        "experience_version": snapshot.version,
                        "agent_pool_members": len(snapshot.agent_pool.profiles), "paid_requests": 0,
                        "fixture_execution_mode": software_config.execution_mode}
        finally:
            store.close()
    served_models = {role: {"requested_model": config.models[role].model,
                            "returned_model": "synthetic-software-fixture", "synthetic": True}
                     for role in ROLES}
    report = {"version": "jit-mas-independent-preflight-v1", "passed": True,
              "synthetic_only": True, "configuration_sha256": digest(config.model_dump(mode="json")),
              "code_fingerprint": code_fingerprint(), "served_models": served_models,
              "software": software, "environment_keys_checked": sorted({model.key_env for model in config.models.values()}),
              "credentials_used": False, "network_requests": 0}
    report["report_sha256"] = digest(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if output_path.exists() and output_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Refusing to replace changed preflight report: {output_path}")
    output_path.write_text(encoded, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_preflight(args.config.resolve(), args.output.resolve())
    print(json.dumps({"passed": report["passed"], "synthetic_only": report["synthetic_only"],
                      "network_requests": report["network_requests"], "report_sha256": report["report_sha256"],
                      "output": str(args.output.resolve())}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
