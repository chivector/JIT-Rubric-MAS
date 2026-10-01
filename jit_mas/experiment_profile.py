"""Enforce the registered shared inference envelope before any formal request."""


def validate_execution_profile(config, suite):
    profile = suite["execution_profile"]
    fields = ("max_agents", "max_parallel", "team_max_calls", "local_rounds", "candidates",
              "max_repairs", "max_model_calls", "max_total_tokens", "max_tool_calls", "execution_timeout")
    if config.backend != "native_jit" or config.available_tools:
        raise ValueError("Registered controlled-evidence runs require native_jit and no actor external tools")
    for field in fields:
        if getattr(config, field) != profile[field]:
            raise ValueError(f"Configuration {field} differs from the registered shared profile")
    if set(config.models) != set(profile["output_tokens"]):
        raise ValueError("All registered model roles must be specified")
    for role, tokens in profile["output_tokens"].items():
        model = config.models[role]
        expected = profile["judge_model" if role == "judge" else "model"]
        if (model.model != expected or model.max_tokens != tokens or model.temperature != profile["temperature"]
                or model.timeout != profile["request_timeout"]):
            raise ValueError(f"Model role {role} differs from the registered model/resource profile")
    return config
