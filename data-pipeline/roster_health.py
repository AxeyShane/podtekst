"""
Shared model-health tracking, used by stage_a_generate.py's in-run circuit
breaker and check_roster.py's pre-flight check.

Why this exists: qwen's reliability problem (18/166 calls failed in batch 1)
sat unnoticed until the whole batch was already spent, and dropping it was a
manual, one-off decision. Akshay asked for the general version: once a model
is genuinely failing, stop calling it automatically instead of relying on
someone noticing after the fact.

Health state lives in its own file (config/model_health.json by default),
separate from the hand-curated config/models.json, so scripts can update it
freely without touching human-authored roster notes/pricing comments.

A model is auto-disabled only after DISABLE_AFTER_BAD_RUNS consecutive runs
in which it tripped the in-run circuit breaker -- not after a single failed
call, and not even after a single bad run, so one transient network blip or
rate-limit storm doesn't retire a model that's actually fine. A run in which
the model is attempted and does NOT trip the breaker resets its streak to 0
-- recovery is automatic once a model behaves normally again.
"""

import json
import os

HEALTH_PATH_DEFAULT = "config/model_health.json"
DISABLE_AFTER_BAD_RUNS = 2


def load_health(path=HEALTH_PATH_DEFAULT):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_health(health, path=HEALTH_PATH_DEFAULT):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_active_generators(config, health):
    """Returns (active_generators, skipped) where skipped is a list of
    (slug, reason) for any generator config lists that health has marked
    disabled. Doesn't mutate config or health."""
    active, skipped = [], []
    for g in config["stage_a_generators"]:
        state = health.get(g["slug"], {})
        if state.get("disabled"):
            skipped.append((g["slug"], state.get("disabled_reason", "auto-disabled after repeated failures")))
        else:
            active.append(g)
    return active, skipped


def record_run_result(health, slug, tripped_breaker, failure_rate=None, run_label=None):
    """Updates health in place for one model after one run/smoke-test. Returns
    True if this call is what newly crossed the disable threshold (so the
    caller can print a loud one-time notice), False otherwise."""
    state = health.setdefault(slug, {"consecutive_bad_runs": 0, "disabled": False})
    if tripped_breaker:
        state["consecutive_bad_runs"] = state.get("consecutive_bad_runs", 0) + 1
        state["last_bad_run"] = run_label
        state["last_failure_rate"] = failure_rate
        if state["consecutive_bad_runs"] >= DISABLE_AFTER_BAD_RUNS and not state.get("disabled"):
            state["disabled"] = True
            state["disabled_reason"] = (
                f"tripped the circuit breaker in {state['consecutive_bad_runs']} consecutive "
                f"runs (last: {run_label})"
            )
            return True
    else:
        state["consecutive_bad_runs"] = 0
    return False


def reinstate(health, slug):
    """Manually clear a model's disabled state. Returns True if it was disabled."""
    state = health.get(slug)
    if not state or not state.get("disabled"):
        return False
    state["disabled"] = False
    state["consecutive_bad_runs"] = 0
    state.pop("disabled_reason", None)
    return True
