"""
Builds a crewai.LLM from a config/models.json entry, honoring the same
provider/api_model discount-routing fields stage_a_generate.py reads directly
(see that script's call_model() and the 2026-09-20 discount-routing patch).
One place for this logic so every agent script stays in sync with the roster
file instead of hand-rolling its own OpenRouter wiring.
"""

import json
import os
from pathlib import Path

from crewai import LLM

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.json"

DEFAULT_TEMPERATURE = 0.3  # judgment tasks; sourcing/guideline scripts override per-call


def load_models_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_llm(model_cfg: dict, temperature: float = DEFAULT_TEMPERATURE) -> LLM:
    """model_cfg is one entry from config/models.json -- a stage_a_generators item,
    or stage_b_prefilter_model / seed_generator_model / stage_c_coverage_model.
    Must have at least a 'slug' key; 'provider' and 'api_model' are optional
    overrides, same meaning as in stage_a_generate.py."""

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key "
            "(same requirement as stage_a_generate.py / generate_seeds.py / "
            "stage_b_prefilter.py)."
        )

    slug = model_cfg.get("api_model") or model_cfg["slug"]
    additional_params = {}
    if model_cfg.get("provider"):
        # Passed straight through to OpenRouter's request body via litellm's
        # additional_params passthrough -- same {"order": [...], "allow_fallbacks": true}
        # shape used everywhere else in this pipeline (config/models.json, stage_a_generate.py).
        additional_params["provider"] = model_cfg["provider"]

    return LLM(
        model=f"openrouter/{slug}",
        api_key=api_key,
        temperature=temperature,
        additional_params=additional_params or None,
    )


def build_llm_by_role(role_key: str, temperature: float = DEFAULT_TEMPERATURE) -> LLM:
    """Convenience wrapper for the single-model roles keyed directly in
    config/models.json (stage_b_prefilter_model, seed_generator_model,
    stage_c_coverage_model). For stage_a_generators (plural, the Annotator
    roster) use load_models_config()["stage_a_generators"] + build_llm() per
    entry instead -- see agent_defs.build_annotator_agents()."""

    cfg = load_models_config()
    if role_key not in cfg:
        raise KeyError(f"'{role_key}' not found in {CONFIG_PATH} (has: "
                        f"{[k for k in cfg if not k.startswith('_')]})")
    return build_llm(cfg[role_key], temperature=temperature)
