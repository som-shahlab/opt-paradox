# src/evals/token_cost.py

import json
from pathlib import Path
from typing import Dict

from src.config import CONFIG

# Flatten SimpleNamespaces into plain dicts
_raw_ct = vars(CONFIG.cost_tracking.cost_table)
COST_TABLE = {k: vars(v) for k, v in _raw_ct.items()}
MODEL_MAP  = vars(CONFIG.cost_tracking.model_cost_mapping)


def _lookup_rates(model_key: str) -> Dict[str, float]:
    mapped = MODEL_MAP.get(model_key, model_key)
    return COST_TABLE.get(mapped, {"input": 0.0, "output": 0.0})


def compute_token_cost(log_dir: str) -> float:
    """
    Reads token_usage.json from `log_dir`, multiplies each
    model's input/output tokens by its per‐token rate, and returns
    the grand total cost.
    """
    stats = json.loads((Path(log_dir) / "token_usage.json").read_text())
    total = 0.0

    for role, count in stats.items():
        if role.endswith("_input_tokens"):
            model_role = role[: -len("_input_tokens")]
            rate_key   = "input"
        elif role.endswith("_output_tokens"):
            model_role = role[: -len("_output_tokens")]
            rate_key   = "output"
        else:
            continue

        # stats should have e.g. "main_model" or "matcher_model"
        model_id = stats.get(f"{model_role}_model")
        rates    = _lookup_rates(model_id)
        total   += count * rates[rate_key]

    return total
