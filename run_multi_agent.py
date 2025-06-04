#!/usr/bin/env python3
import os
import json
import gc
import torch
import argparse
from typing import Any, Dict

from src.config import CONFIG
from src.models import load_model
from src.agents.multi_agent import build_graph
from src.utils.pipeline_runner import process_all_patients
from src.utils.logging import console, initialize_file_logging

def main(args: Dict[str, Any]) -> None:
    """
    1) Load patient JSON for the requested split
    2) Initialize optional file logging
    3) Load the four LLMs (info, interpretation, matcher, diagnosis)
    4) Build the multi-agent LangGraph
    5) Run the graph on every patient (via process_all_patients)
    6) Save token usage stats
    7) Clean up
    """
    # ---------- resolve dataset path ----------------------------------------
    base_path = CONFIG.paths.dataset_base_path
    split     = args["dataset_type"]
    if split not in {"train", "val", "test"}:
        raise ValueError(f"dataset_type must be train | val | test, got {split}")

    dataset_path = os.path.join(base_path, f"master_patient_data_{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"✘ dataset not found: {dataset_path}")

    # ---------- logging ------------------------------------------------------
    if args.get("log_to_file", False):
        initialize_file_logging(
            args.get(
                "log_filename",
                f"log_{args['model_id_info']}_{args['model_id_interpretation']}_{args['model_id_diagnosis']}_{split}.txt"
            )
        )
    console.log(
        f"▶ Evaluating Info:{args['model_id_info']}  "
        f"Interpretation:{args['model_id_interpretation']}  "
        f"Matcher:{args['model_id_matcher']}  "
        f"Diagnosis:{args['model_id_diagnosis']}  on **{split}** split…"
    )

    # ---------- load models --------------------------------------------------
    info_llm           = load_model(model_id=args["model_id_info"], matcher=False)
    interpretation_llm = load_model(model_id=args["model_id_interpretation"], matcher=False)
    matcher_llm        = load_model(model_id=args["model_id_matcher"], matcher=True)
    diagnosis_llm      = load_model(model_id=args["model_id_diagnosis"], matcher=False)

    # ---------- build & run --------------------------------------------------
    with open(dataset_path) as f:
        patient_data = json.load(f)

    graph = build_graph(
        info_llm=info_llm,
        interpretation_llm=interpretation_llm,
        matcher_llm=matcher_llm,
        diagnosis_llm=diagnosis_llm,
        patient_data=patient_data
    )

    log_dir = os.path.join(
        "logs",
        f"multi_{args['model_id_info']}_{args['model_id_interpretation']}_{args['model_id_diagnosis']}_{split}"
    )
    process_all_patients(
        graph,
        dataset_path=dataset_path,
        log_dir=log_dir,
        max_workers=args.get("concurrency", 1)
    )

    # ---------- Gather token statistics --------------------------------------
    token_stats_path = os.path.join(log_dir, "token_usage.json")
    token_stats = {
        "info_model":             args["model_id_info"],
        "interpret_model":   args["model_id_interpretation"],
        "matcher_model":          args["model_id_matcher"],
        "diagnosis_model":        args["model_id_diagnosis"],
        "info_input_tokens":      info_llm.total_input_tokens,
        "info_output_tokens":     info_llm.total_output_tokens,
        "interpret_input_tokens": interpretation_llm.total_input_tokens,
        "interpret_output_tokens":interpretation_llm.total_output_tokens,
        "matcher_input_tokens":   matcher_llm.total_input_tokens,
        "matcher_output_tokens":  matcher_llm.total_output_tokens,
        "diagnosis_input_tokens": diagnosis_llm.total_input_tokens,
        "diagnosis_output_tokens":diagnosis_llm.total_output_tokens,
    }
    with open(token_stats_path, "w", encoding="utf-8") as tf:
        json.dump(token_stats, tf, indent=2)
    console.log(f"✅ Token stats saved to {token_stats_path}")

    # ---------- final cleanup ------------------------------------------------
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.log("✓ Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clinical-Agent Multi-Agent Runner")
    platform_choices = ["gpt", "gpt-4.1", "gpt-4.1-mini", "claude",
                        "gemini-flash", "gemini", "llama", "o3-mini", "deepseek"]

    p.add_argument("--model_id_info", required=True, help=f"one of {', '.join(platform_choices)}")
    p.add_argument("--model_id_interpretation", required=True, help=f"one of {', '.join(platform_choices)}")
    p.add_argument("--model_id_matcher", required=True, help=f"one of {', '.join(platform_choices)}")
    p.add_argument("--model_id_diagnosis", required=True, help=f"one of {', '.join(platform_choices)}")
    p.add_argument("--dataset_type", default="val", choices=["train", "val", "test"])
    p.add_argument("--log_to_file", action="store_true")
    p.add_argument("--log_filename")
    p.add_argument("--concurrency", type=int, default=1, help="#threads for network-bound inference (1 = serial)")

    args_ns = p.parse_args()
    main(vars(args_ns))
