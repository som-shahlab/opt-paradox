
"""
Run a *built* LangGraph pipeline over an entire dataset of patients.

This module is shared by both **single-agent** and **multi-agent** workflows.

Core helpers
------------
process_patient(stream)
    Consume the streamed graph output for one patient, record the full
    conversation, and collect tool-usage metrics.

process_all_patients(graph, dataset_path, log_dir, *, max_workers=1, …)
    Iterate over every patient in the dataset, feed them into the compiled
    `graph`, and save a per-patient JSON log.  
    Supports optional thread-pool concurrency for network-bound LLM calls and
    shows periodic progress / GC.

Typical usage (inside run_single_agent.py / run_multi_agent.py)
----------------------------------------------------------------
    from pipeline_runner import process_all_patients
    graph = build_graph(...)          # compile your LangGraph
    process_all_patients(graph,
                         dataset_path="…/master_patient_data_val.json",
                         log_dir="logs/…",
                         max_workers=4)

All paths, prompts, and logging settings live elsewhere; this file is purely
about **executing** a compiled pipeline cleanly and reproducibly.
"""

import os
import time
import json
import gc
import re
import torch
from concurrent.futures import ThreadPoolExecutor
import orjson

from langgraph.graph import StateGraph
from rich.panel import Panel

from .logging import safe_print, file_console, console
from ..prompts import QUERY_PROMPT


# should I get these from config.py
PRINT_EVERY = 10
GC_EVERY = 50 


def process_patient(stream):
    """
    Consume the agent’s stream for one patient, record the full conversation
    and basic tool-usage metrics.

    Args:
        stream: generator yielding dicts with {"messages": [...]}.

    Returns:
        final_message (str): last model output
        metrics (dict): counts & requested‐lists
        conversation (list of dict): [{"type": ..., "content": ...}, ...]
    """
    final_message = None
    conversation = []

    # tool‐usage flags & counters
    physical_exam_first = False
    physical_exam_requested = False
    first_tool_call_processed = False

    lab_tests_requested = []
    physical_exam_maneuvers_requested = []
    imaging_requested = []

    lab_count = 0
    imaging_count = 0
    tool_call_count = 0
    physical_exam_count = 0
    
    first_printed = False

    for packet in stream:
        message = packet["messages"][-1]
        msg_type = message.__class__.__name__
        msg_content = message.content
        
        # only print the very first turn
        if not first_printed:
            safe_print(msg_content, msg_type)
            if file_console:
                file_console.print(Panel(msg_content, title=msg_type))
            first_printed = True

        # record the turn
        conversation.append({
            "type": msg_type,
            "content": msg_content
        })

        # inspect any tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for call in message.tool_calls:
                tool_call_count += 1
                rd = call.get("args", {}).get("response_dict", {})
                action = rd.get("action", "").lower()

                # first tool call?
                if not first_tool_call_processed:
                    physical_exam_first = (action == "physical examination")
                    first_tool_call_processed = True

                # per‐action bookkeeping
                if action == "physical examination":
                    physical_exam_requested = True
                    physical_exam_count += 1
                    maneuvers = [
                        m.strip().lower()
                        for m in rd.get("action_input", "").split(",")
                        if m.strip()
                    ]
                    physical_exam_maneuvers_requested.extend(maneuvers)

                elif action == "laboratory tests":
                    lab_count += 1
                    raw = rd.get("action_input", "")
                    # split on commas that are _not_ followed (before the next ')' ) by a ')'
                    pattern = r',\s*(?![^()]*\))'
                    tests = [t.strip().lower() for t in re.split(pattern, raw) if t.strip()]
                    lab_tests_requested.extend(tests)

                elif action == "imaging":
                    imaging_count += 1
                    imgs = [
                        i.strip().lower()
                        for i in rd.get("action_input", "").split(",")
                        if i.strip()
                    ]
                    imaging_requested.extend(imgs)


        final_message = msg_content
        
    # after the loop, print the final turn
    safe_print(final_message, msg_type)
    if file_console:
        file_console.print(Panel(final_message, title=msg_type))

    metrics = {
        "physical_exam_first": physical_exam_first,
        "physical_exam_requested": physical_exam_requested,
        "lab_tests_requested": lab_tests_requested,
        "imaging_requested": imaging_requested,
        "physical_exam_maneuvers_requested": physical_exam_maneuvers_requested,
        "lab_count": lab_count,
        "imaging_count": imaging_count,
        "tool_call_count": tool_call_count,
        "physical_exam_count": physical_exam_count,
    }

    return final_message, metrics, conversation


def process_all_patients(
    graph: StateGraph,
    dataset_path: str,
    log_dir: str,
    *,
    patient_data=None,
    max_workers: int = 1
) -> None:
    """
    Run the agent on every patient, save per-patient JSON logs,
    and track only crash/error flags for quick debugging.
    """

    # ------------- I/O & initialisation --------------------------------------
    t0_global = time.time()
    with open(dataset_path) as f:
        data_json = json.load(f) if patient_data is None else patient_data

    patients       = list(data_json.items())
    total_patients = len(patients)
    console.log(f"[green]▶ Starting evaluation on {total_patients} patients")
    if file_console:
        file_console.print(f"▶ Starting evaluation on {total_patients} patients")

    os.makedirs(log_dir, exist_ok=True)

    # ------------- helper ----------------------------------------------------
    def _run_single(item):
        pid, pdata = item
        # build the initial user query
        query = QUERY_PROMPT.format(patient_history=pdata["Patient History"])
        inputs = {
            "messages": [("user", query)],
            "patient_id": pid,
            "iteration": 0
        }

        t0 = time.time()
        try:
            final_txt, metrics, conversation = process_patient(
                graph.stream(inputs, stream_mode="values")
            )
            error_flag = "[ERROR] exceeded retries" in final_txt
        except Exception as e:
            console.log(f"[red]Patient {pid} crashed in stream: {e}")
            final_txt    = f"[ERROR] stream failure: {e}"
            metrics      = {}
            conversation = []
            error_flag   = True

        dur = time.time() - t0

        # ───── save per-patient log ─────────────────────────────────────────
        meta = {
            "final":           final_txt,
            "metrics":         metrics,
            "error":           error_flag,
            "duration_sec":    dur,
            "gold_diagnosis":  pdata["Discharge Diagnosis"].lower()
        }
        
        output_path = os.path.join(log_dir, f"{pid}.txt")
        # open in BINARY mode so we can write the raw bytes from orjson.dumps()
        with open(output_path, "wb") as f:
            # 1) write a JSON front-matter
            f.write(orjson.dumps(meta) + b"\n\n")

            # 2) then the human-readable transcript
            if conversation:
                for turn in conversation:
                    f.write(f"{turn['type']}:\n".encode("utf-8"))
                    f.write(turn['content'].encode("utf-8") + b"\n\n")
            else: # log the error at least
                f.write(final_txt.encode("utf-8") + b"\n")


        return dur

    # ------------- optional thread pool --------------------------------------
    executor = (
        ThreadPoolExecutor(max_workers=max_workers)
        if max_workers > 1
        else None
    )
    iterator = (
        executor.map(_run_single, patients)
        if executor
        else map(_run_single, patients)
    )

    # ------------- main loop --------------------------------------------------
    total_time = 0.0
    for idx, dur in enumerate(iterator, start=1):
        total_time += dur
        if idx % PRINT_EVERY == 0:
            avg = total_time / idx
            console.log(f"processed {idx}/{total_patients} "
                        f"(last {dur:.1f}s, avg {avg:.1f}s)")
            if file_console:
                file_console.print(f"processed {idx}/{total_patients} "
                                   f"(last {dur:.1f}s, avg {avg:.1f}s)")
        if idx % GC_EVERY == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if executor:
        executor.shutdown(wait=True)

    # ------------- summary ---------------------------------------------------
    elapsed = time.time() - t0_global
    console.log(f"[green]▶ Finished {total_patients} patients in {elapsed:.1f}s")
    if file_console:
        file_console.print(f"Finished {total_patients} patients in {elapsed:.1f}s")
