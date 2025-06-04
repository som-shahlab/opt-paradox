#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lab-interpretation evaluator – GPT-synonym + abbr-map edition
------------------------------------------------------------
• Exact-match → abbreviation-map → fuzzy (dynamic threshold) → GPT synonym
  (unless --no_llm_match / --skip_fuzzy flags are set).
• Resolves failures such as “WBC” ↔ “White Blood Cells”.
• CLI flags:
    --no_llm_match   : disable GPT step
    --skip_fuzzy     : skip fuzzy step (EM → abbr-map → GPT)
Wraps the original CLI into a class-based evaluator with update()/compute_metrics().
"""
from __future__ import annotations
import os
import re
import json
import argparse
from ast import literal_eval
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict

import pandas as pd
from thefuzz import fuzz, process   # pip install thefuzz[speedup]

# ---------------- project config -------------------------------------------
try:
    from config import BASE_PATHS, OPENAI_API_KEY
except ImportError:
    BASE_PATHS, OPENAI_API_KEY = {}, ""

# optional shared helpers ----------------------------------------------------
try:
    from clinical_agent import load_model
    from langchain_core.messages import HumanMessage
except ImportError:
    load_model = HumanMessage = None

# ---------------- abbreviation dictionary ----------------------------------
ABBREV_MAP = {
    "wbc": "White Blood Cells",
    "rbc": "Red Blood Cells",
    "hgb": "Hemoglobin",
    "hct": "Hematocrit",
    "plt": "Platelet Count",
    "ast": "Aspartate Aminotransferase (AST)",
    "alt": "Alanine Aminotransferase (ALT)",
    "crp": "C-Reactive Protein",
    "esr": "Erythrocyte Sedimentation Rate",
    "cmp": "Comprehensive Metabolic Panel",
    "bun": "Urea Nitrogen",
    "total bilirubin": "Bilirubin, Total",
    "free t4": "Thyroxine (T4), Free",
    "direct bilirubin": "Bilirubin, Direct"
}

# ---------------- build GPT synonym matcher --------------------------------
def _build_matcher():
    if load_model is None:
        return None
    try:
        return load_model(backend="azure", model_id="gpt", matcher=True)
    except Exception:
        return None

MATCHER_LLM = _build_matcher()

@lru_cache(maxsize=2048)
def llm_equivalent(a: str, b: str) -> bool:
    if MATCHER_LLM is None:
        return False
    prompt = (
        "You are an expert clinical terminologist. Answer ONLY 'yes' or 'no'.\n\n"
        f"Are these two lab test names equivalent (including abbreviations)?\n"
        f"Test-1: {a}\nTest-2: {b}\nAnswer:"
    )
    try:
        resp = MATCHER_LLM.invoke([HumanMessage(content=prompt)]).strip().lower()
        return resp.startswith("y")
    except Exception:
        return False

# ---------------- interpretation normaliser --------------------------------
def normalize_interpretation(it: Any) -> str:
    if not isinstance(it, str):
        return "unknown"
    it = it.strip().lower()
    syn = {
        "high": ["high", "elevated", "slightly elevated", "increased", "borderline high"],
        "low":  ["low", "decreased", "reduced", "slightly low", "borderline low"],
        "normal": ["normal", "within normal limits", "wnl"],
        "unknown": ["unknown", "n/a", "not available", "none", ""]
    }
    for k, v in syn.items():
        if it in v:
            return k
    return it

SAFE_PARSE_PATTERN = re.compile(
    r'(?:Lab Interpretation"?\s*:)?\s*'
    r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})',
    re.DOTALL | re.I
)

# ---------------- evaluator class -----------------------------------------
class LabInterpretationEvaluator:
    """
    Stateful evaluator: call update(pid, transcript) for each patient,
    then compute_metrics() for overall lab-interpretation accuracy.
    """
    def __init__(self, patients_json: str, skip_fuzzy: bool = False, no_llm_match: bool = False):
        """patients_json: path to master_patient_data_{split}.json"""
        with open(patients_json, 'r') as f:
            self.patients = json.load(f)
        self.skip_fuzzy = skip_fuzzy
        self.no_llm = no_llm_match
        self.per_patient: Dict[str, Dict[str, int]] = {}
        self.total_correct = 0
        self.total_tests = 0

    def _safe_parse(self, block: str):
        """Clean the Lab Interpretation dict so literal_eval doesn't choke."""
        try:
            return literal_eval(block)
        except Exception:
            txt = block
            # blood pressure like 126/63 → '126/63'
            txt = re.sub(r"(:\s*)(\d+/\d+)", r"\1'\2'", txt)
            # strip trailing % sign
            txt = re.sub(r"(:\s*)(\d+\.?\d*)%", r"\1\2", txt)
            # quote bare NEG / POS
            txt = re.sub(r"(:\s*)(NEG|POS)([\s,}\]])", r"\1'\2'\3", txt, flags=re.I)
            # JSON null → Python None
            txt = re.sub(r"(:\s*)null([\s,}\]])", r"\1None\2", txt, flags=re.I)
            # inequalities >x or <x → x
            txt = re.sub(r"(:\s*)[><]\s*(\d+\.?\d*)", r"\1\2", txt)
            return literal_eval(txt)

    def update(self, pid: str, transcript: str):
        """Extract lab-interpretation blocks from transcript and score them."""
        enriched: Dict[str, Any] = {}
        # ----- harvest ------------------------------------------------------
        for blk in SAFE_PARSE_PATTERN.findall(transcript):
            try:
                parsed = self._safe_parse(blk)
                if isinstance(parsed, dict):
                    # inner dict case
                    for k, v in parsed.items():
                        if isinstance(v, dict):
                            parsed = v
                            break
                    enriched.update(parsed)
            except Exception:
                print(f"❌ parse error in {pid}")

        labs = self.patients.get(pid, {}).get("Laboratory Tests", {})
        low  = self.patients.get(pid, {}).get("Reference Range Lower", {})
        hi   = self.patients.get(pid, {}).get("Reference Range Upper", {})

        correct = 0
        total = 0

        # ----- enrich & score ----------------------------------------------
        for name, info in enriched.items():
            if not isinstance(info, dict):
                continue

            # match test name to key
            key = None
            # exact match
            for k in labs:
                if k.lower() == name.lower():
                    key = k
                    break
            # abbreviation map
            if not key:
                mapped = ABBREV_MAP.get(name.lower())
                if mapped and mapped in labs:
                    key = mapped
            # fuzzy match
            if not key and not self.skip_fuzzy:
                thr = 70 if len(name) <= 4 else 85
                for cand, score in process.extract(name, labs.keys(), scorer=fuzz.partial_ratio, limit=3):
                    if score >= thr:
                        key = cand
                        break
            # LLM synonym
            if not key and not self.no_llm:
                for cand in labs:
                    if llm_equivalent(name, cand):
                        key = cand
                        break
            if not key:
                print(f"⚠️  No match found for '{name}' (patient {pid})")
                continue

            total += 1
            # compute ground truth interpretation
            try:
                raw = labs[key]
                val = float(str(raw).replace("NEG.", "0").split()[0])
                lo, hi_val = low.get(key), hi.get(key)
                if isinstance(lo, (int, float)) and isinstance(hi_val, (int, float)):
                    if val < lo:
                        gt = "low"
                    elif val > hi_val:
                        gt = "high"
                    else:
                        gt = "normal"
                else:
                    gt = "unknown"
            except Exception:
                gt = "unknown"

            # compare with model-provided interpretation
            model_it = info.get("interpretation", "")
            if normalize_interpretation(model_it) == normalize_interpretation(gt):
                correct += 1

        # record per-patient and aggregate totals
        self.per_patient[pid] = {"correct": correct, "total": total}
        self.total_correct += correct
        self.total_tests += total

    def compute_metrics(self) -> Dict[str, float]:
        """Return overall lab-interpretation accuracy metrics."""
        overall = (self.total_correct / self.total_tests) if self.total_tests else 0.0
        return {
            "lab_interp_overall_accuracy": overall,
            "lab_interp_total_tests":      self.total_tests,
            "lab_interp_correct":          self.total_correct,
        }

# ---------------- standalone CLI -------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--dataset_type", choices=["train", "val", "test"], default="test")
    ap.add_argument("--txt_dir", required=True)
    ap.add_argument("--output_csv", default="enriched_lab_interpretations.csv")
    ap.add_argument("--accuracy_csv", default="per_patient_accuracy.csv")
    ap.add_argument("--overall_accuracy_file", default="overall_interpretation_accuracy.txt")
    ap.add_argument("--no_llm_match", action="store_true")
    ap.add_argument("--skip_fuzzy", action="store_true")
    args = ap.parse_args()

    base_dir = BASE_PATHS.get(args.user, "data")
    patients_json = os.path.join(base_dir, f"master_patient_data_{args.dataset_type}.json")
    evaluator = LabInterpretationEvaluator(patients_json,
                                           skip_fuzzy=args.skip_fuzzy,
                                           no_llm_match=args.no_llm_match)

    for fn in os.listdir(args.txt_dir):
        if not fn.endswith(".txt"):
            continue
        pid = os.path.splitext(fn)[0]
        txt = open(os.path.join(args.txt_dir, fn)).read()
        evaluator.update(pid, txt)

    # write per-patient enriched CSV
    pd.DataFrame([{"Patient ID": p, "Enriched Lab Interpretation": i}
                  for p, i in evaluator.per_patient.items()]) \
      .to_csv(args.output_csv, index=False)
    print("CSV →", args.output_csv)

    # write per-patient accuracy CSV
    rows = [{"Patient ID": p, **metrics} for p, metrics in evaluator.per_patient.items()]
    pd.DataFrame(rows).to_csv(args.accuracy_csv, index=False)
    print("Acc CSV →", args.accuracy_csv)

    # overall metrics
    metrics = evaluator.compute_metrics()
    tot = metrics["lab_interp_total_tests"]
    cor = metrics["lab_interp_correct"]
    overall = metrics["lab_interp_overall_accuracy"]
    print(f"Overall Accuracy: {overall:.4%} ({cor}/{tot})")

    with open(args.overall_accuracy_file, "w") as f:
        f.write(f"Overall Accuracy: {overall:.4f}\nCorrect: {cor}\nTotal: {tot}\n")
    print("Overall accuracy file →", args.overall_accuracy_file)
