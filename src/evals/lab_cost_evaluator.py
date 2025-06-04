# src/evals/lab_cost_evaluator.py

"""
Aggregate lab‐cost evaluation against Medicare CLFS reimbursement rates.

This module loads the CLFS 2025 fee schedule (data/CLFS 2025 Q2V1.csv by default),
matches each requested lab test to its HCPCS code and rate (via fuzzy matching + aliasing),
and then accumulates both per‐patient and dataset‐level cost metrics.
"""

import re
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
from fuzzywuzzy import process, fuzz

# by default, point to <repo_root>/data/CLFS 2025 Q2V1.csv
REPO_ROOT    = Path(__file__).resolve().parents[2]
DEFAULT_CLFS = REPO_ROOT / "data" / "CLFS 2025 Q2V1.csv"

ALIAS_MAP = {
    "crp": "c reactive protein",
    "esr": "erythrocyte sedimentation rate",
    "cmp": "comprehen metabolic panel",
    "comprehensive metabolic panel": "comprehen metabolic panel",
    "serum lipase": "assay of lipase",
}

def load_clfs(clfs_path: Path = None) -> pd.DataFrame:
    """
    Load the CLFS 2025 CSV, skipping any metadata before the 'YEAR,' header.
    """
    path = clfs_path or DEFAULT_CLFS
    if not path.exists():
        raise FileNotFoundError(f"CLFS file not found: {path}")
    header_row = None
    with open(path, encoding="latin1") as fh:
        for i, line in enumerate(fh):
            if line.lstrip().startswith("YEAR,"):
                header_row = i
                break
    if header_row is None:
        raise ValueError("Could not find 'YEAR,' header in CLFS file")
    df = pd.read_csv(
        path,
        skiprows=header_row,
        dtype=str,
        encoding="latin1",
        engine="python"
    )
    df["RATE"] = df["RATE"].astype(float)
    return df

def clean(text: str) -> str:
    tmp = re.sub(r"[(),]", " ", text.lower())
    return re.sub(r"\s+", " ", tmp).strip()

def build_lookup(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    short_lookup: Dict[str, Any] = {}
    long_lookup: Dict[str, Any] = {}
    for _, row in df.iterrows():
        info = {"hcpcs": row["HCPCS"], "rate": row["RATE"]}
        s = clean(row["SHORTDESC"])
        l = clean(row["LONGDESC"])
        short_lookup[s] = info
        long_lookup[l]  = info
    return short_lookup, {**short_lookup, **long_lookup}

def find_ngram_match(q: str, short_keys: List[str], all_keys: List[str]) -> str:
    toks = q.split()
    for n in range(len(toks), 1, -1):
        for i in range(len(toks) - n + 1):
            gram = " ".join(toks[i:i + n])
            for k in all_keys:
                if gram in k:
                    return k
    if len(toks) == 1:
        tok = toks[0]
        for k in short_keys:
            if tok in k:
                return k
    return ""

def match_test(
    raw: str,
    lookups: Tuple[Dict[str, Any], Dict[str, Any]],
    threshold: int
) -> Dict[str, Any]:
    short_lookup, merged_lookup = lookups
    m = re.match(r"^(?P<base>.+?)\s*\([^)]*\)\s*$", raw)
    base = (m.group("base") if m else raw).strip()
    key = clean(base)
    if key in ALIAS_MAP:
        key = clean(ALIAS_MAP[key])
    sk = list(short_lookup.keys())
    ak = list(merged_lookup.keys())
    sub = find_ngram_match(key, sk, ak)
    if sub:
        inf = merged_lookup[sub]
        return {"requested": raw, "matched_key": sub, "hcpcs": inf["hcpcs"], "rate": inf["rate"], "score": 100}
    best, score = process.extractOne(key, ak, scorer=fuzz.token_set_ratio)
    if score >= threshold:
        inf = merged_lookup[best]
        return {"requested": raw, "matched_key": best, "hcpcs": inf["hcpcs"], "rate": inf["rate"], "score": score}
    return {"requested": raw, "matched_key": best, "hcpcs": None, "rate": None, "score": score}

class LabCostEvaluator:
    """
    Given a list of logged lab request names, compute and aggregate
    total Medicare CLFS cost across patients.
    """
    def __init__(self, clfs_path: Path = None, threshold: int = 70):
        """
        clfs_path: optional path to CLFS CSV (defaults to data/CLFS…)
        threshold: fuzzy matching cutoff
        """
        self.threshold      = threshold
        df                   = load_clfs(clfs_path)
        self.short_lookup, self.merged_lookup = build_lookup(df)
        self.total_cost     = 0.0
        self.num_patients   = 0

    def update(self, tests: List[str]) -> float:
        """
        Add one patient's test list to the running total.
        Returns that patient's cost, too.
        """
        cost = 0.0
        lookups = (self.short_lookup, self.merged_lookup)
        for t in tests:
            match = match_test(t, lookups, self.threshold)
            cost += match.get("rate", 0.0) or 0.0
        self.total_cost   += cost
        self.num_patients += 1
        return cost

    def compute_metrics(self) -> Dict[str, float]:
        """
        Return cumulative and per‐patient average cost:
          - total_cost
          - avg_cost_per_patient
        """
        avg = self.total_cost / self.num_patients if self.num_patients else 0.0
        return {"total_cost": self.total_cost, "avg_cost_per_patient": avg}
