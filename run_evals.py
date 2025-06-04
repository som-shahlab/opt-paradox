import os
import glob
import json
import csv
from collections import defaultdict
from pathlib import Path

from rich.console import Console

from src.config import CONFIG
from src.evals.information_evaluator import InformationRequestEvaluator
from src.evals.lab_cost_evaluator import LabCostEvaluator
from src.evals.treatment_evaluator import (
    AppendicitisEvaluator,
    CholecystitisEvaluator,
    PancreatitisEvaluator,
    DiverticulitisEvaluator,
)
from src.evals.treatment_utils import extract_treatment
from src.evals.diagnosis    import (
    match_pathology,
    parse_ranked_diagnoses,
    parse_diagnosis,
    check_diagnosis_match,
)
from src.evals.lab_interpretation_evaluator import LabInterpretationEvaluator

from src.utils.logging import console
from src.evals.token_cost import compute_token_cost




def test_logs(
    log_dir: str,
    *,
    csv_out: str | None = None,
    skip_fuzzy: bool = False,
    no_llm_match: bool = False
) -> None:
    """
    Read per-patient .txt logs from `log_dir/*.txt`, extract the JSON
    front-matter and metrics, update all evaluators, write a CSV of
    patient-level results, and produce a summary TXT.
    """
    pattern   = os.path.join(log_dir, "*.txt")
    log_files = sorted(glob.glob(pattern))
    total     = len(log_files)
    console.log(f"[green]▶ Evaluating {total} patients from {log_dir}")

    # CSV setup (default into results/<logname>.csv)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_out = csv_out or os.path.join(results_dir, f"{os.path.basename(log_dir)}.csv")
    summary_file = os.path.join(results_dir, f"summary_{os.path.basename(log_dir)}.txt")
    
    fieldnames = [
        "patient_id","correct","top1","top3","top5","processing_sec",
        "tool_calls","lab_req","img_req","exam_req",
        "estimated_lab_cost", "status"
    ]
    csv_f      = open(csv_out, "w", newline="")
    writer     = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    # ------------- light‑weight global counters ------------------------------
    totals             = defaultdict(float)   # numeric
    totals["lab_cost"] = 0
    totals["top1"]     = totals["top3"] = totals["top5"] = 0
    totals["failed"]   = 0
    pathology_total    = defaultdict(int)   # per‑pathology denominators
    pathology_correct  = defaultdict(int)
    
    # Patient Dataset
    split = Path(log_dir).name.split("_")[-1]
    if split not in {"train","val","test"}:
        raise ValueError(f"dataset_type must be train|val|test, got {split}")
    dataset_path = os.path.join(CONFIG.paths.dataset_base_path,
                                f"master_patient_data_{split}.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"✘ dataset not found: {dataset_path}")

    # Evaluators
    interp_eval = LabInterpretationEvaluator(
    dataset_path,
    skip_fuzzy=skip_fuzzy,
    no_llm_match=no_llm_match
    )
    info_eval     = InformationRequestEvaluator()
    lab_cost_eval = LabCostEvaluator()
    app_eval      = AppendicitisEvaluator()
    cho_eval      = CholecystitisEvaluator()
    pan_eval      = PancreatitisEvaluator()
    div_eval      = DiverticulitisEvaluator()


    # Per‐patient loop
    for idx, path in enumerate(log_files, start=1):
        pid = Path(path).stem
        text = open(path, "r", encoding="utf-8").read()
        front, _, _ = text.partition("\n\n")
        meta = json.loads(front)

        m           = meta["metrics"]
        dur         = meta["duration_sec"]
        final_txt   = meta["final"]
        error_flag  = meta["error"]
        gold_diag   = meta["gold_diagnosis"].lower()

        # ───── diagnosis scoring ────────────────────────────────────────────
        if error_flag:
            top1 = top3 = top5 = False
            correct = False
            pathology = None
            totals["failed"] += 1
        else:
            pathology = match_pathology(gold_diag)

            if pathology is None:
                top1 = top3 = top5 = False

            else:
                # Extract up to 5 *ranked* diagnoses from the final transcript
                ranked_diags = parse_ranked_diagnoses(final_txt)
                # If none found, fall back to parsing a single final diagnosis
                if not ranked_diags:
                    ranked_diags = [parse_diagnosis(final_txt)]

                console.log(f"{pid}  ranked_diags = {ranked_diags}")

                top1 = check_diagnosis_match(pathology, ranked_diags[0])
                top3 = any(check_diagnosis_match(pathology, d) for d in ranked_diags[:3])
                top5 = any(check_diagnosis_match(pathology, d) for d in ranked_diags[:5])

            # bookkeeping
            totals["top1"] += int(top1)
            totals["top3"] += int(top3)
            totals["top5"] += int(top5)
            correct = top1   # “micro accuracy”

        # ───── global counters (single increment) ───────────────────────────
        # lab interp
        interp_eval.update(pid, text)
        
        # lab cost
        lab_cost = lab_cost_eval.update(m.get("lab_tests_requested",[]))
        totals["lab_cost"] += lab_cost

        # other counters
        totals["patients"]      += 1
        totals["correct_cases"] += int(correct)
        totals["time"]          += dur
        totals["physical_exam_first"]     += int(m.get("physical_exam_first",False))
        totals["physical_exam_requested"] += int(m.get("physical_exam_requested",False))

        if pathology:
            pathology_total[pathology]   += 1
            pathology_correct[pathology] += int(correct)

            # Info‐request
            info_eval.update(
                pathology,
                m.get("lab_tests_requested",[]),
                m.get("requested_imaging",[]),
                m.get("physical_exam_maneuvers_requested",[]),
                m.get("physical_exam_count",0)
            )

            # Treatment
            treat_txt = extract_treatment(final_txt) or ""
            if pathology=="appendicitis":   app_eval.score_treatment(treat_txt)
            elif pathology=="cholecystitis": cho_eval.score_treatment(treat_txt)
            elif pathology=="pancreatitis":  pan_eval.score_treatment(treat_txt)
            elif pathology=="diverticulitis":div_eval.score_treatment(treat_txt)

        # ───── CSV row ──────────────────────────────────────────────────────
        writer.writerow({
            "patient_id"         : pid,
            "correct"            : "yes" if correct else "no",
            "top1"               : int(top1),
            "top3"               : int(top3),
            "top5"               : int(top5),
            "processing_sec"     : round(dur,2),
            "tool_calls"         : m.get("tool_call_count",0),
            "lab_req"            : m.get("lab_count",0),
            "img_req"            : m.get("imaging_count",0),
            "exam_req"           : m.get("physical_exam_count",0),
            "estimated_lab_cost": round(lab_cost,2),
            "status"             : "failed" if error_flag else "ok",
        })

    csv_f.close()
    

    # ------------- derive metrics --------------------------------------------
    micro_acc = 100 * totals["correct_cases"] / totals["patients"]
    if pathology_total:
        macro_acc = (
            sum(100*pathology_correct[p]/pathology_total[p]
                for p in pathology_total if pathology_total[p])
            / len(pathology_total)
        )
    else:
        macro_acc = 0.0

    phys_first = 100 * totals["physical_exam_first"]     / totals["patients"]
    phys_any   = 100 * totals["physical_exam_requested"] / totals["patients"]
    info_metrics = info_eval.compute_metrics()
    interp_metrics = interp_eval.compute_metrics()


    # lab costs
    lab_cost_per_patient = totals["lab_cost"] / totals["patients"]
    
    # Token Cost
    totals["token_cost"] = compute_token_cost(log_dir)

    elapsed_time = totals["time"]
    top1_acc = 100 * totals["top1"] / totals["patients"]
    top3_acc = 100 * totals["top3"] / totals["patients"]
    top5_acc = 100 * totals["top5"] / totals["patients"]

    # Treatment summaries
    app_stats = app_eval.calculate_treatment_percentages()
    cho_stats = cho_eval.calculate_treatment_percentages()
    pan_stats = pan_eval.calculate_treatment_percentages()
    div_stats = div_eval.calculate_treatment_percentages()

    # Write summary TXT
    with open(summary_file,"w") as f:
        f.write("==== Summary Statistics ====\n")
        # TODO: Change this to token cost
        f.write("Cost Metrics:\n")
        f.write(f"Total lab cost: ${totals['lab_cost']:.2f}\n")
        f.write(f"Lab cost per patient: ${lab_cost_per_patient:.2f}\n")
        f.write(f"Total token cost: ${totals['token_cost']:.2f}\n")

        f.write(f"Results CSV: {csv_out}\n")
        f.write(f"Elapsed time: {elapsed_time}\n")
        f.write(f"Failed cases: {int(totals['failed'])}\n\n")

        f.write(f"Micro Average (Overall Diagnosis Accuracy): {micro_acc:.2f}%\n")
        f.write(f"Macro Average (Per-Pathology Accuracy):     {macro_acc:.2f}%\n\n")
        f.write("Pathology-wise Accuracy:\n")
        # ——— write pathology-wise breakdown safely ———————————————————————————————————————
        if pathology_total:
            f.write("Pathology-wise Accuracy:\n")
            for p, total in pathology_total.items():
                if total:
                    acc = 100*pathology_correct[p] / total
                    f.write(f"{p}: {acc:.2f}% ({pathology_correct[p]}/{total})\n")
                else:
                    f.write(f"{p}: no cases\n")
        else:
            f.write("Pathology-wise Accuracy: no cases to report\n")
        f.write("\nAppendicitis:\n"   + app_stats)
        f.write("\nCholecystitis:\n"  + cho_stats)
        f.write("\nDiverticulitis:\n" + div_stats)
        f.write("\nPancreatitis:\n"   + pan_stats)

        f.write("\n\nInformation Request Evaluation:\n")
        f.write(f"Physical Exam First: {phys_first:.2f}%\n")
        f.write(f"Physical Exam Any  : {phys_any:.2f}%\n\n")

        f.write(f"Top-1 accuracy : {top1_acc:.2f}%\n")
        f.write(f"Top-3 accuracy : {top3_acc:.2f}%\n")
        f.write(f"Top-5 accuracy : {top5_acc:.2f}%\n\n")

        f.write("Information Request Breakdown:\n")
        for k,v in info_metrics.items():
            f.write(f"{k}: {v}\n")
            
        f.write("\nLab Interpretation Evaluation:\n")
        for k, v in interp_metrics.items():
            f.write(f"{k}: {v}\n")

    console.rule("[bold green]Evaluation complete")
    console.print(f"Micro accuracy : {micro_acc:.2f}%")
    console.print(f"Total token cost     : ${totals['token_cost']:.2f}")
    console.print(f"Total lab cost       : ${totals['lab_cost']:.2f}")
    console.print(f"Results CSV    : {csv_out}")
    console.print(f"Summary file   : {summary_file}")
    console.print(f"Elapsed        : {elapsed_time}")
    console.print(f"Failed cases   : {int(totals['failed'])}")
    console.print(f"Top-1 accuracy : {top1_acc:.2f}%")
    console.print(f"Top-3 accuracy : {top3_acc:.2f}%")
    console.print(f"Top-5 accuracy : {top5_acc:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-hoc evaluation of clinical-agent logs")
    parser.add_argument("--log_dir", required=True, help="Directory containing per-patient JSON logs")
    parser.add_argument("--csv_out", default=None, help="Path to write the per-patient CSV (defaults to results_<log_dir>.csv)")
    parser.add_argument("--skip_fuzzy", action="store_true", help="turn off fuzzy matching in lab interp evaluator")
    parser.add_argument("--no_llm_match", action="store_true", help="turn off GPT-synonym matching in lab interp evaluator")

    args = parser.parse_args()
    test_logs(
        log_dir=args.log_dir,
        csv_out=args.csv_out,
        skip_fuzzy=args.skip_fuzzy,
        no_llm_match=args.no_llm_match
    )
