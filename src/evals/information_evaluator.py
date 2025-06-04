# src/evals/information_evaluator.py

"""
InformationRequestEvaluator
---------------------------

Compute coverage and efficiency metrics for model information‐gathering behavior
(physical exam maneuvers, lab tests, and imaging) against guideline‐based
recommendations for four acute intra-abdominal pathologies.
"""

from collections import defaultdict

from thefuzz import fuzz

from .recommended_tests import (
    GUIDELINE_LAB_TESTS,
    GUIDELINE_IMAGING_TESTS,
    PHYSICAL_EXAM_MANEUVER_SYNONYMS,
)



def exact_or_fuzzy_match(user_str: str, ref_str: str, threshold: int = 80) -> bool:
    """
    Return True if the maximum of fuzz.token_set_ratio and fuzz.partial_ratio 
    between user_str and ref_str is >= threshold.
    """
    user_str = user_str.lower().strip()
    ref_str = ref_str.lower().strip()

    token_score = fuzz.token_set_ratio(user_str, ref_str)
    partial_score = fuzz.partial_ratio(user_str, ref_str)
    return max(token_score, partial_score) >= threshold


def correct_maneuver_requested(requested_maneuvers: list[str], pathology: str, threshold: int = 80) -> bool:
    """
    Returns True if any of the synonyms for the given pathology's
    physical exam maneuver appears in the requested maneuvers list,
    using fuzzy matching.
    """
    synonyms = PHYSICAL_EXAM_MANEUVER_SYNONYMS.get(pathology, [])
    requested_lower = [m.lower().strip() for m in requested_maneuvers]

    for maneuver in requested_lower:
        for synonym in synonyms:
            if fuzz.partial_ratio(maneuver, synonym.lower()) >= threshold:
                return True
    return False


class InformationRequestEvaluator:
    def __init__(self, fuzzy_threshold: int = 90):
        """
        Initialize the evaluator to track information request metrics across patients.

        Args:
            fuzzy_threshold (int): Threshold for fuzzy string matching lab test names.
        """
        
        self.fuzzy_threshold = fuzzy_threshold
        
        # === General Tracking ===
        self.total_patients = 0 
        self.total_physical_exams = 0
        self.num_patients_with_requested_labs = 0
        self.num_patients_with_requested_imaging = 0 
        
        # === Coverage ===
        self.total_possible_to_cover = 0
        self.total_model_covered = 0 
        self.coverage_scores = []
        
        # === Efficiency ===
        self.total_maneuvers = 0
        self.total_labs = 0 
        self.total_imaging = 0 

    def update(self,
               pathology: str,
               requested_labs: list[str],
               requested_imaging: list[str],
               requested_maneuvers: list[str],
               physical_exam_count: int) -> None:
        
        """
        Update the stats for a single patient encounter 
        """
        
        # Update Efficiency Metrics 
        self.total_patients += 1
        self.total_physical_exams += physical_exam_count
        self.total_maneuvers += len(requested_maneuvers)
        self.total_labs += len(requested_labs)
        self.total_imaging += len(requested_imaging)
        if requested_labs:
            self.num_patients_with_requested_labs += 1
        if requested_imaging:
            self.num_patients_with_requested_imaging += 1
        
        # Update Lab Metrics 
        recommended_labs = GUIDELINE_LAB_TESTS.get(pathology, [])
        total_lab_categories = len(recommended_labs)
        covered_lab_categories = 0 
        
        # We are calculating coverage per category and not per test 
        for category_info in recommended_labs:
            category_covered = False
            for test_def in category_info["tests"]:
                canonical = test_def["canonical"].lower()
                synonyms = [canonical] + [x.lower() for x in test_def.get("contained_in", [])]

                found = any(
                    exact_or_fuzzy_match(req_lab, syn, self.fuzzy_threshold)
                    for req_lab in requested_labs
                    for syn in synonyms
                )
                if found:
                    category_covered = True
                    break

            if category_covered:
                covered_lab_categories += 1
                
        # Update Imaging Metrics 
        # Is one of the recmmended imaging requested?
        recommended_imaging = GUIDELINE_IMAGING_TESTS.get(pathology, [])
        all_imaging_options = []
        for cat_info in recommended_imaging:
            # gives a list of imaging definitions
            all_imaging_options.extend(cat_info["options"])
            
        imaging_covered = False 
        for opt_def in all_imaging_options:
            canonical = opt_def["canonical"].lower()
            synonyms = [canonical] + [x.lower() for x in opt_def.get("contained_in", [])]

            found = any(
                exact_or_fuzzy_match(req_img, syn, self.fuzzy_threshold)
                for req_img in requested_imaging
                for syn in synonyms
            )
            if found:
                imaging_covered = True
                break
            
            
        imaging_score = 1 if imaging_covered else 0
        # TODO: Is there a way to evaluate ordering of imaging requests?
         
        # Update Physical Exam Metrics 
        maneuver_score = 1 if correct_maneuver_requested(
            requested_maneuvers, pathology, threshold=self.fuzzy_threshold
        ) else 0
        
        # Overall Coverage Score
        # Each lab category is 1 point if covered; imaging is 1 point if covered; 
        # maneuver is 1 point if correct.
        total_possible = total_lab_categories + 1 + 1
        self.total_possible_to_cover += total_possible
        covered_points = covered_lab_categories + imaging_score + maneuver_score
        self.total_model_covered += covered_points

        coverage_ratio = (covered_points / float(total_possible)) if total_possible else 0.0
        
        self.coverage_scores.append(coverage_ratio) 
        
        
    def compute_metrics(self) -> dict:
        """
        Compute overall metrics across the entire dataset.
        
        Returns a dictionary with the following keys:
        - "Total Patients": Total number of cases processed.
        - "Total # of tool calls": Total number of tools (labs + imaging + maneuvers) requested.
        - "Average number of tools per case": Efficiency metric (tools per patient).
        - "Average Labs per Case": Total labs requested / total patients.
        - "Average Imaging per Case": Total imaging requests / total patients.
        - "Average Maneuvers per Case": Total maneuvers requested / total patients.
        - "Fraction of Patients with Labs Requested": (Patients with at least one lab request) / total patients.
        - "Fraction of Patients with Imaging Requested": (Patients with at least one imaging request) / total patients.
        - "Coverage (Averaged Per Patient)": Average of the per-case coverage scores.
        - "Coverage (Averaged over the entire Dataset)": Overall dataset coverage computed as:
                (Total model-covered points) / (Total possible points to cover)
        """
        # Efficiency Metrics:
        total_tools = self.total_labs + self.total_imaging + self.total_physical_exams
        efficiency_average = total_tools / self.total_patients if self.total_patients else 0.0
        
        # Average labs, imaging, and maneuvers per case:
        avg_labs = self.total_labs / self.total_patients if self.total_patients else 0.0
        avg_imaging = self.total_imaging / self.total_patients if self.total_patients else 0.0
        avg_maneuvers = self.total_maneuvers / self.total_patients if self.total_patients else 0.0
        
        # Fraction (or percentage) of patients with labs and imaging requested:
        frac_patients_labs = (self.num_patients_with_requested_labs / self.total_patients) if self.total_patients else 0.0
        frac_patients_imaging = (self.num_patients_with_requested_imaging / self.total_patients) if self.total_patients else 0.0

        # Coverage Metrics:
        if self.coverage_scores:
            avg_coverage = sum(self.coverage_scores) / len(self.coverage_scores)
        else:
            avg_coverage = 0.0
        overall_coverage = (self.total_model_covered / self.total_possible_to_cover) if self.total_possible_to_cover else 0.0
        
        cov_to_test = avg_coverage / efficiency_average if efficiency_average else 0.0

        metrics = {
            "Total Patients": self.total_patients,
            "Total # of tool calls": total_tools,
            "Average number of tools per case": efficiency_average,
            "Average Labs per Case": avg_labs,
            "Average Imaging per Case": avg_imaging,
            "Average Maneuvers per Case": avg_maneuvers,
            "Fraction of Patients with Labs Requested": frac_patients_labs,
            "Fraction of Patients with Imaging Requested": frac_patients_imaging,
            "Coverage (Averaged Per Patient)": avg_coverage,
            "Coverage (Averaged over the entire Dataset)": overall_coverage,
            "Coverage-to-Test Ratio": cov_to_test,
        }

        return metrics



