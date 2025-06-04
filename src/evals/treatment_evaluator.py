# src/evals/treatment_evaluator.py

"""
Treatment evaluators for the four acute intra-abdominal conditions.
Each class implements:
  - score_treatment(text): mark which guideline treatments were requested
  - calculate_treatment_percentages(): compute % of cases requesting each treatment
"""

from typing import List, Union
import re

from src.utils.nlp import keyword_positive
from .treatment_utils import procedure_checker, treatment_alternative_procedure_checker
from .treatment_mappings import (
    APPENDECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_APPENDECTOMY_KEYWORDS,
    CHOLECYSTECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_CHOLECYSTECTOMY_KEYWORDS,
    COLECTOMY_PROCEDURES_KEYWORDS,
    ALTERNATE_COLECTOMY_KEYWORDS,
    DRAINAGE_PROCEDURES_KEYWORDS,
    DRAINAGE_PROCEDURES_PANCREATITIS_ICD10,
    ALTERNATE_DRAINAGE_KEYWORDS_DIVERTICULITIS,
    ALTERNATE_DRAINAGE_KEYWORDS_PANCREATITIS,
    ERCP_PROCEDURES_KEYWORDS,
)



class AppendicitisEvaluator:
    """Evaluate the trajectory according to clinical diagnosis guidelines of appendicitis."""

    def __init__(self):
        self.correct_appendicitis_count = 0
        self.appendectomy_count = 0
        self.antibiotics_count = 0
        self.support_count = 0

        self.answers = {
            "Treatment Requested": {
                "Appendectomy": False,
                "Antibiotics": False,
                "Support": False,
            },
            "Treatment Required": {
                "Appendectomy": False,
                "Antibiotics": True,
                "Support": True,
            }
        }

    def score_treatment(self, treatment) -> None:
        ### APPENDECTOMY ###
        self.answers["Treatment Required"]["Appendectomy"] = True

        if procedure_checker(
            APPENDECTOMY_PROCEDURES_KEYWORDS, [treatment]
        ) or treatment_alternative_procedure_checker(
            ALTERNATE_APPENDECTOMY_KEYWORDS, treatment
        ):
            self.answers["Treatment Requested"]["Appendectomy"] = True

        ### ANTIBIOTICS ###
        if keyword_positive(treatment, "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

        ### SUPPORT ###
        if (
            keyword_positive(treatment, "fluid")
            or keyword_positive(treatment, "analgesi")
            or keyword_positive(treatment, "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        self.correct_appendicitis_count += 1
        if self.answers["Treatment Requested"]["Appendectomy"]:
            self.appendectomy_count += 1
        if self.answers["Treatment Requested"]["Antibiotics"]:
            self.antibiotics_count += 1
        if self.answers["Treatment Requested"]["Support"]:
            self.support_count += 1

    def calculate_treatment_percentages(self):
        if self.correct_appendicitis_count > 0:
            appendectomy_percentage = (self.appendectomy_count / self.correct_appendicitis_count) * 100
            antibiotics_percentage = (self.antibiotics_count / self.correct_appendicitis_count) * 100
            support_percentage = (self.support_count / self.correct_appendicitis_count) * 100
        else:
            appendectomy_percentage = antibiotics_percentage = support_percentage = 0

        log_text = (
            f"Appendectomy Requested: {appendectomy_percentage:.2f}% ({self.appendectomy_count}/{self.correct_appendicitis_count})\n"
            f"Antibiotics Requested: {antibiotics_percentage:.2f}% ({self.antibiotics_count}/{self.correct_appendicitis_count})\n"
            f"Support Requested: {support_percentage:.2f}% ({self.support_count}/{self.correct_appendicitis_count})\n"
        )
        return log_text
    
    
class CholecystitisEvaluator:
    """Evaluate the trajectory according to clinical diagnosis guidelines of cholecystitis."""

    def __init__(self):
        self.correct_cholecystitis_count = 0
        self.cholecystectomy_count = 0
        self.antibiotics_count = 0
        self.support_count = 0

        self.answers = {
            "Treatment Requested": {
                "Cholecystectomy": False,
                "Antibiotics": False,
                "Support": False,
            },
            "Treatment Required": {
                "Cholecystectomy": False,
                "Antibiotics": True,
                "Support": True,
            }
        }

    def score_treatment(self, treatment) -> None:
        self.answers["Treatment Required"]["Cholecystectomy"] = True

        if procedure_checker(CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, [treatment]) or treatment_alternative_procedure_checker(ALTERNATE_CHOLECYSTECTOMY_KEYWORDS, treatment):
            self.answers["Treatment Requested"]["Cholecystectomy"] = True

        if keyword_positive(treatment, "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

        if keyword_positive(treatment, "fluid") or keyword_positive(treatment, "analgesi") or keyword_positive(treatment, "pain"):
            self.answers["Treatment Requested"]["Support"] = True

        self.correct_cholecystitis_count += 1
        if self.answers["Treatment Requested"]["Cholecystectomy"]:
            self.cholecystectomy_count += 1
        if self.answers["Treatment Requested"]["Antibiotics"]:
            self.antibiotics_count += 1
        if self.answers["Treatment Requested"]["Support"]:
            self.support_count += 1

    def calculate_treatment_percentages(self):
        if self.correct_cholecystitis_count > 0:
            cholecystectomy_percentage = (self.cholecystectomy_count / self.correct_cholecystitis_count) * 100
            antibiotics_percentage = (self.antibiotics_count / self.correct_cholecystitis_count) * 100
            support_percentage = (self.support_count / self.correct_cholecystitis_count) * 100
        else:
            cholecystectomy_percentage = antibiotics_percentage = support_percentage = 0

        log_text = (
            f"Cholecystectomy Requested: {cholecystectomy_percentage:.2f}% ({self.cholecystectomy_count}/{self.correct_cholecystitis_count})\n"
            f"Antibiotics Requested: {antibiotics_percentage:.2f}% ({self.antibiotics_count}/{self.correct_cholecystitis_count})\n"
            f"Support Requested: {support_percentage:.2f}% ({self.support_count}/{self.correct_cholecystitis_count})\n"
        )
        return log_text
    
class DiverticulitisEvaluator:
    """Evaluate the trajectory according to clinical diagnosis guidelines of diverticulitis."""

    def __init__(self):
        self.correct_diverticulitis_count = 0
        self.colonoscopy_count = 0
        self.antibiotics_count = 0
        self.support_count = 0
        self.drainage_count = 0
        self.colectomy_count = 0

        self.answers = {
            "Treatment Requested": {
                "Colonoscopy": False,
                "Antibiotics": False,
                "Support": False,
                "Drainage": False,
                "Colectomy": False,
            },
            "Treatment Required": {
                "Colonoscopy": True,
                "Antibiotics": True,
                "Support": True,
                "Drainage": False,
                "Colectomy": False,
            }
        }

    def score_treatment(self, treatment) -> None:
        ### COLONOSCOPY ###
        if procedure_checker(COLECTOMY_PROCEDURES_KEYWORDS, [treatment]) or treatment_alternative_procedure_checker(ALTERNATE_COLECTOMY_KEYWORDS, treatment):
            self.answers["Treatment Requested"]["Colonoscopy"] = True

        ### ANTIBIOTICS ###
        if keyword_positive(treatment, "antibiotic"):
            self.answers["Treatment Requested"]["Antibiotics"] = True

        ### SUPPORT ###
        if (
            keyword_positive(treatment, "fluid")
            or keyword_positive(treatment, "analgesi")
            or keyword_positive(treatment, "pain")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        ### DRAINAGE ###
        if procedure_checker(DRAINAGE_PROCEDURES_PANCREATITIS_ICD10, [treatment]) or treatment_alternative_procedure_checker(ALTERNATE_DRAINAGE_KEYWORDS_DIVERTICULITIS, treatment):
            self.answers["Treatment Requested"]["Drainage"] = True

        ### COLECTOMY ###
        if procedure_checker(COLECTOMY_PROCEDURES_KEYWORDS, [treatment]) or treatment_alternative_procedure_checker(ALTERNATE_COLECTOMY_KEYWORDS, treatment):
            self.answers["Treatment Requested"]["Colectomy"] = True

        self.correct_diverticulitis_count += 1
        if self.answers["Treatment Requested"]["Colonoscopy"]:
            self.colonoscopy_count += 1
        if self.answers["Treatment Requested"]["Antibiotics"]:
            self.antibiotics_count += 1
        if self.answers["Treatment Requested"]["Support"]:
            self.support_count += 1
        if self.answers["Treatment Requested"]["Drainage"]:
            self.drainage_count += 1
        if self.answers["Treatment Requested"]["Colectomy"]:
            self.colectomy_count += 1

    def calculate_treatment_percentages(self):
        if self.correct_diverticulitis_count > 0:
            colonoscopy_percentage = (self.colonoscopy_count / self.correct_diverticulitis_count) * 100
            antibiotics_percentage = (self.antibiotics_count / self.correct_diverticulitis_count) * 100
            support_percentage = (self.support_count / self.correct_diverticulitis_count) * 100
            drainage_percentage = (self.drainage_count / self.correct_diverticulitis_count) * 100
            colectomy_percentage = (self.colectomy_count / self.correct_diverticulitis_count) * 100
        else:
            colonoscopy_percentage = antibiotics_percentage = support_percentage = drainage_percentage = colectomy_percentage = 0

        log_text = (
            f"Colonoscopy Requested: {colonoscopy_percentage:.2f}% ({self.colonoscopy_count}/{self.correct_diverticulitis_count})\n"
            f"Antibiotics Requested: {antibiotics_percentage:.2f}% ({self.antibiotics_count}/{self.correct_diverticulitis_count})\n"
            f"Support Requested: {support_percentage:.2f}% ({self.support_count}/{self.correct_diverticulitis_count})\n"
            f"Drainage Requested: {drainage_percentage:.2f}% ({self.drainage_count}/{self.correct_diverticulitis_count})\n"
            f"Colectomy Requested: {colectomy_percentage:.2f}% ({self.colectomy_count}/{self.correct_diverticulitis_count})\n"
        )
        return log_text

class PancreatitisEvaluator:
    """Evaluate the trajectory according to clinical diagnosis guidelines of pancreatitis."""

    def __init__(self):
        self.treatment_count = 0
        self.support_count = 0
        self.drainage_count = 0
        self.ercp_count = 0
        self.cholecystectomy_count = 0

    def score_treatment(self, treatment) -> None:
        ### SUPPORT ###
        if keyword_positive(treatment, "fluid") or keyword_positive(treatment, "analgesi") or keyword_positive(treatment, "pain"):
            self.support_count += 1

        ### DRAINAGE ###
        if procedure_checker(DRAINAGE_PROCEDURES_KEYWORDS, [treatment]) or treatment_alternative_procedure_checker(ALTERNATE_DRAINAGE_KEYWORDS_PANCREATITIS, treatment):
            self.drainage_count += 1

        ### ERCP ###
        if procedure_checker(ERCP_PROCEDURES_KEYWORDS, [treatment]):
            self.ercp_count += 1

        ### CHOLECYSTECTOMY ###
        if procedure_checker(CHOLECYSTECTOMY_PROCEDURES_KEYWORDS, [treatment]) or treatment_alternative_procedure_checker(ALTERNATE_CHOLECYSTECTOMY_KEYWORDS, treatment):
            self.cholecystectomy_count += 1

        self.treatment_count += 1

    def calculate_treatment_percentages(self):
        if self.treatment_count > 0:
            support_percentage = (self.support_count / self.treatment_count) * 100
            drainage_percentage = (self.drainage_count / self.treatment_count) * 100
            ercp_percentage = (self.ercp_count / self.treatment_count) * 100
            cholecystectomy_percentage = (self.cholecystectomy_count / self.treatment_count) * 100
        else:
            support_percentage = drainage_percentage = ercp_percentage = cholecystectomy_percentage = 0

        log_text = (
            f"Support Requested: {support_percentage:.2f}% ({self.support_count}/{self.treatment_count})\n"
            f"Drainage Requested: {drainage_percentage:.2f}% ({self.drainage_count}/{self.treatment_count})\n"
            f"ERCP Requested: {ercp_percentage:.2f}% ({self.ercp_count}/{self.treatment_count})\n"
            f"Cholecystectomy Requested: {cholecystectomy_percentage:.2f}% ({self.cholecystectomy_count}/{self.treatment_count})\n"
        )
        return log_text
    
    
    
