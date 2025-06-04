# src/evals/treatment_utils.py

"""
Shared treatment evaluation utilities:
- procedure_checker
- treatment_alternative_procedure_checker
- extract_treatment
"""

import re
from typing import List, Union

from src.utils.nlp import keyword_positive


def treatment_alternative_procedure_checker(operation_keywords: List[dict], text: str) -> bool:
    """Check if a treatment procedure alternative exists in the text."""
    for alternative_operations in operation_keywords:
        op_loc = alternative_operations["location"]
        for op_mod in alternative_operations["modifiers"]:
            for sentence in text.split("."):
                if keyword_positive(sentence, op_loc) and keyword_positive(sentence, op_mod):
                    return True
    return False


def procedure_checker(valid_procedures: List[Union[str, int]], done_procedures: List[str]) -> bool:
    """Check if a valid procedure exists in the done procedures."""
    for valid_procedure in valid_procedures:
        if isinstance(valid_procedure, int):
            if valid_procedure in done_procedures:
                return True
        else:
            for done_procedure in done_procedures:
                if keyword_positive(done_procedure, valid_procedure):
                    return True
    return False


def extract_treatment(text):
    match = re.search(r"Treatment:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else None
