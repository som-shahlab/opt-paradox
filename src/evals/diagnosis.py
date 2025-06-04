# src/evals/diagnosis.py

"""
Pathology name mappings and diagnosis evaluation utilities.
"""

from typing import List, Dict
from src.utils.nlp import keyword_positive, remove_punctuation, is_negated
from fuzzywuzzy import fuzz
import re
from src.utils.logging import console, file_console


# Define pathologies for fuzzy matching
PATHOLOGIES = ["appendicitis", "pancreatitis", "cholecystitis", "diverticulitis"]

# Alternative pathology names for each condition
ALTERNATIVE_PATHOLOGY_NAMES: Dict[str, List[Dict[str, List[str]]]] = {
    "appendicitis": [
        {"location": "appendi", "modifiers": ["gangren", "infect", "inflam", "abscess", "rupture", "necros", "perf"]},
    ],
    "cholecystitis": [
        {"location": "gallbladder", "modifiers": ["gangren", "infect", "inflam", "abscess", "necros", "perf"]},
        {"location": "cholangitis", "modifiers": ["cholangitis"]},
    ],
    "diverticulitis": [
        {"location": "diverticul", "modifiers": ["inflam", "infect", "abscess", "perf", "rupture"]},
    ],
    "pancreatitis": [
        {"location": "pancrea", "modifiers": ["gangren", "infect", "inflam", "abscess", "necros"]},
    ],
}

# Gracious alternative pathology names for each condition
GRACIOUS_ALTERNATIVE_PATHOLOGY_NAMES: Dict[str, List[Dict[str, List[str]]]] = {
    "appendicitis": [],
    "cholecystitis": [
        {"location": "acute gallbladder", "modifiers": ["disease", "attack"]},
        {"location": "acute biliary", "modifiers": ["colic"]},
    ],
    "diverticulitis": [
        {"location": "acute colonic", "modifiers": ["perfor"]},
        {"location": "sigmoid", "modifiers": ["perfor"]},
        {"location": "sigmoid", "modifiers": ["colitis"]},
    ],
    "pancreatitis": [],
}

def check_diagnosis_match(correct_diagnosis: str, diagnosis: str) -> bool:
    """
    Performs fuzzy matching, negation checking, and alternative diagnosis evaluation.
    
    Args:
        correct_diagnosis (str): The primary correct diagnosis.
        diagnosis (str): The predicted diagnosis.

    Returns:
        bool: Whether the diagnosis is correct (including gracious alternatives).
    """
    if not diagnosis:  # empty or None
        return False
    correct_diagnosis = correct_diagnosis.lower()
    diagnosis = diagnosis.lower()

    alternative_names = ALTERNATIVE_PATHOLOGY_NAMES[correct_diagnosis]
    gracious_alternative_names = GRACIOUS_ALTERNATIVE_PATHOLOGY_NAMES[correct_diagnosis]

    # Remove punctuation before checking
    diagnosis_clean = remove_punctuation(diagnosis)

    # Use fuzzy substring matching for primary pathology name
    similarity_score = fuzz.partial_ratio(correct_diagnosis, diagnosis_clean)
    is_present = similarity_score > 90  # High similarity threshold

    # Check if the diagnosis is negated
    is_negated_diagnosis = is_negated(diagnosis, correct_diagnosis)

    # Initial correctness check: Must be present and NOT negated
    is_correct = is_present and not is_negated_diagnosis

    # Check alternative pathology names
    if not is_correct:
        for alt in alternative_names:
            patho_loc = alt["location"]
            for patho_mod in alt["modifiers"]:
                if (
                    patho_loc in diagnosis_clean
                    and patho_mod in diagnosis_clean
                    and keyword_positive(diagnosis, patho_loc)
                    and keyword_positive(diagnosis, patho_mod)
                ):
                    is_correct = True
                    break
            if is_correct:
                break

    # Check gracious alternative pathology names (for more lenient matches)
    if not is_correct:
        for gracious_alt in gracious_alternative_names:
            patho_loc = gracious_alt["location"]
            for patho_mod in gracious_alt["modifiers"]:
                if (
                    patho_loc in diagnosis_clean
                    and patho_mod in diagnosis_clean
                    and keyword_positive(diagnosis, patho_loc)
                    and keyword_positive(diagnosis, patho_mod)
                ):
                    is_correct = True
                    break
            if is_correct:
                break

    # Debug print for logging results
    console.print(
        f"[bold blue]Gold diagnosis: {correct_diagnosis} | Model diagnosis: {diagnosis} | "
        f"Similarity: {similarity_score}% | Negated: {is_negated_diagnosis} | "
        f"Correct: {is_correct}"
    )

    # Also log to file if file_console exists
    if 'file_console' in globals() and file_console:
        file_console.print(
            f"Gold diagnosis: {correct_diagnosis} | Model diagnosis: {diagnosis} | "
            f"Similarity: {similarity_score}% | Negated: {is_negated_diagnosis} | "
            f"Correct: {is_correct}"
        )

    return is_correct

def match_pathology(diagnosis: str) -> str:
    """
    Matches a diagnosis to one of the predefined pathologies using fuzzy matching.
    
    Args:
        diagnosis (str): The diagnosis text to match.
        
    Returns:
        str: The matched pathology name if found, None otherwise.
    """
    for pathology in PATHOLOGIES:
        if fuzz.partial_ratio(diagnosis.lower(), pathology) > 80:
            return pathology
    return None

def parse_ranked_diagnoses(text: str) -> List[str]:
    """
    Extract up to 5 *distinct* diagnoses from text.
    Specifically looks for "Final Diagnosis (ranked)" format.
    
    Returns:
        List[str]: List of up to 5 distinct diagnoses.
    """
    ranked_block = re.search(
        r"\**Final Diagnosis\s*\(ranked\)\s*:\**\s*\n(.*?)(?=\n\s*\n|Treatment:)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    
    if ranked_block:
        diagnosis_content = ranked_block.group(1)
        numbered_diagnoses = re.findall(
            r'^\s*(\d+)[\.:\)]\s*(.*?)(?=^\s*\d+[\.:\)]|$)', 
            diagnosis_content,
            re.MULTILINE | re.DOTALL
        )
        if numbered_diagnoses:
            diags = []
            for num, diag in numbered_diagnoses:
                diag = diag.strip()
                match = re.search(r'^(.*?)(?:\s+-+\s+|\s*:\s+).*$', diag)
                if match:
                    diag = match.group(1).strip()
                cleaned = remove_punctuation(diag.lower())
                if cleaned and cleaned not in diags:
                    diags.append(cleaned)
                if len(diags) == 5:
                    break
            if diags:
                return diags

    block = re.search(
        r"Final Diagnosis\s*:\s*(.*?)(?=\n\s*\n|Treatment:)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not block:
        return []

    raw_lines = [
        ln.strip(" -*\t")
        for ln in block.group(1).splitlines()
        if ln.strip()
    ]

    diags = []
    for ln in raw_lines:
        if re.match(r"(treatment|thought)\b", ln, re.I):
            break
        ln = re.sub(r"^\d+\.\s*", "", ln)
        ln = re.split(r"[-:]", ln, 1)[0].strip()
        cleaned = remove_punctuation(ln.lower())
        if cleaned and cleaned not in diags:
            diags.append(cleaned)
        if len(diags) == 5:
            break

    return diags

def parse_diagnosis(prediction: str) -> str:
    """
    Takes prediction string and parses it for a single diagnosis.
    """
    custom_parsing = False
    regex = r"Final Diagnosis:\s*\**([A-Za-z\s\-]+)\**"

    matches = re.findall(regex, prediction, flags=re.IGNORECASE)
    if matches:
        diagnosis = matches[-1].strip()

        # Remove Llama2 Chat intros
        modify_check = diagnosis
        diagnosis = re.sub(r"^Based on.*:\n\n", "", diagnosis)
        if modify_check != diagnosis:
            custom_parsing = True

        # Strip extra sections
        for section in [
            "rationale", "note", "recommendation", "explanation",
            "finding", "other.*diagnos.*include", "other.*diagnos.*considered(?: were)?",
            "management", "action", "plan", "reasoning", "assessment",
            "justification", "tests", "additional diagnoses", "notification",
            "impression", "background", "additional findings include",
        ]:
            diagnosis = re.sub(
                rf"{section}[s]?:.*", "", diagnosis,
                flags=re.IGNORECASE | re.DOTALL
            )

        # Lists with numbers
        match = re.search(r"^1\.(.*)", diagnosis, flags=re.MULTILINE)
        if match:
            diagnosis = match.group(1).strip()
            custom_parsing = True
            diagnosis = re.sub(r"[-:].*", "", diagnosis)

        # Lists with stars
        match = re.search(r"^\*(.*)", diagnosis, flags=re.MULTILINE)
        if match:
            diagnosis = match.group(1).strip()
            custom_parsing = True
            diagnosis = re.sub(r"[-:] .*", "", diagnosis)

        # Remove trailing explanation
        modify_check = diagnosis
        diagnosis = re.sub(r"\n\n.*", "", diagnosis)
        if modify_check != diagnosis:
            custom_parsing = True

        # In-sentence extraction
        match = re.search(
            r".*?diagnosis[^.\n]*?\bis\b(.*?)[.\n]", diagnosis,
            flags=re.DOTALL
        )
        if match:
            diagnosis = match.group(1).strip()
            custom_parsing = True

        # "patient has" removal
        modify_check = diagnosis
        diagnosis = re.sub(r".*?patient has", "", diagnosis, count=1, flags=re.DOTALL)
        if modify_check != diagnosis:
            custom_parsing = True

        # Split multiple diagnoses—take first
        diagnoses = re.split(r"[,.\n]|(?:\s*\b(?:and|or|vs[.]?)\b\s*)", diagnosis)
        diagnoses = [d for d in diagnoses if d]
        if len(diagnoses) > 1:
            custom_parsing = True
        diagnosis = diagnoses[0] if diagnoses else ""
        return diagnosis.strip()

    return ""
