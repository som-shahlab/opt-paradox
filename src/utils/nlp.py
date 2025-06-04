import spacy
import string
from negspacy.negation import Negex
from typing import List, Union
from fuzzywuzzy import fuzz

# Load spaCy model
nlp = spacy.load("en_core_sci_lg")

# Register and add the Negex component
nlp.add_pipe("negex", config={"chunk_prefix": ["no"]}, last=True)

def keyword_positive(sentence: str, keyword: str) -> bool:
    """Check if a keyword is positively stated in a sentence (not negated)."""
    doc = nlp(sentence)
    for e in doc.ents:
        if keyword.lower() in e.text.lower():
            return not e._.negex
    return keyword.lower() in sentence.lower()

def remove_punctuation(input_string: str) -> str:
    """Remove punctuation from a string."""
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)

def contains(keyword: str, strings: List[str]) -> bool:
    """Check if a keyword is present in any string from a list."""
    return any(keyword_positive(string, keyword) for string in strings)

def is_negated(text: str, keyword: str) -> bool:
    """Check if a keyword appears in the text and is negated."""
    doc = nlp(text)
    try:
        for ent in doc.ents:
            if fuzz.partial_ratio(keyword.lower(), ent.text.lower()) > 90:
                return ent._.negex
        return False
    except Exception:
        return False
