# src/evals/recommended_tests.py

"""
Canonical recommendations for evaluating information requests in four acute
intra-abdominal conditions:

  • Lab tests per pathology (GUIDELINE_LAB_TESTS)
  • Imaging studies per pathology (GUIDELINE_IMAGING_TESTS)
  • Key physical exam maneuvers per pathology (PHYSICAL_EXAM_MANEUVER_SYNONYMS)

These definitions drive the InformationRequestEvaluator to measure coverage
and appropriateness of model requests against standard clinical guidelines.
"""

# =============================================================================
# RECOMMENDED LABS
# =============================================================================

GUIDELINE_LAB_TESTS = {
    "appendicitis": [
        {
            "category": "inflammation",
            "tests": [
                {
                    "canonical": "white blood cell count (WBC)",
                    "contained_in": ["complete blood count (CBC)", "cbc with differential"],
                },
                {
                    "canonical": "c-reactive protein (CRP)",
                    "contained_in": [],
                },
            ],
        },
    ],
    "cholecystitis": [
        {
            "category": "inflammation",
            "tests": [
                {
                    "canonical": "white blood cell count (WBC)",
                    "contained_in": ["complete blood count (CBC)", "cbc with differential"],
                },
                {
                    "canonical": "c-reactive protein (CRP)",
                    "contained_in": [],
                },
            ],
        },
        {
            "category": "cbds_risk",
            "tests": [
                {
                    "canonical": "alanine transaminase (ALT)",
                    "contained_in": ["comprehensive metabolic panel (CMP)", "liver function panel (LFP)", "liver enzymes", "liver function test (LFT)"],
                },
                {
                    "canonical": "aspartate transaminase (AST)",
                    "contained_in": ["comprehensive metabolic panel (CMP)", "liver function panel (LFP)", "liver enzymes", "liver function test (LFT)"],
                },
                {
                    "canonical": "alkaline phosphatase (ALP)",
                    "contained_in": ["comprehensive metabolic panel (CMP)", "liver function panel (LFP)", "liver enzymes", "liver function test (LFT)"],
                },
                {
                    "canonical": "gamma glutamyltransferase (GGT)",
                    "contained_in": ["liver function panel (LFP)", "liver enzymes", "liver function test (LFT)"],
                },
                {
                    "canonical": "bilirubin",
                    "contained_in": ["comprehensive metabolic panel (CMP)", "liver function panel (LFP)", "liver function test (LFT)"],
                },
            ],
        },
    ],
    "diverticulitis": [
        {
            "category": "inflammation",
            "tests": [
                {
                    "canonical": "white blood cell count (WBC)",
                    "contained_in": ["complete blood count (CBC)", "cbc with differential"],
                },
                {
                    "canonical": "c-reactive protein (CRP)",
                    "contained_in": [],
                },
            ],
        },
    ],
    "pancreatitis": [
        {
            "category": "serum_enzymes",
            "tests": [
                {
                    "canonical": "lipase",
                    "contained_in": ["serum lipase"],
                },
                {
                    "canonical": "amylase",
                    "contained_in": ["serum amylase"],
                },
            ],
        },
        {
            "category": "other_markers",
            "tests": [
                {"canonical": "c-reactive protein (CRP)", "contained_in": []},
                {"canonical": "hematocrit",               "contained_in": ["complete blood count (CBC)"]},
                {"canonical": "blood urea nitrogen (BUN)", "contained_in": ["basic metabolic panel (BMP)", "comprehensive metabolic panel (CMP)"]},
                {"canonical": "procalcitonin",             "contained_in": []},
                {"canonical": "serum triglycerides",       "contained_in": []},
                {"canonical": "calcium",                   "contained_in": ["basic metabolic panel (BMP)", "comprehensive metabolic panel (CMP)"]},
            ],
        },
    ],
}

# =============================================================================
# RECOMMENDED IMAGING
# =============================================================================

GUIDELINE_IMAGING_TESTS = {
    "appendicitis": [
        {
            "category": "initial_imaging",
            "options": [
                {"canonical": "abdominal ultrasound (US)", "contained_in": ["ultrasound abdomen", "ultrasound (abdomen)", "abdominal ultrasound"]},
            ],
        },
        {
            "category": "if_inconclusive",
            "options": [
                {"canonical": "ct scan of the abdomen and pelvis", "contained_in": ["ct abdomen", "ct pelvis", "ct abdomen/pelvis", "abdominal ct", "pelvic ct"]},
            ],
        },
        {
            "category": "if_ct_contraindicated",
            "options": [
                {"canonical": "mri (abdomen)", "contained_in": ["mri abdomen", "abdominal mri"]},
            ],
        },
    ],
    "cholecystitis": [
        {
            "category": "initial_imaging",
            "options": [
                {"canonical": "abdominal ultrasound (US)", "contained_in": ["ultrasound abdomen", "ultrasound (abdomen)", "abdominal ultrasound"]},
            ],
        },
        {
            "category": "if_inconclusive",
            "options": [
                {"canonical": "hida scan (cholescintigraphy)", "contained_in": ["hida scan", "hepatobiliary iminodiacetic acid scan", "cholescintigraphy"]},
                {"canonical": "mri (abdomen)",                "contained_in": ["mri abdomen", "abdominal mri"]},
            ],
        },
        {
            "category": "less_frequent",
            "options": [
                {"canonical": "ct scan (abdomen)", "contained_in": ["ct abdomen", "abdominal ct"]},
            ],
        },
    ],
    "diverticulitis": [
        {
            "category": "initial_imaging",
            "options": [
                {"canonical": "ct scan of the abdomen and pelvis", "contained_in": ["ct abdomen", "ct pelvis", "ct abdomen/pelvis", "abdominal ct", "pelvic ct"]},
            ],
        },
        {
            "category": "if_unavailable_or_contraindicated",
            "options": [
                {"canonical": "abdominal ultrasound (US)", "contained_in": ["ultrasound abdomen", "ultrasound (abdomen)", "abdominal ultrasound"]},
                {"canonical": "mri (abdomen/pelvis)",     "contained_in": ["mri abdomen", "mri pelvis", "mri abdomen/pelvis", "abdominal mri", "pelvic mri"]},
            ],
        },
    ],
    "pancreatitis": [
        {
            "category": "initial_imaging",
            "options": [
                {"canonical": "abdominal ultrasound (US)", "contained_in": ["ultrasound abdomen", "ultrasound (abdomen)", "abdominal ultrasound"]},
            ],
        },
        {
            "category": "if_doubt_exists",
            "options": [
                {"canonical": "ct scan (abdomen)",                  "contained_in": ["ct abdomen", "abdominal ct"]},
            ],
        },
        {
            "category": "if_severe",
            "options": [
                {"canonical": "contrast-enhanced ct (ce-ct)",      "contained_in": ["contrast ct abdomen", "ce-ct abdomen", "enhanced ct scan", "ct scan with contrast"]},
                {"canonical": "mri (abdomen)",                     "contained_in": ["mri abdomen", "abdominal mri"]},
            ],
        },
        {
            "category": "screen_for_cbds",
            "options": [
                {"canonical": "magnetic resonance cholangiopancreatography (mrcp)", "contained_in": ["mrcp", "magnetic resonance cholangiopancreatography"]},
                {"canonical": "endoscopic ultrasound (eus)",                    "contained_in": ["eus", "endoscopic ultrasonography"]},
            ],
        },
    ],
}

# =============================================================================
# Recommended Physical Exam Maneuvers
# =============================================================================

PHYSICAL_EXAM_MANEUVER_SYNONYMS = {
    "appendicitis": [
        "mcburney",
        "mcburney's",
        "mcburney point",
        "mcburney's point",
        "point of mcburney",
        "mcburney tenderness",
        "right iliac tenderness",
        "tenderness at mcburney",
        "tenderness at mcburney's point"
    ],
    "cholecystitis": [
        "murphy",
        "murphy's",
        "murphy sign",
        "murphy's sign",
        "inspiratory arrest",
        "halted inspiration",
        "interruption of breath",
        "breath catching",
        "respiratory arrest with palpation"
    ],
    "diverticulitis": [
        "left lower quadrant",
        "llq",
        "sigmoid",
        "sigmoid tenderness",
        "tenderness over sigmoid",
        "left iliac fossa",
        "lif",
        "left-sided abdominal tenderness",
        "sigmoid colon tenderness"
    ],
    "pancreatitis": [
        "epigastric",
        "epigastrium",
        "upper abdominal",
        "mid-upper abdomen",
        "central upper abdomen",
        "transabdominal tenderness",
        "midline upper abdomen",
        "central abdominal tenderness",
        "mid-epigastric"
    ]
}

