"""Item-to-construct mapping and global settings for the TRFS study.

Data: Ravselj et al. (2024) "Higher Education Students' Early Perceptions of
ChatGPT: Global Survey Data", Mendeley Data V2, doi:10.17632/ymg9nsn6kn.2
(CC BY 4.0). Item codes follow the questionnaire in data/raw/questionnaire.pdf.
All Likert items are coded 1 (strongly disagree) .. 5 (strongly agree).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "final_dataset.xlsx"
PROCESSED_CSV = ROOT / "data" / "processed" / "analytic_sample.csv"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

SEED = 2024

# Items whose wording is a *concern* (agreement = less suitable for AI) are
# reverse-coded as 6 - x before they enter a construct.
REVERSED = {"Q22a", "Q22b", "Q22c", "Q22d", "Q22f", "Q22g", "Q22i"}

# Eight evaluation factors of the framework (Table 2 of the manuscript). The
# final item sets were fixed after comparing candidate mappings on CFA fit,
# AVE and HTMT (see results/tables/measurement_alternatives.csv).
FACTORS = {
    "TE": {"name": "Teaching effectiveness",
           "items": ["Q19d", "Q19f", "Q19g"]},
    "SE": {"name": "Student engagement",
           "items": ["Q25b", "Q25c", "Q25d"]},
    "P":  {"name": "Personalization",
           "items": ["Q26d", "Q26a", "Q26c"]},
    "AC": {"name": "Assessment capability",
           "items": ["Q27a", "Q27b", "Q26j"]},
    "C":  {"name": "Communication",
           "items": ["Q19a", "Q19b", "Q19c"]},
    "A":  {"name": "Adaptability",
           "items": ["Q19h", "Q19i", "Q19j"]},
    "R":  {"name": "Reliability",
           "items": ["Q19e", "Q24f", "Q24g"]},
    "E":  {"name": "Ethical suitability",
           "items": ["Q22a", "Q22b", "Q22c", "Q22d", "Q22f"]},
}
FACTOR_ORDER = list(FACTORS)

# Independent criterion (never used as a FIS input): perceived substitutability
# of the professor by ChatGPT.
#   Q24b  It is easier for me to interact with ChatGPT than with my professors.
#   Q24d  The information I get from ChatGPT is clearer than the one provided
#         by my professors.
TARGET_ITEMS = ["Q24b", "Q24d"]
TARGET = "TSI"  # Teacher Substitution Index, rescaled to 0-100

# Ten teaching activities from the original proposal, each operationalised by
# the questionnaire items that describe ChatGPT performing that activity.
ACTIVITIES = {
    "A1":  {"name": "Content explanation",        "items": ["Q19g", "Q19f", "Q26b"]},
    "A2":  {"name": "Question answering",         "items": ["Q19a", "Q19b", "Q19d"]},
    "A3":  {"name": "Assignment feedback",        "items": ["Q26j", "Q28a", "Q28b"]},
    "A4":  {"name": "Student assessment",         "items": ["Q27a", "Q27b"]},
    "A5":  {"name": "Personalized learning",      "items": ["Q26d", "Q26e"]},
    "A6":  {"name": "Student motivation",         "items": ["Q26f", "Q27i"]},
    "A7":  {"name": "Classroom communication",    "items": ["Q19h", "Q26h"]},
    "A8":  {"name": "Academic guidance",          "items": ["Q27h", "Q27g", "Q27j"]},
    "A9":  {"name": "Problem solving",            "items": ["Q29c", "Q29d", "Q29e"]},
    "A10": {"name": "Emotional & social support", "items": ["Q28f", "Q19c", "Q22g", "Q22i"]},
}

# Decision thresholds on the 0-100 TRFS scale = crossover points of the
# output membership functions (Low/Medium at 35, Medium/High at 65).
T_LOW, T_HIGH = 35.0, 65.0
CATEGORIES = ["Human involvement required", "Partial AI assistance", "Full AI replacement"]

DEMOGRAPHICS = {
    "Q2": {1: "Male", 2: "Female", 3: "Other", 4: "Prefer not to say"},
    "Q8": {1: "Undergraduate", 2: "Postgraduate (Master)", 3: "Doctoral"},
    "Q10": {1: "Arts & Humanities", 2: "Social Sciences", 3: "Applied Sciences",
            4: "Natural & Life Sciences"},
    "Q11": {1: "Traditional", 2: "Online", 3: "Blended"},
    "Q14": {1: "ChatGPT-3.5 (free)", 2: "ChatGPT-4 (paid)", 3: "Both"},
}


def all_model_items():
    items = set(TARGET_ITEMS)
    for d in FACTORS.values():
        items.update(d["items"])
    for d in ACTIVITIES.values():
        items.update(d["items"])
    return sorted(items, key=lambda s: (int(s[1:3]), s[3:]))
