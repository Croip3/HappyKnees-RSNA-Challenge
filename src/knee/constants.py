"""Fixed competition facts (column names, label order, planes)."""

STUDY = "StudyInstanceUID"
SERIES = "SeriesInstanceUID"
REPORT = "Report"

# Submission column order (must match sample_submission.csv exactly).
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
N_LABELS = len(LABELS)

PLANES = ["sagittal", "coronal", "axial"]
PLANE_ID = {p: i for i, p in enumerate(PLANES)}

# Columns of train_series.csv / test_series.csv
PLANE_COL = "Anatomical_Plane"
FLUID_COL = "Fluid_Sensitive"
FATSAT_COL = "Fat_Suppression"

LLM_STATES = ["positive", "negative", "uncertain", "not_mentioned"]
