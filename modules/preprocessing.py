from pathlib import Path

import pandas as pd


EDUCATION_SCORE_MAP = {
    "High School": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4,
}


def load_resume_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the resume dataset from disk."""
    return pd.read_csv(csv_path)


def count_list_items(value: object) -> int:
    if pd.isna(value):
        return 0

    cleaned_items = [
        item.strip()
        for item in str(value).split(",")
        if item.strip() and item.strip().lower() != "none"
    ]
    return len(cleaned_items)


def preprocess_resumes(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready, interpretable features from raw resume fields."""
    processed = df.copy()

    processed["education_score"] = (
        processed["Education_Level"]
        .map(EDUCATION_SCORE_MAP)
        .fillna(2)
        .astype(float)
    )

    processed["skills_score"] = processed["Skills"].apply(count_list_items)
    processed["certification_score"] = processed["Certifications"].apply(count_list_items)

    processed["Experience_Years"] = (
        pd.to_numeric(processed["Experience_Years"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

    return processed


def load_and_preprocess(csv_path: str | Path) -> pd.DataFrame:
    return preprocess_resumes(load_resume_data(csv_path))
