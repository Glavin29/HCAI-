import pandas as pd

FEATURE_LABELS = {
    "Experience_Years": "Experience",
    "education_score": "Education",
    "skills_score": "Skills",
    "certification_score": "Certifications",
}


def candidate_profile(candidate: pd.Series) -> dict[str, object]:
    return {
        "Name": candidate.get("Name"),
        "Gender": candidate.get("Gender"),
        "Education": candidate.get("Education_Level"),
        "Experience": candidate.get("Experience_Years"),
        "Skills": candidate.get("Skills"),
        "Certifications": candidate.get("Certifications"),
        "Predicted Role": candidate.get("predicted_role"),
        "Model Confidence": round(candidate.get("model_confidence", 0) * 100, 2),
        "Final Score": round(candidate.get("candidate_score", 0), 2),
    }


def score_breakdown(
    candidate: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Signal": "Predicted role",
                "Value": candidate.get("predicted_role"),
                "Explanation": "The model's most likely job-title class for this resume.",
            },
            {
                "Signal": "Model confidence",
                "Value": f"{candidate.get('model_confidence', 0) * 100:.2f}%",
                "Explanation": "The probability assigned to the predicted role; this becomes the ranking score.",
            },
            {
                "Signal": "Experience",
                "Value": candidate.get("Experience_Years", 0),
                "Explanation": "Numeric feature used by the model, not a manually weighted score.",
            },
            {
                "Signal": "Education",
                "Value": candidate.get("Education_Level"),
                "Explanation": "Categorical and numeric education signals used by the model.",
            },
            {
                "Signal": "Skills count",
                "Value": candidate.get("skills_score", 0),
                "Explanation": "Counted skills are used together with TF-IDF text features.",
            },
            {
                "Signal": "Certification count",
                "Value": candidate.get("certification_score", 0),
                "Explanation": "Certification count is used as a numeric model feature.",
            },
        ]
    )


def candidate_strengths(candidate: pd.Series) -> list[str]:
    strengths = []

    if candidate.get("Experience_Years", 0) >= 5:
        strengths.append("Strong professional experience")
    if candidate.get("education_score", 0) >= 3:
        strengths.append("Advanced education level")
    if candidate.get("skills_score", 0) >= 5:
        strengths.append("Broad skill coverage")
    if candidate.get("certification_score", 0) >= 2:
        strengths.append("Multiple relevant certifications")

    return strengths or ["Balanced profile with no single dominant factor"]
