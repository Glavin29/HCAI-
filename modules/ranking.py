import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.model import train_and_score_candidates


def score_candidates(
    df: pd.DataFrame,
    score_column: str = "candidate_score",
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Train the role-consistency model and score every candidate."""
    scored, metrics = train_and_score_candidates(df)
    if score_column != "candidate_score":
        scored = scored.rename(columns={"candidate_score": score_column})
    return scored, metrics


def rank_candidates(
    df: pd.DataFrame,
    score_column: str = "candidate_score",
) -> pd.DataFrame:
    ranked = df.sort_values(score_column, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def shortlist_top_percent(
    ranked_df: pd.DataFrame,
    percent: float = 0.2,
) -> pd.DataFrame:
    shortlist_size = max(1, int(len(ranked_df) * percent))
    return ranked_df.head(shortlist_size).copy()


def match_job_description(
    df: pd.DataFrame,
    job_description: str,
    text_columns: tuple[str, ...] = ("Skills", "Certifications", "Target_Job_Description"),
) -> pd.DataFrame:
    """Rank candidates by TF-IDF similarity to a pasted job description."""
    matched = df.copy()
    candidate_text = (
        matched.loc[:, list(text_columns)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    corpus = candidate_text.tolist() + [job_description]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(corpus)

    similarity_scores = cosine_similarity(vectors[:-1], vectors[-1]).flatten()
    matched["job_match_score"] = similarity_scores * 100

    return rank_candidates(matched, score_column="job_match_score")


def ranking_metrics(ranked_df: pd.DataFrame, shortlist_df: pd.DataFrame) -> dict[str, float]:
    return {
        "average_score": ranked_df["candidate_score"].mean(),
        "top_score": ranked_df["candidate_score"].max(),
        "shortlist_average_score": shortlist_df["candidate_score"].mean(),
        "shortlist_size": len(shortlist_df),
    }
