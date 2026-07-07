import pandas as pd
import numpy as np


SENSITIVE_OR_PROXY_COLUMNS = [
    "Gender",
    "Age",
    "Institute_Name",
    "Graduation_Year",
    "Field_of_Study",
]


def selection_rate_by_group(
    population_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    group_column: str = "Gender",
) -> pd.DataFrame:
    population_counts = population_df[group_column].fillna("Unknown").value_counts()
    selected_counts = selected_df[group_column].fillna("Unknown").value_counts()

    metrics = (
        pd.DataFrame(
            {
                "pool_count": population_counts,
                "selected_count": selected_counts,
            }
        )
        .fillna(0)
        .astype({"pool_count": int, "selected_count": int})
    )

    metrics["selection_rate_pct"] = (
        metrics["selected_count"] / metrics["pool_count"].replace(0, pd.NA) * 100
    ).fillna(0)

    highest_rate = metrics["selection_rate_pct"].max()
    metrics["impact_ratio"] = (
        metrics["selection_rate_pct"] / highest_rate if highest_rate > 0 else 0
    )

    return metrics.reset_index(names=group_column)


def score_parity_by_group(
    df: pd.DataFrame,
    group_column: str = "Gender",
    score_column: str = "candidate_score",
) -> pd.DataFrame:
    parity = (
        df.assign(**{group_column: df[group_column].fillna("Unknown")})
        .groupby(group_column)[score_column]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    parity["score_gap_from_overall"] = parity["mean"] - df[score_column].mean()
    return parity


def fairness_report(
    population_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    group_column: str = "Gender",
    score_column: str = "candidate_score",
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    rates = selection_rate_by_group(population_df, selected_df, group_column)
    score_parity = score_parity_by_group(population_df, group_column, score_column)

    report = rates.merge(score_parity, on=group_column, how="left")

    min_impact_ratio = report["impact_ratio"].min() if not report.empty else 0
    selection_gap = (
        report["selection_rate_pct"].max() - report["selection_rate_pct"].min()
        if not report.empty
        else 0
    )

    summary = {
        "min_impact_ratio": float(min_impact_ratio),
        "selection_gap_pct": float(selection_gap),
        "status": "Needs review" if min_impact_ratio < 0.8 else "Within 80% rule",
    }

    return report, summary


def compute_gender_fairness_metrics(
    df: pd.DataFrame,
    score_column: str = "candidate_score",
    protected_column: str = "Gender",
    threshold: float | None = None,
) -> dict[str, float | int | str]:
    if protected_column not in df.columns:
        return {
            "fairness_status": "Gender column missing. Fairness metrics not computed."
        }

    if score_column not in df.columns:
        return {
            "fairness_status": "Candidate score column missing. Fairness metrics not computed."
        }

    data = df[[protected_column, score_column]].copy()
    data[protected_column] = data[protected_column].fillna("Unknown").astype(str)
    data[score_column] = pd.to_numeric(data[score_column], errors="coerce")
    data = data.dropna(subset=[score_column])

    if data.empty:
        return {
            "fairness_status": "No valid scores available for fairness metrics."
        }

    if threshold is None:
        threshold = data[score_column].median()

    data["selected"] = data[score_column] >= threshold

    group_stats = {}

    for group, group_data in data.groupby(protected_column):
        group_stats[group] = {
            "count": int(len(group_data)),
            "average_score": float(group_data[score_column].mean()),
            "median_score": float(group_data[score_column].median()),
            "selection_rate": float(group_data["selected"].mean()),
            "selection_rate_pct": float(group_data["selected"].mean() * 100),
        }

    if len(group_stats) < 2:
        return {
            "fairness_status": "Less than two gender groups found. Fairness metrics not computed."
        }

    selection_rates = [stats["selection_rate"] for stats in group_stats.values()]
    average_scores = [stats["average_score"] for stats in group_stats.values()]

    max_selection_rate = max(selection_rates)
    min_selection_rate = min(selection_rates)

    demographic_parity_difference = max_selection_rate - min_selection_rate

    disparate_impact_ratio = (
        min_selection_rate / max_selection_rate if max_selection_rate > 0 else 0
    )

    average_score_difference = max(average_scores) - min(average_scores)

    metrics = {
        "fairness_status": "Gender fairness metrics computed successfully.",
        "protected_attribute": protected_column,
        "threshold_used": float(threshold),
        "gender_groups": ", ".join(group_stats.keys()),
        "demographic_parity_difference": float(demographic_parity_difference),
        "demographic_parity_difference_pct": float(demographic_parity_difference * 100),
        "disparate_impact_ratio": float(disparate_impact_ratio),
        "average_score_difference": float(average_score_difference),
        "fairness_interpretation": (
            "Needs review"
            if disparate_impact_ratio < 0.8 or demographic_parity_difference > 0.1
            else "No major gender selection-rate imbalance detected"
        ),
    }

    for group, stats in group_stats.items():
        clean_group = str(group).lower().replace(" ", "_")

        metrics[f"{clean_group}_count"] = stats["count"]
        metrics[f"{clean_group}_average_score"] = stats["average_score"]
        metrics[f"{clean_group}_median_score"] = stats["median_score"]
        metrics[f"{clean_group}_selection_rate"] = stats["selection_rate"]
        metrics[f"{clean_group}_selection_rate_pct"] = stats["selection_rate_pct"]

    return metrics


def rebalance_shortlist(
    ranked_df: pd.DataFrame,
    top_n: int,
    group_column: str = "Gender",
) -> pd.DataFrame:
    groups = ranked_df[group_column].fillna("Unknown").drop_duplicates().tolist()

    if not groups:
        return ranked_df.head(top_n).copy()

    target_per_group = max(1, top_n // len(groups))
    selected_parts = []

    for group in groups:
        group_candidates = ranked_df[ranked_df[group_column].fillna("Unknown") == group]
        selected_parts.append(group_candidates.head(target_per_group))

    balanced = pd.concat(selected_parts, ignore_index=False)

    if len(balanced) < top_n:
        remaining = ranked_df.drop(index=balanced.index, errors="ignore")
        balanced = pd.concat([balanced, remaining.head(top_n - len(balanced))])

    balanced = balanced.sort_values("candidate_score", ascending=False).head(top_n).copy()
    balanced["adjusted_rank"] = range(1, len(balanced) + 1)

    return balanced


def dataset_fairness_audit(
    df: pd.DataFrame,
    group_column: str = "Gender",
    proxy_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    audit_columns = proxy_columns or SENSITIVE_OR_PROXY_COLUMNS
    available_columns = [column for column in audit_columns if column in df.columns]

    rows = []

    for column in available_columns:
        missing_pct = df[column].isna().mean() * 100
        unique_values = df[column].nunique(dropna=True)

        most_common_share = (
            df[column].value_counts(normalize=True, dropna=True).iloc[0] * 100
            if df[column].notna().any()
            else 0
        )

        rows.append(
            {
                "Field": column,
                "Missing (%)": float(missing_pct),
                "Unique Values": int(unique_values),
                "Largest Group Share (%)": float(most_common_share),
                "Review Focus": _review_focus(column),
            }
        )

    recommendations = [
        "Treat rankings as decision support and keep a human review step before rejection decisions.",
        "Investigate whether under-represented groups reflect the applicant pool, sourcing channels, or historical hiring patterns.",
        "Review proxy variables such as institute, graduation year, and field of study before using them in automated scoring.",
        "Document why each scoring factor is job-relevant and remove factors that cannot be justified.",
    ]

    if group_column in df.columns:
        group_share = df[group_column].value_counts(normalize=True, dropna=False)

        if not group_share.empty and group_share.min() < 0.1:
            recommendations.append(
                f"One {group_column} group is below 10% of the dataset; quantitative fairness metrics may be unstable."
            )

    return pd.DataFrame(rows), recommendations


def _review_focus(column: str) -> str:
    if column == "Gender":
        return "Protected attribute; monitor selection and score parity."
    if column == "Age":
        return "Protected or sensitive attribute; avoid direct scoring unless legally justified."
    if column == "Institute_Name":
        return "Potential socioeconomic and geographic proxy."
    if column == "Graduation_Year":
        return "Potential age or career-break proxy."
    if column == "Field_of_Study":
        return "May encode historical access and occupational segregation."
    return "Review for job relevance and proxy bias."