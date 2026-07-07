import streamlit as st
import pandas as pd


EXPLANATION_COLUMNS = [
    "Experience_Years",
    "education_score",
    "skills_score",
    "certification_score",
    "Graduation_Year",
]


def render_explainability_dashboard(
    scored_df: pd.DataFrame,
    score_column: str = "candidate_score",
):
    st.subheader("Explainability Dashboard")

    if score_column not in scored_df.columns:
        st.warning("Candidate score column not found.")
        return

    available_columns = [
        col for col in EXPLANATION_COLUMNS if col in scored_df.columns
    ]

    if not available_columns:
        st.warning("No explainability columns found.")
        return

    st.markdown("### Feature Correlation With Candidate Score")

    explanation_df = scored_df[available_columns + [score_column]].copy()

    for col in available_columns:
        explanation_df[col] = pd.to_numeric(explanation_df[col], errors="coerce").fillna(0)

    explanation_df[score_column] = pd.to_numeric(
        explanation_df[score_column], errors="coerce"
    ).fillna(0)

    correlations = (
        explanation_df.corr(numeric_only=True)[score_column]
        .drop(score_column)
        .sort_values(ascending=False)
    )

    st.dataframe(
        correlations.reset_index().rename(
            columns={"index": "Feature", score_column: "Correlation With Score"}
        ),
        use_container_width=True,
    )

    st.bar_chart(correlations)

    st.markdown("### Top Candidate Explanation")

    top_candidate = scored_df.sort_values(score_column, ascending=False).iloc[0]

    st.write("**Predicted Role:**", top_candidate.get("predicted_role", "Unknown"))
    st.write("**Candidate Score:**", round(float(top_candidate[score_column]), 2))

    rows = []

    for col in available_columns:
        value = pd.to_numeric(top_candidate.get(col, 0), errors="coerce")
        avg_value = pd.to_numeric(scored_df[col], errors="coerce").mean()

        rows.append(
            {
                "Feature": col,
                "Candidate Value": value,
                "Dataset Average": avg_value,
                "Difference": value - avg_value,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)