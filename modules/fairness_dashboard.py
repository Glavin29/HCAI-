import streamlit as st
import pandas as pd

from modules.fairness import (
    fairness_report,
    dataset_fairness_audit,
)


def render_fairness_dashboard(
    scored_df: pd.DataFrame,
    top_n: int = 10,
    group_column: str = "Gender",
    score_column: str = "candidate_score",
):
    st.subheader("Gender Fairness Dashboard")

    if group_column not in scored_df.columns:
        st.warning("Gender column not found. Fairness dashboard cannot be displayed.")
        return

    if score_column not in scored_df.columns:
        st.warning("Candidate score column not found. Fairness dashboard cannot be displayed.")
        return

    selected_df = scored_df.sort_values(score_column, ascending=False).head(top_n)

    report, summary = fairness_report(
        population_df=scored_df,
        selected_df=selected_df,
        group_column=group_column,
        score_column=score_column,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Disparate Impact Ratio", round(summary["min_impact_ratio"], 3))
    col2.metric("Selection Gap (%)", round(summary["selection_gap_pct"], 2))
    col3.metric("Fairness Status", summary["status"])

    st.markdown("### Selection and Score Parity by Gender")
    st.dataframe(report, use_container_width=True)

    st.markdown("### Selection Rate by Gender")
    st.bar_chart(report.set_index(group_column)["selection_rate_pct"])

    st.markdown("### Average Candidate Score by Gender")
    st.bar_chart(report.set_index(group_column)["mean"])

    st.markdown("### Dataset Fairness Audit")
    audit_df, recommendations = dataset_fairness_audit(scored_df)

    st.dataframe(audit_df, use_container_width=True)

    st.markdown("### Fairness Recommendations")
    for recommendation in recommendations:
        st.write(f"- {recommendation}")