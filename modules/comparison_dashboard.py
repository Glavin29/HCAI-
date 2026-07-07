import streamlit as st
import pandas as pd

from modules.model_comparison import (
    run_tfidf_baseline,
    build_model_comparison_table,
)


def render_model_comparison_dashboard(
    df: pd.DataFrame,
    roberta_metrics: dict,
):
    st.subheader("Model Comparison: TF-IDF vs RoBERTa")

    with st.spinner("Running TF-IDF baseline comparison..."):
        tfidf_metrics = run_tfidf_baseline(df)

    comparison_df = build_model_comparison_table(
        roberta_metrics=roberta_metrics,
        tfidf_metrics=tfidf_metrics,
    )

    st.dataframe(comparison_df, use_container_width=True)

    chart_df = comparison_df.set_index("model")[["accuracy", "macro_f1"]]
    st.bar_chart(chart_df)

    st.markdown("### Interpretation")
    st.write(
        "TF-IDF acts as the baseline model, while RoBERTa represents the transformer-based semantic model. "
        "A stronger RoBERTa result supports the claim that semantic representations improve resume-role matching."
    )