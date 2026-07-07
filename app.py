import pandas as pd
import streamlit as st

from modules.fairness_dashboard import render_fairness_dashboard
from modules.comparison_dashboard import render_model_comparison_dashboard
from modules.explainability import candidate_profile, candidate_strengths, score_breakdown
from modules.fairness import dataset_fairness_audit, fairness_report, rebalance_shortlist
from modules.preprocessing import load_and_preprocess
from modules.explainability_dashboard import render_explainability_dashboard
from modules.ranking import (
    match_job_description,
    rank_candidates,
    ranking_metrics,
    score_candidates,
    shortlist_top_percent,
)


DATA_PATH = "modules/resume_dataset_1200.csv"
TABLE_COLUMNS = [
    "rank",
    "Name",
    "Gender",
    "predicted_role",
    "Education_Level",
    "Experience_Years",
    "candidate_score",
]


st.set_page_config(
    page_title="FairHire AI",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fb;
    }
    .block-container {
        color: #0f172a;
    }
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container p,
    .block-container label {
        color: #0f172a;
    }
    section[data-testid="stSidebar"] {
        background: #101827;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #172033 0%, #27415f 55%, #2e6f79 100%);
        color: white !important;
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: 0;
        color: white !important;
    }
    .hero p {
        margin: .45rem 0 0;
        color: #dbeafe !important;
        font-size: 1.02rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
    }
    div[data-testid="stMetric"] * {
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
    }
    button[data-baseweb="tab"] p {
        color: #334155 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ff4b4b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_ranked_candidates() -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    processed_df = load_and_preprocess(DATA_PATH)
    scored_df, model_metrics = score_candidates(processed_df)
    return rank_candidates(scored_df), model_metrics


def render_metric_row(ranked_df: pd.DataFrame, shortlist_df: pd.DataFrame) -> None:
    metrics = ranking_metrics(ranked_df, shortlist_df)
    missing_values = int(ranked_df.isnull().sum().sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates", f"{len(ranked_df):,}")
    col2.metric("Shortlist Size", f"{metrics['shortlist_size']:,}")
    col3.metric("Top Score", f"{metrics['top_score']:.2f}")
    col4.metric("Missing Values", f"{missing_values:,}")


def render_candidate_table(df: pd.DataFrame, columns: list[str], height: int = 480) -> None:
    st.dataframe(
        df.loc[:, columns],
        use_container_width=True,
        hide_index=True,
        height=height,
    )


ranked_df, model_metrics = get_ranked_candidates()


with st.sidebar:
    st.title("FairHire AI")
    st.caption("Human-centered resume screening")

    page = st.radio(
        "Navigation",
        [
            "Candidate Ranking",
            "Fairness Analysis",
            "Fairness Mitigation",
            "Job Matching",
        ],
    )

    shortlist_percent = st.slider(
        "Shortlist percentage",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    st.divider()
    st.caption("Scores come from a trained prototype model for decision support, not automated hiring decisions.")

shortlisted = shortlist_top_percent(ranked_df, shortlist_percent / 100)

st.markdown(
    """
    <div class="hero">
        <h1>FairHire AI</h1>
        <p>Fair, explainable, and human-centered resume screening for responsible hiring decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_metric_row(ranked_df, shortlisted)

if page == "Candidate Ranking":
    st.header("Candidate Ranking")
    ranking_tab, explain_tab, model_tab = st.tabs(
        ["Ranked Candidates", "Candidate Explanation", "Model Details"]
    )

    with ranking_tab:
        st.subheader("Top Ranked Candidates")
        render_candidate_table(shortlisted, TABLE_COLUMNS)

    with explain_tab:
        candidate = st.selectbox("Choose candidate", ranked_df["Name"])
        row = ranked_df[ranked_df["Name"] == candidate].iloc[0]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Candidate Profile")
            st.json(candidate_profile(row))

        with col2:
            st.subheader("Score Breakdown")
            breakdown = score_breakdown(row)
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

        st.subheader("Strength Signals")
        st.write(", ".join(candidate_strengths(row)))

    with model_tab:
        st.subheader("Trained Model")
        st.json(model_metrics)
        st.info(
            "The model predicts resume role consistency from non-name, non-gender features. "
            "Its confidence is used as the ranking score because this dataset has no real hiring outcome label."
        )

elif page == "Fairness Analysis":
    st.header("Fairness Analysis")
    report, summary = fairness_report(ranked_df, shortlisted)
    audit_df, recommendations = dataset_fairness_audit(ranked_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Fairness Status", summary["status"])
    col2.metric("Minimum Impact Ratio", f"{summary['min_impact_ratio']:.2f}")
    col3.metric("Selection Gap", f"{summary['selection_gap_pct']:.1f}%")

    metrics_tab, audit_tab, shortlist_tab = st.tabs(
        ["Quantitative Metrics", "HCAI Audit", "Shortlist"]
    )

    with metrics_tab:
        st.subheader("Selection Rate and Score Parity by Gender")
        st.dataframe(report, use_container_width=True, hide_index=True)

        chart_data = report.set_index("Gender")[["selection_rate_pct", "impact_ratio"]]
        st.bar_chart(chart_data)

    with audit_tab:
        st.subheader("Dataset and Proxy-Risk Review")
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

        st.subheader("Governance Notes")
        for recommendation in recommendations:
            st.info(recommendation)

    with shortlist_tab:
        st.subheader(f"Top {shortlist_percent}% Shortlisted Candidates")
        st.caption(f"{len(shortlisted)} candidates selected from {len(ranked_df)} total candidates.")
        render_candidate_table(shortlisted, TABLE_COLUMNS)

elif page == "Fairness Mitigation":
    st.header("Fairness Mitigation")

    fair_shortlist = rebalance_shortlist(ranked_df, len(shortlisted))
    original_report, original_summary = fairness_report(ranked_df, shortlisted)
    adjusted_report, adjusted_summary = fairness_report(ranked_df, fair_shortlist)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Shortlist")
        st.metric("Minimum Impact Ratio", f"{original_summary['min_impact_ratio']:.2f}")
        st.dataframe(original_report, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Representation-Aware Shortlist")
        st.metric("Minimum Impact Ratio", f"{adjusted_summary['min_impact_ratio']:.2f}")
        st.dataframe(adjusted_report, use_container_width=True, hide_index=True)

    st.subheader("Adjusted Candidates")
    adjusted_columns = ["adjusted_rank"] + [column for column in TABLE_COLUMNS if column != "rank"]
    render_candidate_table(fair_shortlist, adjusted_columns)
    st.success("The shortlist was rebalanced for group representation while keeping rank order within each group.")

elif page == "Job Matching":
    st.header("Job Matching")
    st.caption("Paste a job description to compare it with candidate skills, certifications, and target roles.")

    job_description = st.text_area("Job description", height=220)
    match_button = st.button("Match Candidates", type="primary")

    if match_button:
        if not job_description.strip():
            st.warning("Please enter a job description.")
        else:
            matched_df = match_job_description(ranked_df, job_description)
            match_columns = [
                "rank",
                "Name",
                "Gender",
                "predicted_role",
                "Education_Level",
                "Experience_Years",
                "job_match_score",
            ]

            best = matched_df.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Match", best["Name"])
            col2.metric("Match Score", f"{best['job_match_score']:.1f}%")
            col3.metric("Candidates Compared", f"{len(matched_df):,}")

            st.subheader("Top Matching Candidates")
            render_candidate_table(matched_df.head(20), match_columns)
