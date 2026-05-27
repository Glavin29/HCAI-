import streamlit as st
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from transformers import BertTokenizer

from model import FairBERTModel


# ---------------- Load Resources ----------------
@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")


@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@st.cache_resource
def load_model():
    label_encoder = joblib.load("label_encoder.pkl")
    model = FairBERTModel(num_labels=len(label_encoder.classes_))
    model.load_state_dict(torch.load("fairbert_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model


# ---------------- Prediction ----------------
def predict(text, model, tokenizer, label_encoder, max_len=256):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        class_logits, gender_logits = model(input_ids, attention_mask, lambda_adv=0.5)

        class_probs = torch.nn.functional.softmax(class_logits, dim=1).cpu().numpy()[0]
        gender_probs = torch.nn.functional.softmax(gender_logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(class_probs)[::-1][:5]
    top_labels = label_encoder.inverse_transform(top_indices)
    top_probs = class_probs[top_indices]

    gender_confidence_gap = abs(gender_probs[0] - gender_probs[1])
    fairness_score = (1 - gender_confidence_gap) * 100

    return list(zip(top_labels, top_probs)), fairness_score


# ---------------- Matching Logic ----------------
def match_percentage(candidate, criteria):
    score = 0
    total = 4

    if criteria["gender"] != "Any":
        score += int(candidate["gender"] == criteria["gender"])
    else:
        score += 1

    exp_range = list(map(int, criteria["experience"].replace("years", "").strip().split("-")))
    if exp_range[0] <= candidate["experience"] <= exp_range[1]:
        score += 1

    age_range = list(map(int, criteria["age_range"].strip().split("-")))
    if age_range[0] <= candidate["age"] <= age_range[1]:
        score += 1

    overlap = len(set(candidate["skills"]).intersection(set(criteria["skills"])))
    if overlap > 0:
        score += 1

    return int((score / total) * 100)


def recommend_best_companies(candidate, companies, top_n=5):
    scores = {}

    for name, criteria in companies.items():
        scores[name] = match_percentage(candidate, criteria)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]


# ---------------- Visualization ----------------
def plot_bar_chart(data, title, xlabel="Score", ylabel="Category"):
    labels, values = zip(*data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    return fig


# ---------------- Skills and Companies ----------------
general_skills = [
    "Python", "Java", "AWS", "Machine Learning", "Cloud", "Excel", "Accounting",
    "Communication", "JavaScript", "React", "UX", "AI", "Data Analysis", "Spring",
    "MySQL", "Creative Suite", "Design", "Azure", "Testing", "Agile", "IT Support",
    "Linux", "Networking", "DevOps", "Security", "Consulting", "SAP",
    "Project Management", "Database", "SQL", "Automation", "Engineering", "IoT",
    "Payments", "APIs", "E-commerce", "CRM", "Product Development",
    "Customer Support", "SaaS"
]

companies = {
    "Google": {"gender": "Any", "age_range": "21-40", "experience": "2-5 years",
               "skills": ["Python", "Machine Learning", "Cloud"]},
    "Deloitte": {"gender": "Any", "age_range": "24-35", "experience": "1-3 years",
                 "skills": ["Excel", "Accounting", "Communication"]},
    "Amazon": {"gender": "Any", "age_range": "23-40", "experience": "1-5 years",
               "skills": ["AWS", "Java", "Microservices"]},
    "Meta": {"gender": "Any", "age_range": "22-38", "experience": "2-4 years",
             "skills": ["React", "JavaScript", "UX"]},
    "IBM": {"gender": "Any", "age_range": "25-45", "experience": "3-6 years",
            "skills": ["AI", "Python", "Data Analysis"]},
    "Infosys": {"gender": "Any", "age_range": "21-35", "experience": "1-4 years",
                "skills": ["Java", "Spring", "MySQL"]},
    "Adobe": {"gender": "Any", "age_range": "23-40", "experience": "2-5 years",
              "skills": ["Creative Suite", "Design", "UX"]},
    "Capgemini": {"gender": "Any", "age_range": "24-40", "experience": "2-5 years",
                  "skills": ["Cloud", "Azure", "Python"]},
    "TCS": {"gender": "Any", "age_range": "22-35", "experience": "1-4 years",
            "skills": ["Java", "Testing", "Agile"]},
    "Wipro": {"gender": "Any", "age_range": "21-34", "experience": "1-3 years",
              "skills": ["IT Support", "Linux", "Networking"]},
    "HCL": {"gender": "Any", "age_range": "22-36", "experience": "2-4 years",
            "skills": ["DevOps", "Python", "Security"]},
    "Accenture": {"gender": "Any", "age_range": "23-38", "experience": "1-5 years",
                  "skills": ["Consulting", "SAP", "Project Management"]},
    "Cisco": {"gender": "Any", "age_range": "24-40", "experience": "2-5 years",
              "skills": ["Networking", "Security", "Linux"]},
    "Oracle": {"gender": "Any", "age_range": "25-42", "experience": "3-6 years",
               "skills": ["Database", "SQL", "Cloud"]},
    "Siemens": {"gender": "Any", "age_range": "24-38", "experience": "2-4 years",
                "skills": ["Automation", "Engineering", "IoT"]},
    "PayPal": {"gender": "Any", "age_range": "23-40", "experience": "2-5 years",
               "skills": ["Payments", "APIs", "Java"]},
    "Flipkart": {"gender": "Any", "age_range": "22-35", "experience": "2-5 years",
                 "skills": ["E-commerce", "Data Analysis", "Python"]},
    "Zoho": {"gender": "Any", "age_range": "21-36", "experience": "1-4 years",
             "skills": ["CRM", "Java", "Product Development"]},
    "Freshworks": {"gender": "Any", "age_range": "23-38", "experience": "1-5 years",
                   "skills": ["Customer Support", "JavaScript", "SaaS"]}
}


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Fair Resume Category Recommender", layout="wide")
st.title("Fair Resume Category Recommender")

label_encoder = load_label_encoder()
tokenizer = load_tokenizer()
model = load_model()

col1, col2, col3 = st.columns([3, 3, 2])

with col1:
    st.header("Male Candidate")
    male = {
        "gender": "Male",
        "age": st.number_input("Age", min_value=18, max_value=60, value=28, key="male_age"),
        "experience": st.number_input("Experience", min_value=0, max_value=40, value=4, key="male_exp"),
        "skills": st.multiselect(
            "Skills",
            options=general_skills,
            default=["Python", "Machine Learning"],
            key="male_skills"
        ),
        "resume": st.text_area("Paste Male Candidate Resume", height=150, key="male_resume")
    }

with col2:
    st.header("Female Candidate")
    female = {
        "gender": "Female",
        "age": st.number_input("Age ", min_value=18, max_value=60, value=28, key="female_age"),
        "experience": st.number_input("Experience ", min_value=0, max_value=40, value=4, key="female_exp"),
        "skills": st.multiselect(
            "Skills ",
            options=general_skills,
            default=["Python", "Machine Learning"],
            key="female_skills"
        ),
        "resume": st.text_area("Paste Female Candidate Resume", height=150, key="female_resume")
    }

with col3:
    st.header("Company Criteria")
    selected_company = st.selectbox("Select Company", list(companies.keys()))
    criteria = companies[selected_company]

    with st.expander("View Criteria"):
        st.write(f"Gender: {criteria['gender']}")
        st.write(f"Age Range: {criteria['age_range']}")
        st.write(f"Experience: {criteria['experience']}")
        st.write(f"Skills: {', '.join(criteria['skills'])}")

    analyze_button = st.button("Analyze & Compare", type="primary", use_container_width=True)

if analyze_button:
    if male["resume"].strip() and female["resume"].strip():
        with st.spinner("Analyzing resumes..."):
            male_pred, male_fair = predict(male["resume"], model, tokenizer, label_encoder)
            female_pred, female_fair = predict(female["resume"], model, tokenizer, label_encoder)

            male_match = match_percentage(male, criteria)
            female_match = match_percentage(female, criteria)

            male_top = recommend_best_companies(male, companies)
            female_top = recommend_best_companies(female, companies)

            avg_male = np.mean([score for _, score in male_top])
            avg_female = np.mean([score for _, score in female_top])

            fairness_gap = abs(avg_male - avg_female)
            overall_fairness = max(0, 100 - fairness_gap)

        st.markdown("---")
        st.header("Analysis Results")

        st.subheader("Category Predictions")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.pyplot(plot_bar_chart(male_pred, "Male Category Probabilities"))

        with chart_col2:
            st.pyplot(plot_bar_chart(female_pred, "Female Category Probabilities"))

        st.subheader("Fairness Scores")
        fair_col1, fair_col2 = st.columns(2)

        with fair_col1:
            st.metric("Male Candidate Fairness Score", f"{male_fair:.2f}%")

        with fair_col2:
            st.metric("Female Candidate Fairness Score", f"{female_fair:.2f}%")

        st.subheader(f"Match with {selected_company}")
        match_col1, match_col2 = st.columns(2)

        with match_col1:
            st.metric("Male Match", f"{male_match}%")

        with match_col2:
            st.metric("Female Match", f"{female_match}%")

        st.subheader("Top 5 Company Matches")
        top_col1, top_col2 = st.columns(2)

        with top_col1:
            st.pyplot(plot_bar_chart(male_top, "Top Matches: Male", ylabel="Company"))

        with top_col2:
            st.pyplot(plot_bar_chart(female_top, "Top Matches: Female", ylabel="Company"))

        st.subheader("Overall Fairness Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Average Male Match", f"{avg_male:.2f}%")

        with summary_col2:
            st.metric("Average Female Match", f"{avg_female:.2f}%")

        with summary_col3:
            st.metric("Overall Fairness Score", f"{overall_fairness:.2f}%")

        st.info("Higher fairness score indicates reduced gender disparity in company match opportunities.")

    else:
        st.error("Please enter resume text for both candidates before analyzing.")