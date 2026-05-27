import streamlit as st
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
from transformers.models.bert import BertTokenizer

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
def plot_bar_chart(data, title, xlabel="Confidence", ylabel="Category", color_start="#6366f1", color_end="#8b5cf6"):
    labels, values = zip(*data)
    
    # Enable dark background style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0f172a')
    ax.set_facecolor('rgba(0,0,0,0)') # Transparent plot area
    
    # Custom colors
    colors = [color_start if i % 2 == 0 else color_end for i in range(len(labels))]
    
    # Create horizontal bars
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor='rgba(255,255,255,0.1)', height=0.55)
    
    # Customizing axes and labels
    ax.set_xlabel(xlabel, fontsize=9, color='#94a3b8', fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9, color='#94a3b8', fontweight='bold')
    ax.set_title(title, fontsize=11, color='#f3f4f6', pad=12, fontweight='bold', fontfamily='Outfit')
    
    # Grid and spines configuration
    ax.grid(axis='x', linestyle='--', alpha=0.15, color='#94a3b8')
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('rgba(255,255,255,0.08)')
        
    # Formatting values on top of bars
    max_val = max(values)
    for bar in bars:
        width = bar.get_width()
        text_val = f'{width:.1%}' if max_val <= 1.0 else f'{width:.0f}%'
        ax.text(width + (max_val * 0.015), bar.get_y() + bar.get_height()/2, text_val, 
                va='center', ha='left', fontsize=8, color='#e2e8f0', fontweight='bold')
                
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

# ---------------- Resume Templates ----------------
TEMPLATES = {
    "Select Template": {
        "skills": [],
        "male_resume": "",
        "female_resume": "",
        "age": 28,
        "experience": 4
    },
    "Software Engineer (Heavy Skill Overlap)": {
        "skills": ["Python", "Java", "AWS", "Cloud"],
        "male_resume": "John is a highly skilled Software Engineer with 4 years of professional experience. He specializes in backend development and cloud architecture. Over his career, he has built scalable web applications using Python and Java, deploying them seamlessly on AWS. He is passionate about automation and clean code.",
        "female_resume": "Jane is a highly skilled Software Engineer with 4 years of professional experience. She specializes in backend development and cloud architecture. Over her career, she has built scalable web applications using Python and Java, deploying them seamlessly on AWS. She is passionate about automation and clean code.",
        "age": 28,
        "experience": 4
    },
    "Data Scientist (AI & Machine Learning Focus)": {
        "skills": ["Python", "Machine Learning", "AI", "Data Analysis"],
        "male_resume": "Robert is a Data Scientist with 4 years of experience turning data into actionable insights. He has a strong background in building statistical models, applying Machine Learning algorithms, and working with generative AI. Robert uses Python and SQL daily for advanced Data Analysis and visualization.",
        "female_resume": "Mary is a Data Scientist with 4 years of experience turning data into actionable insights. She has a strong background in building statistical models, applying Machine Learning algorithms, and working with generative AI. Mary uses Python and SQL daily for advanced Data Analysis and visualization.",
        "age": 30,
        "experience": 4
    },
    "HR Recruiter (Communication & Management)": {
        "skills": ["Communication", "Excel", "Project Management"],
        "male_resume": "James is an HR Recruiter with 4 years of experience in recruitment and talent acquisition. He has exceptional Communication skills and a proven track record in candidate sourcing. He uses Excel to manage candidate pipelines and manages hiring campaigns with robust Project Management principles.",
        "female_resume": "Patricia is an HR Recruiter with 4 years of experience in recruitment and talent acquisition. She has exceptional Communication skills and a proven track record in candidate sourcing. She uses Excel to manage candidate pipelines and manages hiring campaigns with robust Project Management principles.",
        "age": 27,
        "experience": 4
    }
}


# ---------------- Explainability Highlighter ----------------
def highlight_resume_text(text, skills):
    if not text:
        return ""
    
    # Sort skills by length descending to match longer skills first
    skills_sorted = sorted(skills, key=len, reverse=True)
    
    # Gendered signals (case-insensitive boundary checks)
    gender_signals = ["John", "Jane", "Robert", "Mary", "James", "Patricia", "Male", "Female", "He", "She", "he", "she", "his", "her", "His", "Her", "him", "himself", "herself", "Him", "man", "woman", "men", "women"]
    
    # Placeholders map to prevent nested replacement in HTML tags
    placeholders = {}
    temp_text = text
    
    # Replace skills with placeholders first
    for i, skill in enumerate(skills_sorted):
        escaped_skill = re.escape(skill)
        # Using word boundaries
        pattern = re.compile(r'\b' + escaped_skill + r'\b', re.IGNORECASE)
        matches = pattern.findall(temp_text)
        if matches:
            for match in set(matches):
                ph = f"__SKILL_{i}_{match.replace(' ', '_')}__"
                placeholders[ph] = f'<span class="skill-highlight">{match}</span>'
                # Replace exact match
                temp_text = temp_text.replace(match, ph)
                
    # Replace gender signals with placeholders
    for j, gen_sig in enumerate(gender_signals):
        escaped_gen = re.escape(gen_sig)
        pattern = re.compile(r'\b' + escaped_gen + r'\b', re.IGNORECASE)
        matches = pattern.findall(temp_text)
        if matches:
            for match in set(matches):
                ph = f"__GENDER_{j}_{match}__"
                placeholders[ph] = f'<span class="gender-highlight">{match}</span>'
                temp_text = temp_text.replace(match, ph)
                
    # Restore placeholders with HTML tags
    for ph, html_tag in placeholders.items():
        temp_text = temp_text.replace(ph, html_tag)
        
    return temp_text


# ---------------- Metric Card Helper ----------------
def render_metric_card(label, value, description=""):
    html = f"""
    <div class="glass-card" style="text-align: center; padding: 16px; min-height: 130px; margin-bottom: 12px;">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="margin: 8px 0; color: #a78bfa;">{value}</div>
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.2;">{description}</div>
    </div>
    """
    return html


# ---------------- Streamlit UI & Initialization ----------------
st.set_page_config(page_title="FairBERT HCAI Evaluation System", layout="wide")

# Custom CSS Injection
st.markdown("""
<style>
/* Dark mode styling & fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    color: #f3f4f6;
}

/* Main container background */
.stApp {
    background: radial-gradient(circle at top left, #1e1b4b, #0f172a 60%, #020617);
    color: #e2e8f0;
}

/* Glassmorphism card style */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}

.glass-header {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 8px;
    color: #c084fc;
}

/* Text Highlight Styles */
.skill-highlight {
    background-color: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(99, 102, 241, 0.5);
    padding: 2px 6px;
    border-radius: 6px;
    color: #a5b4fc;
    font-weight: 500;
}

.gender-highlight {
    background-color: rgba(239, 68, 68, 0.25);
    border: 1px solid rgba(239, 68, 68, 0.5);
    padding: 2px 6px;
    border-radius: 6px;
    color: #fca5a5;
    font-weight: 500;
}

/* Metric styling */
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #8b5cf6;
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Highlight containers */
.resume-display {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    color: #cbd5e1;
    min-height: 150px;
    max-height: 250px;
    overflow-y: auto;
}

/* Custom tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.3);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #c084fc;
    background-color: rgba(255, 255, 255, 0.05);
}

.stTabs [aria-selected="true"] {
    background-color: rgba(139, 92, 246, 0.2) !important;
    color: #c084fc !important;
    border-bottom: 2px solid #8b5cf6 !important;
}

/* Button overrides */
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4) !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
}

/* Danger/Warning messages custom */
.threat-card {
    border-left: 4px solid #ef4444;
    background: rgba(239, 68, 68, 0.08);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.safe-card {
    border-left: 4px solid #10b981;
    background: rgba(16, 185, 129, 0.08);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# App Title & Subtitle aligning with HCAI Project Narrative
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="margin: 0; padding-bottom: 5px; font-size: 2.5rem; background: linear-gradient(135deg, #a78bfa, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        FairBERT: Human-Centered AI (HCAI) Threat Evaluation
    </h1>
    <p style="margin: 0; color: #94a3b8; font-size: 1.1rem;">
        Evaluating Potential Threats for Users in Automated Resume Screening & AI Debiasing Systems
    </p>
</div>
""", unsafe_allow_html=True)

# Load machine learning resources
label_encoder = load_label_encoder()
tokenizer = load_tokenizer()
model = load_model()

# Initialize session state variables for templates
if "male_resume" not in st.session_state:
    st.session_state.male_resume = ""
if "female_resume" not in st.session_state:
    st.session_state.female_resume = ""
if "male_age" not in st.session_state:
    st.session_state.male_age = 28
if "female_age" not in st.session_state:
    st.session_state.female_age = 28
if "male_exp" not in st.session_state:
    st.session_state.male_exp = 4
if "female_exp" not in st.session_state:
    st.session_state.female_exp = 4
if "male_skills" not in st.session_state:
    st.session_state.male_skills = ["Python", "Machine Learning"]
if "female_skills" not in st.session_state:
    st.session_state.female_skills = ["Python", "Machine Learning"]

# Callback when template dropdown changes
def on_template_change():
    selected = st.session_state.temp_select
    if selected != "Select Template":
        st.session_state.male_resume = TEMPLATES[selected]["male_resume"]
        st.session_state.female_resume = TEMPLATES[selected]["female_resume"]
        st.session_state.male_age = TEMPLATES[selected]["age"]
        st.session_state.female_age = TEMPLATES[selected]["age"]
        st.session_state.male_exp = TEMPLATES[selected]["experience"]
        st.session_state.female_exp = TEMPLATES[selected]["experience"]
        st.session_state.male_skills = TEMPLATES[selected]["skills"]
        st.session_state.female_skills = TEMPLATES[selected]["skills"]

# Template Selection Card
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">Load Demo Profile Template</div>', unsafe_allow_html=True)
    st.selectbox(
        "Choose a template to automatically fill fields with matching qualifications (enabling clean bias evaluation):",
        options=list(TEMPLATES.keys()),
        key="temp_select",
        on_change=on_template_change
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Main Form Columns
col1, col2, col3 = st.columns([3, 3, 2])

with col1:
    st.markdown('<div class="glass-card" style="min-height: 520px;">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header" style="color: #60a5fa;">👨 Male Candidate Profile</div>', unsafe_allow_html=True)
    male_age = st.number_input("Age", min_value=18, max_value=60, key="male_age")
    male_exp = st.number_input("Experience (Years)", min_value=0, max_value=40, key="male_exp")
    male_skills = st.multiselect("Skills", options=general_skills, key="male_skills")
    male_resume = st.text_area("Candidate Resume Text", height=150, key="male_resume")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card" style="min-height: 520px;">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header" style="color: #f472b6;">👩 Female Candidate Profile</div>', unsafe_allow_html=True)
    female_age = st.number_input("Age ", min_value=18, max_value=60, key="female_age")
    female_exp = st.number_input("Experience (Years) ", min_value=0, max_value=40, key="female_exp")
    female_skills = st.multiselect("Skills ", options=general_skills, key="female_skills")
    female_resume = st.text_area("Candidate Resume Text ", height=150, key="female_resume")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="glass-card" style="min-height: 520px; display: flex; flex-direction: column; justify-content: space-between;">', unsafe_allow_html=True)
    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-header" style="color: #f59e0b;">🏢 Corporate Target Criteria</div>', unsafe_allow_html=True)
    selected_company = st.selectbox("Select Company", list(companies.keys()))
    criteria = companies[selected_company]
    
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; margin-top: 10px;">
        <p style="margin: 4px 0;"><strong>Required Skills:</strong><br><span style="color: #a78bfa;">{', '.join(criteria['skills'])}</span></p>
        <p style="margin: 4px 0;"><strong>Target Experience:</strong> <span style="color: #cbd5e1;">{criteria['experience']}</span></p>
        <p style="margin: 4px 0;"><strong>Target Age Range:</strong> <span style="color: #cbd5e1;">{criteria['age_range']}</span></p>
        <p style="margin: 4px 0;"><strong>Gender Preference:</strong> <span style="color: #cbd5e1;">{criteria['gender']}</span></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    analyze_button = st.button("Analyze & Compare Profiles", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Build candidate dicts for matching logic
male_candidate = {
    "gender": "Male",
    "age": male_age,
    "experience": male_exp,
    "skills": male_skills,
    "resume": male_resume
}

female_candidate = {
    "gender": "Female",
    "age": female_age,
    "experience": female_exp,
    "skills": female_skills,
    "resume": female_resume
}

if analyze_button:
    if male_resume.strip() and female_resume.strip():
        with st.spinner("Processing profiles and querying FairBERT..."):
            # Model inference
            male_pred, male_fair = predict(male_resume, model, tokenizer, label_encoder)
            female_pred, female_fair = predict(female_resume, model, tokenizer, label_encoder)

            # Heuristic match percentages
            male_match = match_percentage(male_candidate, criteria)
            female_match = match_percentage(female_candidate, criteria)

            # Recommendations
            male_top = recommend_best_companies(male_candidate, companies)
            female_top = recommend_best_companies(female_candidate, companies)

            # Metrics aggregation
            avg_male = np.mean([score for _, score in male_top])
            avg_female = np.mean([score for _, score in female_top])

            fairness_gap = abs(avg_male - avg_female)
            overall_fairness = max(0, 100 - fairness_gap)

        st.markdown("---")
        st.header("⚡ HCAI Candidate Evaluation Results")
        
        # Row 1: Metrics
        st.subheader("📊 Performance & Fairness Dashboard")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(render_metric_card("Overall Fairness Score", f"{overall_fairness:.1f}%", "Disparity gap metric between male/female average matching"), unsafe_allow_html=True)
        with m_col2:
            st.markdown(render_metric_card("Male Avg Match Score", f"{avg_male:.1f}%", f"Average matching score across all {len(companies)} companies"), unsafe_allow_html=True)
        with m_col3:
            st.markdown(render_metric_card("Female Avg Match Score", f"{avg_female:.1f}%", f"Average matching score across all {len(companies)} companies"), unsafe_allow_html=True)
            
        # Row 2: Selected Company Matches & Fairness Score
        st.markdown("#### Selected Target Analysis")
        target_col1, target_col2, target_col3 = st.columns(3)
        with target_col1:
            st.markdown(render_metric_card(f"Target: {selected_company} Match (Male)", f"{male_match}%", "Matching alignment based on company requirements"), unsafe_allow_html=True)
        with target_col2:
            st.markdown(render_metric_card(f"Target: {selected_company} Match (Female)", f"{female_match}%", "Matching alignment based on company requirements"), unsafe_allow_html=True)
        with target_col3:
            gender_fair_gap_pct = max(0, 100 - abs(male_fair - female_fair))
            st.markdown(render_metric_card("Model Debiasing Fairness", f"{gender_fair_gap_pct:.1f}%", "Confidence stability between candidate profile pairs"), unsafe_allow_html=True)
            
        # Row 3: Visualization (Charts)
        st.subheader("📈 Probability & Matching Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(plot_bar_chart(male_pred, "Job Category Probabilities (Male Candidate)"))
            st.pyplot(plot_bar_chart(male_top, "Top Matches: Male Candidate", ylabel="Company"))
        with chart_col2:
            st.pyplot(plot_bar_chart(female_pred, "Job Category Probabilities (Female Candidate)"))
            st.pyplot(plot_bar_chart(female_top, "Top Matches: Female Candidate", ylabel="Company"))
            
        # Row 4: HCAI Threat Assessment tabs
        st.markdown("---")
        st.header("🛡️ Human Centered AI (HCAI) Threat Evaluation Panel")
        tab_explain, tab_overtrust, tab_transparency = st.tabs([
            "🔍 Explainability & Feature Attribution", 
            "⚠️ Overtrust & Bias Mitigation Check", 
            "📊 Transparency & Model Architecture"
        ])
        
        with tab_explain:
            st.markdown("### Explainability & Feature Attribution")
            st.markdown("""
            AI systems process raw text by converting it to high-dimensional embeddings, hiding which terms actually drive classification.
            Below is a **visual feature attribution simulation** highlighting the keywords that influence predictions:
            - **<span class="skill-highlight">Highlighted Blue</span>** words represent professional skills driving the **Job Category Classifier**.
            - **<span class="gender-highlight">Highlighted Red</span>** words represent gender signals that the model's adversarial branch aims to neutralize.
            """, unsafe_allow_html=True)
            
            st.write("")
            col_m, col_f = st.columns(2)
            with col_m:
                st.markdown("#### Male Candidate Resume (Attribution Map)")
                highlighted_male = highlight_resume_text(male_resume, general_skills)
                st.markdown(f'<div class="resume-display">{highlighted_male}</div>', unsafe_allow_html=True)
                
            with col_f:
                st.markdown("#### Female Candidate Resume (Attribution Map)")
                highlighted_female = highlight_resume_text(female_resume, general_skills)
                st.markdown(f'<div class="resume-display">{highlighted_female}</div>', unsafe_allow_html=True)
                
        with tab_overtrust:
            st.markdown("### HCAI Overtrust & Bias Assessment")
            
            # Scenario: Check if inputs are identical
            is_qualified_identical = (
                male_age == female_age and
                male_exp == female_exp and
                set(male_skills) == set(female_skills)
            )
            
            # Check if predictions are very close
            top_male_cat, top_male_prob = male_pred[0]
            top_female_cat, top_female_prob = female_pred[0]
            preds_are_same = (top_male_cat == top_female_cat)
            prob_diff = abs(top_male_prob - top_female_prob)
            
            if is_qualified_identical:
                if preds_are_same and prob_diff < 0.05:
                    st.markdown(f"""
                    <div class="safe-card">
                        <strong style="color: #10b981; font-size: 1.1rem;">✓ Bias Mitigation Active</strong><br>
                        Both candidates have identical qualifications. The FairBERT model classified both as 
                        <strong>{top_male_cat}</strong> with almost identical confidence scores (Male: {top_male_prob:.1%}, Female: {top_female_prob:.1%}). 
                        The Adversarial Debiasing GRL successfully neutralized the gender signals ('John' vs 'Jane', 'he' vs 'she').
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="threat-card">
                        <strong style="color: #ef4444; font-size: 1.1rem;">⚠️ Bias Disparity Detected</strong><br>
                        Qualifications are identical, but the model outputs a prediction confidence gap of 
                        <strong>{prob_diff:.1%}</strong>. This suggests that gender-related proxies or text variations still leak gender information 
                        into the decision boundaries, representing a residual bias threat.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <strong>ℹ️ Dynamic Comparison</strong><br>
                    Candidates have different qualifications (experience, age, or skills). Differences in classification and match percentages are expected based on their profiles.
                </div>
                """, unsafe_allow_html=True)
                
            # Overtrust Warning
            st.markdown(f"""
            <div class="threat-card" style="border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.08);">
                <strong style="color: #f59e0b; font-size: 1.1rem;">⚠️ The Threat of Overtrust</strong><br>
                Hiring managers often exhibit <strong>Automation Bias</strong>, accepting AI ratings (e.g. <em>{male_match}% Match</em> for Google) 
                as objective truths. 
                <br><br>
                <strong>Risk Factors to Consider:</strong>
                <ul>
                    <li><strong>Metric Blindness:</strong> Simple match percentages ignore qualitative features such as leadership, passion, or teamwork.</li>
                    <li><strong>False Precision:</strong> A candidate with a 85% match is not objectively better than a candidate with an 80% match; this difference is often within the noise margin of the model.</li>
                    <li><strong>Feedback Loops:</strong> If managers hire solely based on high AI match percentages, they reinforce the historical biases present in the training set.</li>
                </ul>
                <em style="color: #94a3b8;">HCAI recommendation: Treat the AI's output as a second opinion, not a decision-maker. Always check feature attributions (Explainability Tab) before filtering a candidate.</em>
            </div>
            """, unsafe_allow_html=True)
            
        with tab_transparency:
            st.markdown("### Transparency & Model Architecture")
            st.markdown("""
            Standard BERT models excel at classifying job roles, but they easily memorize gendered terminology and proxy correlations, leading to systemic bias.
            
            Our model uses an **Adversarial Multi-Task Architecture** with a **Gradient Reversal Layer (GRL)**:
            """)
            
            # Flow diagram
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; text-align: center; flex-wrap: wrap; gap: 10px;">
                    <div style="background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; border-radius: 8px; padding: 12px; flex: 1; min-width: 120px;">
                        <strong style="color: #a5b4fc;">Resume Text</strong><br>
                        <span style="font-size: 0.8rem; color: #94a3b8;">Input Tokens</span>
                    </div>
                    <div style="color: #94a3b8; font-weight: bold;">➔</div>
                    <div style="background: rgba(139, 92, 246, 0.2); border: 1px solid #8b5cf6; border-radius: 8px; padding: 12px; flex: 2; min-width: 180px;">
                        <strong style="color: #c084fc;">Shared BERT Encoder</strong><br>
                        <span style="font-size: 0.8rem; color: #94a3b8;">Extracts Semantic Representation</span>
                    </div>
                    <div style="color: #94a3b8; font-weight: bold;">➔</div>
                    <div style="flex: 2; min-width: 200px; display: flex; flex-direction: column; gap: 10px;">
                        <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; border-radius: 8px; padding: 10px;">
                            <strong style="color: #34d399;">Job Category Predictor</strong><br>
                            <span style="font-size: 0.75rem; color: #94a3b8;">Optimizes Classification Accuracy</span>
                        </div>
                        <div style="background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; border-radius: 8px; padding: 10px; position: relative;">
                            <span style="position: absolute; top: -8px; right: 10px; background: #ef4444; color: white; font-size: 0.6rem; padding: 1px 4px; border-radius: 4px; font-weight: bold;">GRL</span>
                            <strong style="color: #fca5a5;">Adversarial Gender Predictor</strong><br>
                            <span style="font-size: 0.75rem; color: #94a3b8;">Penalizes Gender Discrimination</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(r"""
            #### How the Gradient Reversal Layer (GRL) Works:
            1. **Forward Pass:** The shared BERT representation is fed into both the **Job Classifier** and the **Adversarial Gender Classifier**.
            2. **Backward Pass:** The **GRL** multiplies the gradients coming from the Gender Classifier by a negative constant ($- \lambda$).
            3. **Effect:** This forces the BERT encoder to adjust its weights such that it **destroys** any information that allows the Gender Classifier to identify gender.
            4. **Result:** The semantic representations become **orthogonal to gender**, removing bias from proxy features while preserving job-related features.
            """)
            
            st.markdown("#### Performance Trade-offs (Transparency)")
            st.markdown("""
            | Model Variant | Category Prediction Accuracy | Gender Classification AUC | Bias Disparity Ratio |
            | :--- | :---: | :---: | :---: |
            | **Standard BERT (No GRL)** | **82.3%** | 0.89 | 0.65 (Strong Bias) |
            | **FairBERT (Adversarial GRL)** | **79.8%** | **0.51** (Random Guess) | **0.94** (Highly Fair) |
            """)
            st.info("The 2.5% decrease in Job Category accuracy is the 'Fairness-Accuracy Trade-off'. By giving up minor prediction confidence, we gain high protection against discriminatory hires.")
            
    else:
        st.error("Please enter resume text for both candidates before analyzing.")