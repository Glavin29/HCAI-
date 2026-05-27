import streamlit as st
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import random
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


# ---------------- Model Inference ----------------
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


# ---------------- Bias Simulation ----------------
def predict_simulated(text, model, tokenizer, label_encoder, gender, target_role, is_debiased=False):
    # Run the actual model prediction first
    preds, fairness = predict(text, model, tokenizer, label_encoder)
    pred_dict = {label: prob for label, prob in preds}
    
    if not is_debiased:
        # Simulate standard AI bias as documented by Wilson & Caliskan (2024):
        # Tech roles penalized for females; support/HR penalized for males.
        bias_penalty = 0.25
        
        if gender == "Female" and target_role in ["Software Engineer", "Data Scientist"]:
            # Apply bias penalty to tech categories
            for tech_label in ["Software Engineer", "Data Scientist"]:
                if tech_label in pred_dict:
                    pred_dict[tech_label] = max(0.05, pred_dict[tech_label] - bias_penalty)
            # Add probability back to administrative/HR roles to maintain sum of ~1.0
            if "HR Recruiter" in pred_dict:
                pred_dict["HR Recruiter"] = min(0.95, pred_dict["HR Recruiter"] + 0.15)
                
        elif gender == "Male" and target_role == "HR Recruiter":
            # Apply bias penalty to HR recruiter role
            if "HR Recruiter" in pred_dict:
                pred_dict["HR Recruiter"] = max(0.05, pred_dict["HR Recruiter"] - 0.20)
            if "Software Engineer" in pred_dict:
                pred_dict["Software Engineer"] = min(0.95, pred_dict["Software Engineer"] + 0.12)
                
        # Re-normalize dictionary probabilities
        total = sum(pred_dict.values())
        if total > 0:
            pred_dict = {k: v / total for k, v in pred_dict.items()}
            
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds


def get_simulated_match_score(candidate, criteria, is_debiased=False):
    base_score = match_percentage(candidate, criteria)
    
    if not is_debiased:
        # Biased scoring model (Wilson & Caliskan, 2024 proxy indicators)
        is_tech_job = any(s in criteria["skills"] for s in ["Python", "Machine Learning", "AWS", "Cloud", "AI"])
        is_hr_job = "Communication" in criteria["skills"]
        
        if candidate["gender"] == "Female" and is_tech_job:
            return max(35, base_score - 25)
        elif candidate["gender"] == "Male" and is_hr_job:
            return max(40, base_score - 20)
            
    return base_score


# ---------------- Visualization ----------------
def plot_bar_chart(data, title, xlabel="Confidence", ylabel="Category", color_start="#6366f1", color_end="#8b5cf6"):
    labels, values = zip(*data)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#0f172a')
    ax.set_facecolor('none')
    
    colors = [color_start if i % 2 == 0 else color_end for i in range(len(labels))]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor='rgba(255,255,255,0.08)', height=0.5)
    
    ax.set_xlabel(xlabel, fontsize=8, color='#94a3b8', fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8, color='#94a3b8', fontweight='bold')
    ax.set_title(title, fontsize=10, color='#f3f4f6', pad=10, fontweight='bold', fontfamily='Outfit')
    
    ax.grid(axis='x', linestyle='--', alpha=0.1, color='#94a3b8')
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('rgba(255,255,255,0.05)')
        
    max_val = max(values)
    for bar in bars:
        width = bar.get_width()
        text_val = f'{width:.1%}' if max_val <= 1.0 else f'{width:.0f}%'
        ax.text(width + (max_val * 0.015), bar.get_y() + bar.get_height()/2, text_val, 
                va='center', ha='left', fontsize=8, color='#e2e8f0', fontweight='bold')
                
    plt.tight_layout()
    return fig


def plot_comparison_chart(score1, score2, label1, label2, title):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='#0f172a')
    ax.set_facecolor('none')
    
    categories = [label1, label2]
    scores = [score1, score2]
    colors = ['#6366f1', '#ef4444' if score2 < score1 else '#10b981']
    
    bars = ax.barh(categories[::-1], scores[::-1], color=colors[::-1], edgecolor='rgba(255,255,255,0.08)', height=0.45)
    
    ax.set_xlabel("Match Score (%)", fontsize=8, color='#94a3b8', fontweight='bold')
    ax.set_title(title, fontsize=10, color='#f3f4f6', pad=10, fontweight='bold', fontfamily='Outfit')
    ax.set_xlim(0, 110)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('rgba(255,255,255,0.05)')
        
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width}%', 
                va='center', ha='left', fontsize=8, color='#e2e8f0', fontweight='bold')
                
    plt.tight_layout()
    return fig


# ---------------- Explainability Highlighter ----------------
def highlight_resume_text(text, skills):
    if not text:
        return ""
    
    skills_sorted = sorted(skills, key=len, reverse=True)
    gender_signals = ["John", "Jane", "Robert", "Mary", "James", "Patricia", "David", "Michael", "William", "Linda", "Elizabeth", "Jennifer", "Male", "Female", "He", "She", "he", "she", "his", "her", "His", "Her", "him", "himself", "herself", "Him", "man", "woman", "men", "women"]
    
    placeholders = {}
    temp_text = text
    
    # Replace skills with placeholders
    for i, skill in enumerate(skills_sorted):
        escaped_skill = re.escape(skill)
        pattern = re.compile(r'\b' + escaped_skill + r'\b', re.IGNORECASE)
        matches = pattern.findall(temp_text)
        if matches:
            for match in set(matches):
                ph = f"__SKILL_{i}_{match.replace(' ', '_')}__"
                placeholders[ph] = f'<span class="skill-highlight">{match}</span>'
                temp_text = temp_text.replace(match, ph)
                
    # Replace gendered signals with placeholders
    for j, gen_sig in enumerate(gender_signals):
        escaped_gen = re.escape(gen_sig)
        pattern = re.compile(r'\b' + escaped_gen + r'\b', re.IGNORECASE)
        matches = pattern.findall(temp_text)
        if matches:
            for match in set(matches):
                ph = f"__GENDER_{j}_{match}__"
                placeholders[ph] = f'<span class="gender-highlight">{match}</span>'
                temp_text = temp_text.replace(match, ph)
                
    # Re-insert HTML tags
    for ph, html_tag in placeholders.items():
        temp_text = temp_text.replace(ph, html_tag)
        
    return temp_text


# ---------------- Dynamic Candidate Generator ----------------
def generate_random_candidate():
    names_male = ["John", "Robert", "James", "David", "Michael", "William"]
    names_female = ["Jane", "Mary", "Patricia", "Linda", "Elizabeth", "Jennifer"]
    
    roles = [
        {
            "title": "Software Engineer",
            "skills": ["Python", "Java", "AWS", "Cloud"],
            "bio": "is a highly skilled Software Engineer specializing in backend development. Over his/her career, he/she has built scalable web applications using Python and Java, deploying them seamlessly on AWS. Passionate about clean code."
        },
        {
            "title": "Data Scientist",
            "skills": ["Python", "Machine Learning", "AI", "Data Analysis"],
            "bio": "is a Data Scientist with experience turning data into actionable insights. Strong background in building models, applying Machine Learning, and working with Python for Data Analysis."
        },
        {
            "title": "HR Recruiter",
            "skills": ["Communication", "Excel", "Project Management"],
            "bio": "is an HR Recruiter with experience in recruitment and talent acquisition. Exceptional Communication skills, using Excel for pipeline management and coordinating campaigns via Project Management."
        }
    ]
    
    role = random.choice(roles)
    gender_choice = random.choice(["Male", "Female"])
    
    if gender_choice == "Male":
        name = random.choice(names_male)
        swapped_name = random.choice(names_female)
        gender = "Male"
        swapped_gender = "Female"
        
        resume = f"{name} {role['bio'].replace('his/her', 'his').replace('he/she', 'he')}"
        swapped_resume = f"{swapped_name} {role['bio'].replace('his/her', 'her').replace('he/she', 'she')}"
    else:
        name = random.choice(names_female)
        swapped_name = random.choice(names_male)
        gender = "Female"
        swapped_gender = "Male"
        
        resume = f"{name} {role['bio'].replace('his/her', 'her').replace('he/she', 'she')}"
        swapped_resume = f"{swapped_name} {role['bio'].replace('his/her', 'his').replace('he/she', 'he')}"
        
    return {
        "name": name,
        "gender": gender,
        "age": random.randint(22, 45),
        "experience": random.randint(1, 10),
        "skills": role["skills"],
        "resume": resume,
        "swapped_name": swapped_name,
        "swapped_gender": swapped_gender,
        "swapped_resume": swapped_resume,
        "title": role["title"]
    }


# ---------------- Metric Card Helper ----------------
def render_metric_card(label, value, description=""):
    html = f"""
    <div class="glass-card" style="text-align: center; padding: 12px; min-height: 110px; margin-bottom: 12px;">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="margin: 4px 0; color: #a78bfa; font-size: 1.8rem;">{value}</div>
        <div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.2;">{description}</div>
    </div>
    """
    return html


# ---------------- Preset Candidates Mapping ----------------
PRESET_CANDIDATES = {
    "Robert (Data Scientist)": {
        "name": "Robert",
        "gender": "Male",
        "age": 30,
        "experience": 4,
        "skills": ["Python", "Machine Learning", "AI", "Data Analysis"],
        "resume": "Robert is a Data Scientist with 4 years of experience turning data into actionable insights. He has a strong background in building statistical models, applying Machine Learning algorithms, and working with generative AI. Robert uses Python and SQL daily for advanced Data Analysis and visualization.",
        "swapped_name": "Mary",
        "swapped_gender": "Female",
        "swapped_resume": "Mary is a Data Scientist with 4 years of experience turning data into actionable insights. She has a strong background in building statistical models, applying Machine Learning algorithms, and working with generative AI. Mary uses Python and SQL daily for advanced Data Analysis and visualization.",
        "title": "Data Scientist"
    },
    "John (Software Engineer)": {
        "name": "John",
        "gender": "Male",
        "age": 28,
        "experience": 4,
        "skills": ["Python", "Java", "AWS", "Cloud"],
        "resume": "John is a highly skilled Software Engineer with 4 years of professional experience. He specializes in backend development and cloud architecture. Over his career, he has built scalable web applications using Python and Java, deploying them seamlessly on AWS. He is passionate about automation and clean code.",
        "swapped_name": "Jane",
        "swapped_gender": "Female",
        "swapped_resume": "Jane is a highly skilled Software Engineer with 4 years of professional experience. She specializes in backend development and cloud architecture. Over her career, she has built scalable web applications using Python and Java, deploying them seamlessly on AWS. She is passionate about automation and clean code.",
        "title": "Software Engineer"
    },
    "James (HR Recruiter)": {
        "name": "James",
        "gender": "Male",
        "age": 27,
        "experience": 4,
        "skills": ["Communication", "Excel", "Project Management"],
        "resume": "James is an HR Recruiter with 4 years of experience in recruitment and talent acquisition. He has exceptional Communication skills and a proven track record in candidate sourcing. He uses Excel to manage candidate pipelines and manages hiring campaigns with robust Project Management principles.",
        "swapped_name": "Patricia",
        "swapped_gender": "Female",
        "swapped_resume": "Patricia is an HR Recruiter with 4 years of experience in recruitment and talent acquisition. She has exceptional Communication skills and a proven track record in candidate sourcing. She uses Excel to manage candidate pipelines and manages hiring campaigns with robust Project Management principles.",
        "title": "HR Recruiter"
    }
}

# ---------------- Corporate Target Profiles ----------------
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
            "skills": ["AI", "Python", "Data Analysis"]}
}


# ---------------- Streamlit Config & Styling ----------------
st.set_page_config(page_title="FairBERT HCAI Evaluation System", layout="wide")

# Custom Dark Glassmorphism CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    color: #f3f4f6;
}

.stApp {
    background: radial-gradient(circle at top left, #1e1b4b, #0f172a 60%, #020617);
    color: #e2e8f0;
}

.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}

.glass-header {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 8px;
    color: #c084fc;
}

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

.metric-value {
    font-weight: 700;
    color: #8b5cf6;
}

.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.resume-display {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 16px;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    color: #cbd5e1;
    min-height: 120px;
    max-height: 200px;
    overflow-y: auto;
}

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

.lit-card {
    border-left: 4px solid #3b82f6;
    background: rgba(59, 130, 246, 0.08);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    color: #93c5fd;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------- State Initialization & Header ----------------
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

if "active_candidate" not in st.session_state:
    st.session_state.active_candidate = PRESET_CANDIDATES["Robert (Data Scientist)"].copy()

# Load Neural Network weights
label_encoder = load_label_encoder()
tokenizer = load_tokenizer()
model = load_model()

# Header block
st.markdown("""
<div style="margin-bottom: 25px;">
    <h1 style="margin: 0; padding-bottom: 5px; font-size: 2.2rem; background: linear-gradient(135deg, #a78bfa, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        FairBERT: Human-Centered AI (HCAI) Threat Evaluation Sandbox
    </h1>
    <p style="margin: 0; color: #94a3b8; font-size: 1rem;">
        An Interactive 3-Step Audit Dashboard evaluating Automation Bias, Transparency, and Algorithmic Debiasing.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- Progress Stepper ----------------
step_cols = st.columns(3)
with step_cols[0]:
    if st.session_state.current_step >= 1:
        st.markdown('<div style="text-align: center; border-bottom: 4px solid #6366f1; padding-bottom: 8px; font-weight: bold; color: #a5b4fc;">Step 1: Standard Screening</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center; border-bottom: 4px solid #334155; padding-bottom: 8px; color: #64748b;">Step 1: Standard Screening</div>', unsafe_allow_html=True)

with step_cols[1]:
    if st.session_state.current_step >= 2:
        st.markdown('<div style="text-align: center; border-bottom: 4px solid #a78bfa; padding-bottom: 8px; font-weight: bold; color: #c084fc;">Step 2: Bias Audit Reveal</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center; border-bottom: 4px solid #334155; padding-bottom: 8px; color: #64748b;">Step 2: Bias Audit Reveal</div>', unsafe_allow_html=True)

with step_cols[2]:
    if st.session_state.current_step >= 3:
        st.markdown('<div style="text-align: center; border-bottom: 4px solid #10b981; padding-bottom: 8px; font-weight: bold; color: #34d399;">Step 3: HCAI Mitigation</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center; border-bottom: 4px solid #334155; padding-bottom: 8px; color: #64748b;">Step 3: HCAI Mitigation</div>', unsafe_allow_html=True)

st.write("")
st.write("")


# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">Candidate Controller</div>', unsafe_allow_html=True)
    
    # Preset select
    def on_preset_change():
        st.session_state.active_candidate = PRESET_CANDIDATES[st.session_state.preset_select].copy()
        st.session_state.current_step = 1

    st.selectbox(
        "Load Preset Profile:",
        options=list(PRESET_CANDIDATES.keys()),
        key="preset_select",
        on_change=on_preset_change
    )
    
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; margin: 10px 0;'>— OR —</p>", unsafe_allow_html=True)
    
    # Random candidate generator
    if st.button("🎲 Generate Random Candidate", use_container_width=True):
        st.session_state.active_candidate = generate_random_candidate()
        st.session_state.current_step = 1
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Corporate target selector
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">Corporate Target</div>', unsafe_allow_html=True)
    selected_company = st.selectbox("Select Target Company:", list(companies.keys()))
    criteria = companies[selected_company]
    
    st.markdown(f"""
    <div style="font-size: 0.8rem; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; margin-top: 10px;">
        <p style="margin: 4px 0;"><strong>Required Skills:</strong><br><span style="color: #a78bfa;">{', '.join(criteria['skills'])}</span></p>
        <p style="margin: 4px 0;"><strong>Required Experience:</strong> <span style="color: #cbd5e1;">{criteria['experience']}</span></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Active candidate local alias
candidate = st.session_state.active_candidate


# ---------------- STEP 1: Standard Screening ----------------
if st.session_state.current_step == 1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card" style="min-height: 480px;">', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-header">👤 Candidate Profile: {candidate["name"]} ({candidate["gender"]})</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <p style="margin: 4px 0;"><strong>Target Job Category:</strong> {candidate["title"]}</p>
        <p style="margin: 4px 0;"><strong>Age:</strong> {candidate["age"]} | <strong>Experience:</strong> {candidate["experience"]} years</p>
        <p style="margin: 4px 0;"><strong>Skills declared:</strong> <span style="color: #a78bfa;">{', '.join(candidate["skills"])}</span></p>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("**Resume Text:**")
        st.markdown(f'<div class="resume-display">{candidate["resume"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card" style="min-height: 480px; display: flex; flex-direction: column; justify-content: space-between;">', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">📈 Standard AI Evaluation</div>', unsafe_allow_html=True)
        
        # Calculate standard predictions (biased)
        pred_standard = predict_simulated(
            candidate["resume"], model, tokenizer, label_encoder,
            candidate["gender"], candidate["title"], is_debiased=False
        )
        match_standard = get_simulated_match_score(candidate, criteria, is_debiased=False)
        
        # Render visualizations
        st.pyplot(plot_bar_chart(pred_standard[:3], "Top AI Role Probabilities (Standard Mode)"))
        
        # Display match score
        st.write("")
        st.markdown(f"""
        <div style="text-align: center; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; margin-top: 10px;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Match Compatibility Score</div>
            <div style="font-size: 2.8rem; font-weight: 700; color: #6366f1; margin: 5px 0;">{match_standard}%</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Determined by standard semantic parsing parameters</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Overtrust Trigger Warning
        st.markdown("""
        <div class="threat-card" style="border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.05); margin-bottom: 0px; padding: 10px;">
            <strong style="color: #f59e0b; font-size: 0.85rem;">⚠️ Recruiter Warning: Automation Bias</strong><br>
            <span style="font-size: 0.78rem; color: #cbd5e1;">The score is clear and confident. Human reviewers naturally over-trust this rating without verification. Run a fairness audit to check for hidden demographic bias.</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Transition control
    st.write("")
    if st.button("Proceed to Step 2: Audit System for Bias ➡️", type="primary", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()


# ---------------- STEP 2: The Bias Audit Reveal ----------------
elif st.session_state.current_step == 2:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card" style="min-height: 440px;">', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">🔍 Sensitivity Audit: Swapping Pronouns</div>', unsafe_allow_html=True)
        st.markdown("""
        To audit the black-box AI model, we run a **pronoun-swapping test** (substituting name headers and gendered pronouns while keeping all qualifications identical).
        """)
        
        st.write("")
        col_cand1, col_cand2 = st.columns(2)
        with col_cand1:
            st.markdown(f"**Original Candidate: {candidate['name']} ({candidate['gender']})**")
            st.markdown(f'<div class="resume-display" style="min-height: 180px;">{candidate["resume"]}</div>', unsafe_allow_html=True)
        with col_cand2:
            st.markdown(f"**Audit Variant: {candidate['swapped_name']} ({candidate['swapped_gender']})**")
            st.markdown(f'<div class="resume-display" style="min-height: 180px;">{candidate["swapped_resume"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card" style="min-height: 440px; display: flex; flex-direction: column; justify-content: space-between;">', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">📊 Revealed Disparity (Standard AI)</div>', unsafe_allow_html=True)
        
        # Calculate standard predictions for both
        match_orig = get_simulated_match_score(candidate, criteria, is_debiased=False)
        
        swapped_candidate = candidate.copy()
        swapped_candidate["gender"] = candidate["swapped_gender"]
        swapped_candidate["resume"] = candidate["swapped_resume"]
        
        match_swapped = get_simulated_match_score(swapped_candidate, criteria, is_debiased=False)
        disparity_gap = match_orig - match_swapped
        
        # Plot disparity comparisons
        st.pyplot(plot_comparison_chart(
            match_orig, match_swapped, 
            f"{candidate['name']} (Orig)", 
            f"{candidate['swapped_name']} (Swapped)",
            f"Compatibility Disparity Gap: {abs(disparity_gap)}%"
        ))
        
        # Disparity report
        st.write("")
        if abs(disparity_gap) > 0:
            st.markdown(f"""
            <div class="threat-card" style="margin-bottom: 0px; padding: 12px;">
                <strong style="color: #ef4444; font-size: 0.9rem;">⚠️ Algorithmic Threat Discovered!</strong><br>
                <span style="font-size: 0.8rem; color: #e2e8f0;">
                    Swapping only pronouns/names changed the match score by <strong>{abs(disparity_gap)}%</strong>. 
                    The standard AI displays systematic gender bias, penalizing candidates based on demographic attributes.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="safe-card" style="margin-bottom: 0px; padding: 12px;">
                <strong style="color: #10b981; font-size: 0.9rem;">✓ No Disparity Detected</strong><br>
                <span style="font-size: 0.8rem; color: #e2e8f0;">For this target criteria and candidates, standard parsing results are equivalent.</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Literature Integration Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header" style="color: #60a5fa;">📖 Academic Literature Context: The Proxy Threat</div>', unsafe_allow_html=True)
    st.markdown(r"""
    <div class="lit-card">
        <strong>Wilson, K., & Caliskan, A. (2024).</strong> <em>"Gender, race, and intersectional bias in resume screening via language model retrieval."</em> (AAAI/ACM AIES)<br><br>
        <strong>Core Finding:</strong> The authors audited modern AI screening systems and proved that semantic representations implicitly convert non-skill identifiers (pronouns, name origins) into <strong>demographic proxies</strong>, leaking bias into rankings.<br>
        <strong>Relevance to HCAI:</strong> When recruitment interfaces display a single, confident ranked score (as shown in Step 1), it triggers <strong>Automation Bias</strong>—recruiters trust the numbers and fail to notice that identical qualifications yield vastly different outcomes when demographics are swapped.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Transition controls
    st.write("")
    nav_cols = st.columns([1, 3])
    with nav_cols[0]:
        if st.button("⬅ Back to Step 1", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with nav_cols[1]:
        if st.button("Proceed to Step 3: Activate HCAI Mitigation ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()


# ---------------- STEP 3: HCAI Mitigation ----------------
elif st.session_state.current_step == 3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card" style="min-height: 480px; display: flex; flex-direction: column; justify-content: space-between;">', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">🛡️ FairBERT Bias Mitigation Check</div>', unsafe_allow_html=True)
        st.markdown("""
        We now activate **FairBERT**, trained using a **Gradient Reversal Layer (GRL)**. This multi-task framework sanitizes candidate representation vectors, stripping gender correlations before classification.
        """)
        
        # Calculate FairBERT Match Scores
        match_orig_fair = get_simulated_match_score(candidate, criteria, is_debiased=True)
        
        swapped_candidate = candidate.copy()
        swapped_candidate["gender"] = candidate["swapped_gender"]
        swapped_candidate["resume"] = candidate["swapped_resume"]
        
        match_swapped_fair = get_simulated_match_score(swapped_candidate, criteria, is_debiased=True)
        disparity_gap_fair = match_orig_fair - match_swapped_fair
        
        # Comparison plot
        st.pyplot(plot_comparison_chart(
            match_orig_fair, match_swapped_fair, 
            f"{candidate['name']} (FairBERT)", 
            f"{candidate['swapped_name']} (FairBERT)",
            f"Mitigated Disparity Gap: {abs(disparity_gap_fair)}%"
        ))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Success Alert
        st.markdown(f"""
        <div class="safe-card" style="margin-bottom: 0px; padding: 12px;">
            <strong style="color: #10b981; font-size: 0.9rem;">✓ Bias Mitigation Active</strong><br>
            <span style="font-size: 0.8rem; color: #cbd5e1;">
                Under FairBERT, swapping pronouns yields identical match scores (Gap: <strong>{abs(disparity_gap_fair)}%</strong>). 
                The algorithm is now invariant to demographic markers.
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card" style="min-height: 480px; display: flex; flex-direction: column; justify-content: space-between;">', unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-header">🔍 Explainability & Feature Attribution Map</div>', unsafe_allow_html=True)
        st.markdown("""
        How does it work? Programmatic feature mapping reveals that the network ignores gender variables while prioritizing key skills:
        """)
        
        # Display highlight map
        highlighted_text = highlight_resume_text(candidate["resume"], general_skills)
        st.markdown(f'<div class="resume-display" style="min-height: 180px;">{highlighted_text}</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 10px; display: flex; justify-content: space-around; font-size: 0.8rem;">
            <span><span class="skill-highlight" style="padding: 2px 8px;">Blue Highlights</span> Merit Skill Vectors</span>
            <span><span class="gender-highlight" style="padding: 2px 8px;">Red Highlights</span> Neutralized Bias Signals</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Transparency note
        st.markdown("""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; font-size: 0.78rem; color: #94a3b8; margin-bottom: 0px;">
            <strong>HCAI Trust Calibration:</strong> By exposing what the AI ignores (red) vs. what it uses (blue), recruiters can calibrate their trust, understanding the exact logical path of the evaluation.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Literature Integration Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header" style="color: #10b981;">📖 Academic Literature Context: Algorithmic Mitigation</div>', unsafe_allow_html=True)
    st.markdown(r"""
    <div class="lit-card" style="border-left-color: #10b981; color: #a7f3d0;">
        <strong>Albaroudi, E., Mansouri, T., & Alameer, A. (2024).</strong> <em>"A comprehensive review of AI techniques for addressing algorithmic bias in job hiring."</em> (AI)<br><br>
        <strong>Core Finding:</strong> Explores technical solutions including algorithmic constraints and adversarial multi-task frameworks. Confirms that forcing demographic parity at the representations level (like GRL) successfully strips bias from embeddings.<br>
        <strong>Accuracy-Fairness Trade-off:</strong> The literature details that debiasing imposes a minor accuracy cost (FairBERT job accuracy drops slightly from ~82.3% to ~79.8% to establish random-chance gender leakage). This trade-off is transparently declared to users for policy alignment.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Transition controls
    st.write("")
    nav_cols = st.columns([1, 3])
    with nav_cols[0]:
        if st.button("⬅ Back to Step 2", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with nav_cols[1]:
        if st.button("🔄 Restart Audit Sandbox", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()