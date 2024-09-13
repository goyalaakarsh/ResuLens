import os
import tempfile
import streamlit as st
import re
from pdfminer.high_level import extract_text
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_ace import st_ace
import pytesseract
from PIL import Image
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(page_title="Advanced Resume Parser & ATS Checker", layout="wide", page_icon="💼")

# Load SpaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Load job roles and skills from JSON file
@st.cache_data
def load_job_roles(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load job roles data
JOB_ROLES = load_job_roles('job_roles.json')

# Improved text extraction function
@st.cache_data
def extract_resume_text(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text(file_path)
        else:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error while extracting text: {e}")
        return None

# Improved skill extraction using NLP and custom rules
def extract_skills_nlp(resume_text, job_roles):
    doc = nlp(resume_text.lower())
    
    # Extract noun phrases, named entities, and adjective-noun pairs
    potential_skills = set(
        [chunk.text for chunk in doc.noun_chunks] + 
        [ent.text for ent in doc.ents] +
        [f"{token.text} {token.head.text}" for token in doc if token.pos_ == "ADJ" and token.head.pos_ == "NOUN"]
    )
    
    # Filter skills based on job roles and custom rules
    all_skills = set(skill.lower() for skills in job_roles.values() for skill in skills)
    custom_skills = set(["machine learning", "deep learning", "natural language processing", "data analysis"])
    extracted_skills = potential_skills.intersection(all_skills.union(custom_skills))
    
    # Add common programming languages and tools
    programming_keywords = set(["python", "java", "c++", "javascript", "sql", "r", "tableau", "excel"])
    extracted_skills.update(programming_keywords.intersection(set(resume_text.lower().split())))
    
    return list(extracted_skills)

# Improved similarity calculation using TF-IDF
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer)[0][1]

# Enhanced job matching algorithm
def match_skills_to_job(extracted_skills, job_roles):
    job_matches = {}
    for job, skills in job_roles.items():
        skill_match = len(set(extracted_skills).intersection(set(skills)))
        total_skills = len(set(skills))
        if total_skills > 0:
            match_score = (skill_match / total_skills) * 100
            job_matches[job] = match_score
    return sorted(job_matches.items(), key=lambda x: x[1], reverse=True)[:3] 

# Improved job description analysis
def analyze_job_description(job_description):
    doc = nlp(job_description)
    
    requirements = [sent.text for sent in doc.sents if any(token.text.lower() in ["required", "must", "should", "essential"] for token in sent)]
    
    responsibilities = [sent.text for sent in doc.sents if any(token.text.lower() in ["responsible", "duties", "will", "task"] for token in sent)]
    
    skills = extract_skills_nlp(job_description, JOB_ROLES)
    
    return {
        "requirements": requirements,
        "responsibilities": responsibilities,
        "skills": skills
    }

# Improved ATS score calculation
def calculate_ats_score(resume_text, job_description, resume_skills, job_skills):
    content_similarity = calculate_similarity(resume_text, job_description)
    skill_match_ratio = len(set(resume_skills).intersection(set(job_skills))) / len(job_skills) if job_skills else 0
    keyword_presence = sum(1 for skill in job_skills if skill.lower() in resume_text.lower()) / len(job_skills) if job_skills else 0
    
    ats_score = (content_similarity * 0.4 + skill_match_ratio * 0.4 + keyword_presence * 0.2) * 100
    return min(ats_score, 100)  # Cap the score at 100

# Display resume editor
def display_resume_editor(initial_content=""):
    st.subheader("📝 Resume Editor")
    st.write("You can draft your resume below.")
    edited_content = st_ace(
        value=initial_content,
        language="markdown",
        theme="monokai",
        key="resume_editor",
        font_size=14,
        tab_size=4,
        height=360,
    )
    return edited_content

def display_job_match_results(ats_score, resume_analysis, job_analysis):
    st.subheader("📊 ATS Analysis Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ ATS Score", f"{ats_score:.2f}%")
    with col2:
        st.metric("✅ Matched Skills", len(set(resume_analysis["skills"]).intersection(set(job_analysis["skills"]))))
    with col3:
        st.metric("🚩 Missing Skills", len(set(job_analysis["skills"]) - set(resume_analysis["skills"])))

    # Progress bar for ATS score
    st.progress(ats_score / 100)

    st.write("### 🧠 Skill Analysis")
    col1, col2, col3 = st.columns([1, 1, 2.5], gap="large")
    with col1:
        st.write("**Matched Skills**")
        matched_skills = set(resume_analysis["skills"]).intersection(set(job_analysis["skills"]))
        st.table(pd.DataFrame({"Skill": list(matched_skills)}))
    with col2:
        st.write("**Skills to Add**")
        missing_skills = set(job_analysis["skills"]) - set(resume_analysis["skills"])
        st.table(pd.DataFrame({"Skill": list(missing_skills)}))
    with col3:
        st.write("### 📊 Skill Distribution")
        all_skills = Counter(resume_analysis["skills"] + job_analysis["skills"])
        skill_df = pd.DataFrame.from_dict(all_skills, orient='index', columns=['Count']).reset_index()
        skill_df.columns = ['Skill', 'Count']
        fig = px.bar(skill_df, x='Skill', y='Count', color='Count', title='Skill Distribution')
        st.plotly_chart(fig, use_container_width=True)

    # Display job requirements and responsibilities in 2 columns
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📌 Job Requirements", expanded=False):
            for req in job_analysis["requirements"]:
                st.write(f"- {req}")
    
    with col2:
        with st.expander("🛠️ Job Responsibilities", expanded=False):
            for resp in job_analysis["responsibilities"]:
                st.write(f"- {resp}")
    
    st.write("### 🎯 Top 3 Suggested Roles")
    suggested_roles = match_skills_to_job(resume_analysis["skills"], JOB_ROLES)
    role_df = pd.DataFrame(suggested_roles, columns=['Role', 'Match Score'])
    # Display only the roles without the match score
    st.table(role_df['Role'])

# Main app logic
def main():
    st.title("Advanced Resume Parser & ATS Checker 📊")

    st.markdown("""
        <style>
        div.stButton > button:first-child {background-color: #4CAF50; color: white;}
        div.stButton > button:hover {background-color: #45a049;}
        .stTextArea textarea {font-size: 14px !important;}
        </style>
    """, unsafe_allow_html=True)

    st.write("Upload your resume and job description to get an ATS score, skill matching, and job role suggestions.")
    
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.header("🔍 Job Description")
        job_description = st.text_area("Paste the job description here", height=560, placeholder="Enter the job description...")

    with col2:
        st.header("📄 Upload Resume")
        uploaded_file = st.file_uploader("Upload your resume (PDF or Image format)", type=["pdf", "png", "jpg", "jpeg"])
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            extracted_text = extract_resume_text(temp_file_path)
            if extracted_text:
                st.subheader("Extracted Resume Text")
                st.text_area("Resume Text", value=extracted_text, height=200, key="resume_text")
                
                # Display extracted skills
                extracted_skills = extract_skills_nlp(extracted_text, JOB_ROLES)
                st.subheader("Extracted Skills")
                st.success(", ".join(extracted_skills) if extracted_skills else "No skills found.")
                
                resume_text = extracted_text
            else:
                resume_text = display_resume_editor("No text extracted. Please try again.")
        else:
            resume_text = display_resume_editor("Upload your resume to see the content here.")

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🚀 Analyze"):
        if resume_text and job_description:
            with st.spinner("Analyzing..."):
                resume_analysis = {
                    "skills": extract_skills_nlp(resume_text, JOB_ROLES)
                }

                job_analysis = analyze_job_description(job_description)

                ats_score = calculate_ats_score(resume_text, job_description, resume_analysis["skills"], job_analysis["skills"])

                display_job_match_results(ats_score, resume_analysis, job_analysis)
        else:
            st.error("Please provide both a resume and a job description.")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()