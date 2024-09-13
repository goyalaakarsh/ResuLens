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
from textstat import flesch_kincaid_grade

st.set_page_config(
    page_title="Advanced Resume Parser & ATS Checker", layout="wide", page_icon="💼")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


nlp = load_spacy_model()

@st.cache_data
def load_job_roles(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

JOB_ROLES = load_job_roles('job_roles.json')

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


def extract_skills_nlp(text, job_roles):
    doc = nlp(text.lower())

    potential_skills = set(
        [chunk.text for chunk in doc.noun_chunks] +
        [ent.text for ent in doc.ents] +
        [f"{token.text} {token.head.text}" for token in doc if token.pos_ ==
            "ADJ" and token.head.pos_ == "NOUN"]
    )

    all_skills = set(skill.lower()
                     for skills in job_roles.values() for skill in skills)
    custom_skills = set(["machine learning", "deep learning",
                        "natural language processing", "data analysis"])
    extracted_skills = potential_skills.intersection(
        all_skills.union(custom_skills))

    programming_keywords = set(
        ["python", "java", "c++", "javascript", "sql", "r", "tableau", "excel"])
    extracted_skills.update(
        programming_keywords.intersection(set(text.lower().split())))

    return list(extracted_skills)



def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer)[0][1]



def match_skills_to_job(extracted_skills, job_roles):
    job_matches = {}
    for job, skills in job_roles.items():
        skill_match = len(set(extracted_skills).intersection(set(skills)))
        total_skills = len(set(skills))
        if total_skills > 0:
            match_score = (skill_match / total_skills) * 100
            job_matches[job] = match_score
    return sorted(job_matches.items(), key=lambda x: x[1], reverse=True)[:3]



def analyze_job_description(job_description):
    doc = nlp(job_description)

    requirements = [sent.text for sent in doc.sents if any(token.text.lower(
    ) in ["required", "must", "should", "essential"] for token in sent)]

    responsibilities = [sent.text for sent in doc.sents if any(
        token.text.lower() in ["responsible", "duties", "will", "task"] for token in sent)]

    skills = extract_skills_nlp(job_description, JOB_ROLES)

    return {
        "requirements": requirements,
        "responsibilities": responsibilities,
        "skills": skills
    }


def extract_education(text):
    education_keywords = ["degree", "bachelor",
                          "master", "phd", "diploma", "certificate"]
    doc = nlp(text.lower())
    education = [sent.text for sent in doc.sents if any(
        keyword in sent.text.lower() for keyword in education_keywords)]
    return education



def extract_experience(text):
    experience_keywords = ["experience", "year",
                           "worked", "job", "position", "role"]
    doc = nlp(text.lower())
    experience = [sent.text for sent in doc.sents if any(
        keyword in sent.text.lower() for keyword in experience_keywords)]
    return experience



def extract_achievements(text):
    achievement_keywords = ["achieved", "increased",
                            "improved", "reduced", "launched", "developed"]
    doc = nlp(text.lower())
    achievements = [sent.text for sent in doc.sents if any(
        keyword in sent.text.lower() for keyword in achievement_keywords)]
    return achievements



def calculate_ats_score(resume_text, job_description, resume_skills, job_skills):

    keyword_match = len(set(resume_skills).intersection(
        set(job_skills))) / len(job_skills) if job_skills else 0


    content_similarity = calculate_similarity(resume_text, job_description)


    achievements = extract_achievements(resume_text)

    achievement_score = min(len(achievements) / 3, 1)

    format_score = 1 if all(keyword in resume_text.lower() for keyword in [
                            "experience", "education", "skills"]) else 0


    scores = {
        "Keyword Matching (35%)": keyword_match * 35,
        "Content Similarity (30%)": content_similarity * 30,
        "Achievements (20%)": achievement_score * 20,
        "Format and Structure (15%)": format_score * 15
    }

    ats_score = sum(scores.values())

    return min(ats_score, 100), scores


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
    return 

def generate_resume_improvement_suggestions(resume_analysis, job_analysis, ats_score, resume_text, job_description):
    suggestions = []
    nlp = spacy.load("en_core_web_sm")

    missing_skills = set(job_analysis["skills"]) - set(resume_analysis["skills"])
    if missing_skills:
        top_missing = list(missing_skills)[:5]
        suggestions.append(f"Consider adding these key skills to your resume: {', '.join(top_missing)}")


    resume_doc = nlp(resume_text.lower())
    job_doc = nlp(job_description.lower())
    
    resume_words = [token.text for token in resume_doc if not token.is_stop and token.is_alpha]
    job_words = [token.text for token in job_doc if not token.is_stop and token.is_alpha]
    
    resume_word_freq = Counter(resume_words)
    job_word_freq = Counter(job_words)
    
    important_job_words = [word for word, count in job_word_freq.most_common(10) if word not in resume_word_freq]
    if important_job_words:
        suggestions.append(f"Try incorporating these important terms from the job description: {', '.join(important_job_words)}")


    achievements = extract_achievements(resume_text)
    if len(achievements) < 3:
        suggestions.append(f"Your resume currently has {len(achievements)} achievement(s). Try to include at least 3 quantifiable achievements to showcase your impact.")
    

    resume_sections = ["experience", "education", "skills"]
    missing_sections = [section for section in resume_sections if section not in resume_text.lower()]
    if missing_sections:
        suggestions.append(f"Consider adding clear sections for: {', '.join(missing_sections)}")

    content_similarity = calculate_similarity(resume_text, job_description)
    if content_similarity < 0.6:
        suggestions.append("Your resume content could be more closely aligned with the job description. Try to tailor your experiences and skills to match the job requirements more closely.")


    word_count = len(resume_text.split())
    if word_count < 300:
        suggestions.append(f"Your resume is quite concise at {word_count} words. Consider adding more details about your experiences and skills.")
    elif word_count > 700:
        suggestions.append(f"Your resume is quite lengthy at {word_count} words. Try to focus on the most relevant information and aim for a more concise presentation.")


    readability_score = flesch_kincaid_grade(resume_text)
    if readability_score > 12:
        suggestions.append("Your resume's readability could be improved. Try using simpler language and shorter sentences to make it easier to scan quickly.")


    resume_verbs = [token for token in resume_doc if token.pos_ == "VERB"]
    past_tense = sum(1 for verb in resume_verbs if verb.tag_ == "VBD")
    present_tense = sum(1 for verb in resume_verbs if verb.tag_ == "VBZ" or verb.tag_ == "VBP")
    if past_tense > 0 and present_tense > 0 and (past_tense / (past_tense + present_tense) < 0.8):
        suggestions.append("Ensure consistent use of verb tenses. Generally, use past tense for previous experiences and present tense for current roles.")


    job_cert_keywords = ["certified", "certificate", "certification", "licensed", "accredited"]
    job_certs = [sent.text for sent in job_doc.sents if any(keyword in sent.text.lower() for keyword in job_cert_keywords)]
    if job_certs and not any(keyword in resume_text.lower() for keyword in job_cert_keywords):
        suggestions.append("The job description mentions certifications. Consider adding relevant certifications or courses to strengthen your qualifications.")

    return suggestions

def display_job_match_results(ats_score, detailed_scores, resume_analysis, job_analysis, resume_text, job_description):
    # st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ ATS Score", f"{ats_score:.2f}%")
    with col2:
        st.metric("✅ Matched Skills", len(set(resume_analysis["skills"]).intersection(set(job_analysis["skills"]))))
    with col3:
        st.metric("🚩 Missing Skills", len(set(job_analysis["skills"]) - set(resume_analysis["skills"])))

    st.progress(ats_score / 100)

    st.write("### 🧠 Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Matched Skills**")
        matched_skills = set(resume_analysis["skills"]).intersection(set(job_analysis["skills"]))
        matched_df = pd.DataFrame({"Skill": list(matched_skills)})
        matched_df.index = matched_df.index + 1
        st.dataframe(matched_df, use_container_width=True)

    with col2:
        st.write("**Skills to Add**")
        missing_skills = set(job_analysis["skills"]) - set(resume_analysis["skills"])
        missing_df = pd.DataFrame({"Skill": list(missing_skills)})
        missing_df.index = missing_df.index + 1
        st.dataframe(missing_df, use_container_width=True)

    with col3:
        st.write("**ATS Score Distribution**")
        score_df = pd.DataFrame.from_dict(detailed_scores, orient='index', columns=['Score']).reset_index()
        score_df.index = score_df.index + 1
        score_df.columns = ['Criteria', 'Score']
        score_df['Score'] = score_df['Score'].round(2)
        st.dataframe(score_df, use_container_width=True)

    st.write("### 📊 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        # st.write("**Skill Distribution**")
        all_skills = Counter(resume_analysis["skills"] + job_analysis["skills"])
        skill_df = pd.DataFrame.from_dict(all_skills, orient='index', columns=['Count']).reset_index()
        skill_df.columns = ['Skill', 'Count']
        fig_skills = px.bar(skill_df, x='Skill', y='Count', color='Count', title='Skill Distribution')
        st.plotly_chart(fig_skills, use_container_width=True)

    with col2:
        # st.write("**ATS Score Distribution**")
        fig_ats = px.bar(score_df, x='Criteria', y='Score', title='ATS Score Distribution by Criteria')
        st.plotly_chart(fig_ats, use_container_width=True)
        
    st.write("### 🚀 Improvement Suggestions")
    suggestions = generate_resume_improvement_suggestions(resume_analysis, job_analysis, ats_score, resume_text, job_description)
    for i, suggestion in enumerate(suggestions, 1):
        st.write(f"{i}. {suggestion}")


def main():
    st.title("Resume ATS Analyzer 📊")

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
        job_description = st.text_area(
            "Paste the job description here", height=560, placeholder="Enter the job description...")

    with col2:
        st.header("📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF or Image format)", type=["pdf", "png", "jpg", "jpeg"])

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            extracted_text = extract_resume_text(temp_file_path)
            if extracted_text:
                st.subheader("Extracted Resume Text")
                st.text_area("Resume Text", value=extracted_text,
                             height=200, key="resume_text")

                extracted_skills = extract_skills_nlp(
                    extracted_text, JOB_ROLES)
                st.subheader("Extracted Skills")
                st.success(", ".join(extracted_skills)
                           if extracted_skills else "No skills found.")

                resume_text = extracted_text
            else:
                resume_text = display_resume_editor(
                    "No text extracted. Please try again.")
        else:
            resume_text = display_resume_editor(
                "Upload your resume to see the content here.")

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("Analyze"):
        if resume_text and job_description:
            with st.spinner("Analyzing..."):
                resume_analysis = {
                    "skills": extract_skills_nlp(resume_text, JOB_ROLES)
                }

                job_analysis = analyze_job_description(job_description)

                ats_score, detailed_scores = calculate_ats_score(
                    resume_text, job_description, resume_analysis["skills"], job_analysis["skills"])

                display_job_match_results(
                    ats_score, detailed_scores, resume_analysis, job_analysis, resume_text, job_description)
        else:
            st.error("Please provide both a resume and a job description.")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()