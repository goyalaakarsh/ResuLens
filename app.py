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
from google.cloud import vision
from collections import Counter
import requests
from textstat import flesch_kincaid_grade
from google.cloud import aiplatform
from dotenv import load_dotenv
import google.generativeai as genai
import pymupdf
import statistics


load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_GEMINI_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

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
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(file_path)
        else:
            st.error("Unsupported file format.")
            return None
        return text
    except Exception as e:
        st.error(f"Error while extracting text: {e}")
        return None

def extract_text_from_pdf(file_path):
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

def extract_text_from_image(image_path):
    api_key = 'K83939171788957'
    url = 'https://api.ocr.space/parse/image'
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            url,
            files={'filename': image_file},
            data={'apikey': api_key}
        )
    result = response.json()
    return result.get('ParsedResults', [{}])[0].get('ParsedText', '')

## Extracting skills
def extract_skills_nlp(text, job_roles):
    doc = nlp(text.lower())
    
    section_keywords = {
        "Languages": ["languages", "programming languages"],
        "Frontend": ["frontend", "front-end", "ui", "user interface"],
        "Backend": ["backend", "back-end", "server-side"],
        "Design": ["design", "ui/ux", "user experience", "prototyping"]
    }
    
    custom_skills = set([
        "machine learning", "deep learning", "natural language processing", 
        "data analysis", "html", "css", "react", "react.js", "next", "next.js", 
        "node", "node.js", "express.js", "mongodb", "flutter", "canva", "blender"
    ])

    skill_variations = {
        "react.js": ["react", "react.js"],
        "next.js": ["next", "next.js"],
        "node.js": ["node", "node.js"],
        "express.js": ["express", "express.js"],
        "mongodb": ["mongo", "mongodb"],
        "flutter": ["flutter"],
        "canva": ["canva"],
        "blender": ["blender"]
    }
    
    potential_skills = set(
        [chunk.text for chunk in doc.noun_chunks] + 
        [ent.text for ent in doc.ents] + 
        [f"{token.text} {token.head.text}" for token in doc if token.pos_ == "ADJ" and token.head.pos_ == "NOUN"]
    )

    all_skills = set(skill.lower() for skills in job_roles.values() for skill in skills)
    
    extracted_skills = potential_skills.intersection(all_skills.union(custom_skills))

    programming_keywords = set([
        "python", "java", "c++", "javascript", "sql", "r", 
        "html", "css", "react", "react.js", "node", "node.js", "express.js", 
        "next", "next.js", "flutter", "figma", "canva", "blender", 
        "mongodb", "c", "swift", "spring boot", "bootstrap", "tailwind"
    ])

    structured_skills = []
    for section, keywords in section_keywords.items():
        for keyword in keywords:
            pattern = re.compile(rf"{keyword}[:\-\s]+([\w\s,./+]+)", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                section_skills = match.group(1).split(",")
                structured_skills.extend([skill.strip().lower() for skill in section_skills])

    extracted_skills.update(programming_keywords.intersection(set(text.lower().split())))
    extracted_skills.update(set(structured_skills))
    
    normalized_skills = set()
    for skill in extracted_skills:
        for base_skill, variations in skill_variations.items():
            if skill in variations:
                normalized_skills.add(base_skill)
                break
        else:
            normalized_skills.add(skill)

    filtered_skills = [skill for skill in normalized_skills if len(skill.split()) <= 2]

    return list(filtered_skills)

## Comparing similarity between the job description and the text extracted from the resume
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

def extract_requirements(doc):
    requirement_phrases = ["required", "must", "should", "essential", "proficiency", "experience with", "ability to"]
    requirements = []

    for sent in doc.sents:
        if any(phrase in sent.text.lower() for phrase in requirement_phrases):
            requirements.append(sent.text.strip())

    return requirements

def extract_responsibilities(doc):
    responsibility_phrases = ["responsible for", "duties include", "will", "task", "role includes", "develop", "build"]
    responsibilities = []

    for sent in doc.sents:
        if any(phrase in sent.text.lower() for phrase in responsibility_phrases):
            responsibilities.append(sent.text.strip())

    return responsibilities

def extract_skills(job_description, job_roles):
    return extract_skills_nlp(job_description, job_roles)

## Job description analyzer
def analyze_job_description(job_description):
    doc = nlp(job_description)

    requirements = extract_requirements(doc)
    responsibilities = extract_responsibilities(doc)
    skills = extract_skills(job_description, JOB_ROLES)
    key_skills = set([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3])
    company_pattern = re.compile(r"company[:\-\s]+([\w\s]+)", re.IGNORECASE)
    company_name = company_pattern.search(job_description).group(1).title() if company_pattern.search(job_description) else "Company"
        
    return {
        "requirements": requirements,
        "responsibilities": responsibilities,
        "skills": skills,
        "key_skills": key_skills,
        "company_name": company_name
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

## ATS Score calculator
def calculate_ats_score(resume_text, job_description, resume_skills, job_skills, pdf_path):
    keyword_match = len(set(resume_skills).intersection(
        set(job_skills))) / len(job_skills) if job_skills else 0

    content_similarity = calculate_similarity(resume_text, job_description)
    achievements = extract_achievements(resume_text)
    achievement_score = min(len(achievements) / 3, 1)
    format_score = 1 if all(keyword in resume_text.lower() for keyword in [
                            "experience", "education", "skills"]) else 0
    evaluation_results = evaluate_pdf_formatting(pdf_path)
    formatting_score = calculate_formatting_score(evaluation_results) / 100 

    scores = {
        "Keyword Matching (30%)": keyword_match * 30,
        "Content Similarity (25%)": content_similarity * 25,
        "Achievements (15%)": achievement_score * 15,
        "Basic Format and Structure (10%)": format_score * 10,
        "PDF Formatting (20%)": formatting_score * 20
    }

    ats_score = sum(scores.values())

    return min(ats_score, 100), scores

## Resume editor
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

## Improvement suggestions generator
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

## Extracting resume details
def extract_resume_details(resume_text):
    """
    Extracts candidate information such as name, skills, and experience from the resume text.
    """
    doc = nlp(resume_text.lower())
    name_pattern = re.compile(r"name[:\-\s]+([\w\s]+)", re.IGNORECASE)
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    phone_pattern = re.compile(r"\+?\d[\d -]{8,12}\d")

    name = name_pattern.search(resume_text).group(1).title() if name_pattern.search(resume_text) else "Candidate"
    email = email_pattern.search(resume_text).group(0) if email_pattern.search(resume_text) else "No email found"
    phone = phone_pattern.search(resume_text).group(0) if phone_pattern.search(resume_text) else "No phone number found"

    skills = set([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3])

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
    }

## Cover Letter Generator Helper Function
def generate_cover_letter(resume_details, job_details):
    prompt = f"""
    Write a professional and personalized cover letter based on the following details. The cover letter should not include any headers or placeholders and should end with the applicant's name and contact information.

    **Applicant Details:**
    - **Name:** {resume_details['name']}
    - **Email:** {resume_details['email']}
    - **Phone:** {resume_details['phone']}
    - **Skills:** {', '.join(resume_details['skills'])}

    **Job Details:**
    - **Company Name:** {job_details['company_name']}
    - **Job Position:** [Insert job position here]
    - **Key Skills Required:** {', '.join(job_details['key_skills'])}

    **Instructions:**
    1. **Salutation:** Begin with a professional salutation to the hiring team.
    2. **Introduction:** Express genuine enthusiasm for the specific job position and the company. Mention how you found out about the opportunity.
    3. **Skills Alignment:** Explain how your skills and experiences align with the job requirements, using specific examples.
    4. **Achievements:** Highlight relevant achievements or projects that demonstrate your fit for the role.
    5. **Closing Statement:** End with a positive closing statement that encourages the hiring team to contact you. Reiterate your interest in the position and express eagerness to contribute to the company.
    6. **Signature and Contact Information:** Conclude with your name and provide your contact details (email and phone number).

    The letter should be engaging, professional, and concise, ensuring all details are correctly included without any extra headings or placeholders.
    """    
    response = model.generate_content(prompt)
    
    return response.text

## Cover Letter Generator
def auto_generate_cover_letter(resume_text, job_description):
    """
    Main function that handles the entire process of generating the cover letter.
    """
    resume_details = extract_resume_details(resume_text)
    job_details = analyze_job_description(job_description)
    cover_letter = generate_cover_letter(resume_details, job_details)
    
    return cover_letter

## Evaluating PDF formattign
def evaluate_pdf_formatting(pdf_path):
    doc = pymupdf.open(pdf_path)
    
    font_info = {}
    margins = []
    line_spacings = []
    text_alignments = []
    page_numbers = []
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        page_fonts = []
        page_margins = {"left": [], "right": [], "top": [], "bottom": []}
        page_line_spacings = []
        page_alignments = []
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font = span["font"]
                        font_size = span["size"]
                        text = span["text"]
                        
                        # Update font_info dictionary
                        if font not in font_info:
                            font_info[font] = {"size": set(), "count": 0}
                        font_info[font]["count"] += 1
                        font_info[font]["size"].add(font_size)
                        
                        page_alignments.append(determine_text_alignment(span["origin"][0], page.rect.width))
                    
                    # Line spacing
                    if len(line["spans"]) > 0:
                        page_line_spacings.append(line["spans"][0]["origin"][1] - line["spans"][-1]["origin"][1])
                
                # Margins
                bbox = pymupdf.Rect(block["bbox"])
                page_margins["left"].append(bbox.x0)
                page_margins["right"].append(page.rect.width - bbox.x1)
                page_margins["top"].append(bbox.y0)
                page_margins["bottom"].append(page.rect.height - bbox.y1)
        
        margins.append(page_margins)
        line_spacings.extend(page_line_spacings)
        text_alignments.extend(page_alignments)
        
        # Check for page numbers
        page_numbers.append(detect_page_number(page, page_num + 1, total_pages))
    
    return {
        "font_consistency": evaluate_font_consistency(font_info),
        "margin_consistency": evaluate_margin_consistency(margins),
        "line_spacing_consistency": evaluate_line_spacing_consistency(line_spacings),
        "text_alignment_consistency": evaluate_text_alignment_consistency(text_alignments),
        "page_number_consistency": evaluate_page_number_consistency(page_numbers, total_pages)
    }

def evaluate_font_consistency(font_info):
    font_issues = []
    for font, data in font_info.items():
        if len(data["size"]) > 3:
            font_issues.append(f"Inconsistent font sizes found for {font}. Sizes: {data['size']}")
    
    return "Consistent" if not font_issues else "\n".join(font_issues)

def evaluate_margin_consistency(margins):
    left_margins = [min(page["left"]) for page in margins]
    right_margins = [min(page["right"]) for page in margins]
    top_margins = [min(page["top"]) for page in margins]
    bottom_margins = [min(page["bottom"]) for page in margins]
    
    margin_consistency = all(
        abs(left_margins[0] - margin) < 5 for margin in left_margins
    ) and all(
        abs(right_margins[0] - margin) < 5 for margin in right_margins
    ) and all(
        abs(top_margins[0] - margin) < 5 for margin in top_margins
    ) and all(
        abs(bottom_margins[0] - margin) < 5 for margin in bottom_margins
    )
    
    if margin_consistency:
        return "Margins are consistent across all pages"
    else:
        return "Inconsistent margins detected across pages"

def evaluate_line_spacing_consistency(line_spacings):
    if not line_spacings:
        return "No line spacing data available"
    
    avg_spacing = statistics.mean(line_spacings)
    std_dev = statistics.stdev(line_spacings) if len(line_spacings) > 1 else 0
    
    if std_dev / avg_spacing < 0.1:
        return f"Consistent line spacing (mean: {avg_spacing:.2f}, std dev: {std_dev:.2f})"
    else:
        return f"Inconsistent line spacing detected (mean: {avg_spacing:.2f}, std dev: {std_dev:.2f})"

def evaluate_text_alignment_consistency(alignments):
    alignment_counts = Counter(alignments)
    total = sum(alignment_counts.values())
    
    if len(alignment_counts) == 1:
        return f"Consistent text alignment: {alignments[0]}"
    else:
        return "Mixed text alignments detected: " + ", ".join(f"{align}: {count/total:.1%}" for align, count in alignment_counts.items())

def evaluate_page_number_consistency(page_numbers, total_pages):
    if total_pages == 1:
        return "Single-page document, page numbering not applicable (best case)"
    
    numbered_pages = sum(page_numbers)
    numbering_percentage = (numbered_pages / total_pages) * 100

    if numbered_pages == total_pages:
        return f"Consistent page numbering detected on all {total_pages} pages"
    elif numbered_pages == 0:
        return f"No page numbers detected in {total_pages}-page document (worst case)"
    else:
        return f"Inconsistent page numbering: {numbered_pages} out of {total_pages} pages numbered ({numbering_percentage:.1f}%)"

def determine_text_alignment(x_position, page_width):
    left_margin = 50
    right_margin = page_width - 50
    
    if x_position < left_margin:
        return "left"
    elif x_position > right_margin:
        return "right"
    else:
        return "center"

def detect_page_number(page, expected_number, total_pages):
    text = page.get_text()
    lines = text.split('\n')
    
    potential_numbers = [lines[0].strip(), lines[-1].strip()]
    
    for num in potential_numbers:
        if num.isdigit() and int(num) == expected_number:
            return True
        elif num.lower() in ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']:
            return True
    
    return False

def calculate_formatting_score(evaluation_results):
    score = 0
    total_weight = 0

    # Font consistency (30% weight)
    if evaluation_results["font_consistency"] == "Consistent":
        score += 30
    total_weight += 30

    # Margin consistency (25% weight)
    if evaluation_results["margin_consistency"] == "Margins are consistent across all pages":
        score += 25
    total_weight += 25

    # Line spacing consistency (20% weight)
    if "Consistent line spacing" in evaluation_results["line_spacing_consistency"]:
        score += 20
    total_weight += 20

    # Text alignment consistency (15% weight)
    if "Consistent text alignment" in evaluation_results["text_alignment_consistency"]:
        score += 15
    total_weight += 15

    # Page number consistency (10% weight)
    if "Consistent page numbering" in evaluation_results["page_number_consistency"]:
        score += 10
    elif "Single-page document" in evaluation_results["page_number_consistency"]:
        score += 10 
    total_weight += 10

    final_score = (score / total_weight) * 100 if total_weight > 0 else 0
    return final_score

## Displaying all the results 
def display_job_match_results(ats_score, detailed_scores, resume_analysis, job_analysis, resume_text, job_description, pdf_analysis):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ ATS Score", f"{ats_score:.2f}%")
    with col2:
        st.metric("✅ Matched Skills", len(set(resume_analysis["skills"]).intersection(set(job_analysis["skills"]))))
    with col3:
        st.metric("🚩 Missing Skills", len(set(job_analysis["skills"]) - set(resume_analysis["skills"])))

    st.progress(ats_score / 100)
    
    st.write("---")

    st.write("### 🧠 Analysis")
    col1, col2, col3 = st.columns(3, gap="large")

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
        
    st.write("---")

    st.write("### 📊 Visualizations")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        all_skills = Counter(resume_analysis["skills"] + job_analysis["skills"])
        skill_df = pd.DataFrame.from_dict(all_skills, orient='index', columns=['Count']).reset_index()
        skill_df.columns = ['Skill', 'Count']
        fig_skills = px.bar(skill_df, x='Skill', y='Count', color='Count', title='Skill Distribution')
        st.plotly_chart(fig_skills, use_container_width=True)

    with col2:
        fig_ats = px.bar(score_df, x='Criteria', y='Score', title='ATS Score Distribution by Criteria')
        st.plotly_chart(fig_ats, use_container_width=True)
        
    st.write("---")

    col1, col2 = st.columns(2, gap="large")
    
    with col1:       
        st.write("### 🚀 Improvement Suggestions")
        suggestions = generate_resume_improvement_suggestions(resume_analysis, job_analysis, ats_score, resume_text, job_description)
        for i, suggestion in enumerate(suggestions, 1):
            st.info(f"{i}. {suggestion}")
    with col2:
        st.write("### ✨ AI Generated Personalized Cover Letter")
        cover_letter_text = auto_generate_cover_letter(resume_text, job_description)
        st.info(cover_letter_text)

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
                    resume_text, job_description, resume_analysis["skills"], job_analysis["skills"], temp_file_path)
                
                pdf_analysis = evaluate_pdf_formatting(temp_file_path)

                display_job_match_results(
                    ats_score, detailed_scores, resume_analysis, job_analysis, resume_text, job_description, pdf_analysis)
        else:
            st.error("Please provide both a resume and a job description.")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()