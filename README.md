
# ResuLens: Resume Parser and ATS Score Analyzer

This project is a comprehensive tool designed to streamline the recruitment process by leveraging advanced resume parsing, skill extraction, and ATS (Applicant Tracking System) scoring algorithms. It aims to optimize how resumes are analyzed in comparison to specific job descriptions, enabling both applicants and recruiters to assess the strength of a resume with precision and clarity. 

Key features include an intelligent resume parser that breaks down the resume into meaningful components, extracting core skills and relevant information crucial for determining job-fit. The ATS score calculation provides an objective metric that evaluates how well the resume aligns with the job description, giving users a clear understanding of their candidacy potential.

Additionally, the platform offers personalized improvement suggestions, guiding users on how to refine their resumes to increase compatibility with job requirements. For a more in-depth analysis, the tool includes data visualization features, showcasing ATS score distribution and skill matching through intuitive charts and graphs. These visual analytics give a clear overview of where improvements can be made.

One of the standout features is the AI-powered Cover Letter Generator, which creates a highly tailored cover letter based on the provided resume and job description. This allows users to efficiently generate a professional cover letter that highlights their qualifications while being customized to the specific role they are applying for.

Whether you are a job seeker looking to improve your application materials or a recruiter aiming to streamline resume evaluations, this project provides an end-to-end solution for enhancing the job application process. Its blend of automation, analytics, and AI-driven customization makes it an indispensable tool for modern recruitment needs.

## Features

- **Resume Parsing**: Extracts the text and key sections from the uploaded resume.
- **Skills Extraction**: Identifies the skills mentioned in the resume and compares them with those required by the job description.
- **ATS Score Calculation**: Calculates an ATS score based on how well the resume matches the job description.
- **Resume and Job Description Comparison**: Provides insights into the gaps between the resume and the job description.
- **Data Visualization**: Offers various plots for visualization, including ATS score distribution and skill matching.
- **Personalized Suggestions**: Provides improvement suggestions based on ATS analysis.
- **AI-Powered Cover Letter Generator**: Automatically generates a personalized cover letter based on the resume and job description using AI.

## Installation

1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/goyalaakarsh/Resume-ATS-Analyzer
   cd Resume-ATS-Analyzer
   ```

2. **Install Dependencies**  
   Install all necessary dependencies using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` File**  
   You will need to create a `.env` file at the root of the project directory and add your `GOOGLE_GEMINI_KEY` (required for AI-related features). Here’s an example of the `.env` file content:
   ```bash
   GOOGLE_GEMINI_KEY=your-google-gemini-api-key
   ```

4. **Run the Application**  
   You can run the app by executing the `app.py` file:
   ```bash
   python -m streamlit run app.py
   ```
