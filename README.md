
# Resume Parser and ATS Score Analyzer

This project is a tool for analyzing resumes by comparing them with job descriptions, extracting relevant skills, calculating an ATS (Applicant Tracking System) score, and providing suggestions for improvement. It also includes an AI-powered Cover Letter Generator for creating personalized cover letters. The project offers detailed visualizations for better analytics.

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
