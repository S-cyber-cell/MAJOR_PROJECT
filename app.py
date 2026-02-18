import streamlit as st
from utils import extract_text
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import json
import re
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------
# ENV
# -------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in .env")
    st.stop()

client = OpenAI(api_key=api_key)

# -------------------------
# PAGE
# -------------------------
st.set_page_config(page_title="AI Recruitment ATS", layout="wide")
st.title("🤖 AI Recruitment ATS Dashboard")

# -------------------------
# SESSION STATE
# -------------------------
if "reports" not in st.session_state:
    st.session_state.reports = None
if "df" not in st.session_state:
    st.session_state.df = None

# -------------------------
# SAFE JSON
# -------------------------
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
        return None

# -------------------------
# PDF
# -------------------------
def generate_pdf_report(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>Candidate:</b> {data['name']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Score:</b> {data['score']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Decision:</b> {data['decision']}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Summary</b>", styles["Heading3"]))
    elements.append(Paragraph(data["summary"], styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------
# HELPERS
# -------------------------
def trim(text, n=10000):
    return text[:n] if len(text) > n else text

def get_ai_report(resume_text, jd_text, retries=2):

    prompt = f"""
You are an ATS system.

STEP 1: Extract top 8 MOST IMPORTANT skills from the JD.
STEP 2: Score candidate on ONLY those skills from 0–5.

Return STRICT JSON ONLY:

{{
"name":"candidate",
"score":0-100,
"jd_skills":["skill1","skill2"],
"resume_skill_scores":{{"skill1":5,"skill2":2}},
"strengths":["..."],
"gaps":["..."],
"decision":"Hire/Maybe/Reject",
"summary":"...",
"why_selected":"..."
}}

RESUME:
{resume_text}

JOB:
{jd_text}
"""

    for _ in range(retries):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        text = response.choices[0].message.content
        parsed = safe_json_parse(text)
        if parsed:
            return parsed

    return None

# -------------------------
# INPUTS
# -------------------------
st.subheader("Upload Files")

jd_file = st.file_uploader("Job Description", type=["pdf","docx","txt"])
resume_files = st.file_uploader("Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)

run = st.button("Run Screening")

# -------------------------
# RUN AI ONCE
# -------------------------
if run:
    if not jd_file or not resume_files:
        st.warning("Upload JD and resumes")
        st.stop()

    jd_text = trim(extract_text(jd_file))
    reports = []
    progress = st.progress(0)

    for i, file in enumerate(resume_files):
        resume_text = trim(extract_text(file))
        data = get_ai_report(resume_text, jd_text)

        if not data:
            st.warning(f"AI failed for {file.name}")
            continue

        data["file"] = file.name
        reports.append(data)
        progress.progress((i+1)/len(resume_files))

    if not reports:
        st.error("No candidates processed")
        st.stop()

    reports = sorted(reports, key=lambda x: x["score"], reverse=True)

    df = pd.DataFrame([
        {"Candidate": r["name"], "Score": r["score"], "File": r["file"]}
        for r in reports
    ])

    st.session_state.reports = reports
    st.session_state.df = df

# -------------------------
# DISPLAY RESULTS
# -------------------------
if st.session_state.reports:

    reports = st.session_state.reports
    df = st.session_state.df

    # TOP CANDIDATE
    top = reports[0]
    st.success(f"🏆 Top Candidate: {top['name']} (Score {top['score']})")
    st.info(top.get("why_selected","Best match"))

    # TABLE
    st.subheader("Ranking")
    st.dataframe(df, use_container_width=True)

    # -------------------------
    # HEATMAP (JD-based)
    # -------------------------
    st.subheader("JD Skill Heatmap")

    jd_skills = top.get("jd_skills", [])
    names = []
    matrix = []

    for r in reports:
        names.append(r["name"])
        scores = r.get("resume_skill_scores", {})
        row = [min(max(scores.get(s,0),0),5) for s in jd_skills]
        matrix.append(row)

    if matrix and jd_skills:
        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(9,4))
        heat = ax.imshow(matrix, aspect="auto")

        ax.set_xticks(range(len(jd_skills)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(jd_skills, rotation=30, ha="right")
        ax.set_yticklabels(names)

        cbar = plt.colorbar(heat)
        cbar.set_label("Skill Strength (0–5)")

        st.pyplot(fig)

    # -------------------------
    # SELECT CANDIDATE
    # -------------------------
    st.subheader("Candidate Detail")

    selected_file = st.selectbox("Select Candidate", df["File"].tolist())
    selected = next(r for r in reports if r["file"] == selected_file)

    st.markdown(f"### {selected['name']}")
    st.write(selected["summary"])

    # -------------------------
    # CORRECT SKILL GRAPH
    # -------------------------
    st.markdown("### JD-Based Skill Strength")

    jd_skills = selected.get("jd_skills", [])
    scores = selected.get("resume_skill_scores", {})

    plot_data = {s: min(max(scores.get(s,0),0),5) for s in jd_skills}

    if plot_data:
        s = pd.Series(plot_data)

        fig2, ax2 = plt.subplots(figsize=(8,4))
        s.plot(kind="bar", ax=ax2)

        ax2.set_ylabel("Strength (0–5)")
        ax2.set_title("Candidate vs JD Skills")

        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig2)

    # PDF
    pdf = generate_pdf_report(selected)
    st.download_button("Download PDF", pdf, file_name=f"{selected['name']}.pdf", mime="application/pdf")

