# Biomedical Symptom Decision Support

Streamlit web app that lets patients or students select symptoms by organ system, capture an overall pain score, and view decision-tree-based differential diagnoses derived from Mayo Clinic public information.

## Biomedical Context

- **Audience:** Biomedical engineering students exploring clinical decision support concepts and patients participating in educational simulations.
- **Goal:** Demonstrate how organ-system symptom grouping plus a pain scale can feed a transparent decision tree and surface likely diagnoses alongside safety guidance.
- **Clinical disclaimer:** Outputs are informational only and do not replace licensed medical care or emergency services.

## Quick Start Instructions

### Opening in GitHub Codespaces

1. Click the green **Code** button in GitHub and choose **Open with Codespaces**.
2. Select this repository and wait for the dev container to finish provisioning (Streamlit prerequisites are already available).

### Running the Streamlit App

```bash
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The app will print a `Local URL` (default `http://localhost:8501`). Open it in the Codespaces forwarded port or your local browser.

## Usage Guide

- **Step 1:** Expand each organ-system accordion (cardio, respiratory, skeletal, neural, GI, muscular, visual) and tick symptoms that apply. Every option is a checkbox for quick selection.
- **Step 2:** Enter how many days each selected symptom has been present in the **Symptom Durations** section.
- **Step 3:** Set the overall pain slider between 0 (no pain) and 10 (extreme pain) to capture severity.
- **Step 4:** Press **Analyze Symptom Profile** to run the duration-aware decision tree. Review the confidence-ranked diagnoses, matched hallmark symptoms, and guidance text. Contact emergency services for high-risk warnings.

## Data Description

- **File:** `data/symptom_knowledge.json`
- **Contents:** Organ-system symptom catalog plus curated diagnosis entries (supporting symptoms, qualitative pain weighting, suggested actions).
- **Source:** Condensed from publicly available Mayo Clinic, Cleveland Clinic, and UF Health Shands patient-education summaries; no proprietary excerpts are stored.

## Project Structure

- `app.py` – Streamlit front end, synthetic decision-tree training routine, and UI logic.
- `data/symptom_knowledge.json` – Human-readable knowledge base powering the model and UI.
- `requirements.txt` – Python dependencies (Streamlit, pandas, scikit-learn, numpy).
- `main.lua` – Original template artifact from Codespaces (unused by the Streamlit workflow but kept for reference).

