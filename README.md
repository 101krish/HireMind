# HireMind 🌈 — AI Talent Intelligence & Candidate Screening Platform

> A next-generation, multi-agent AI recruiting and scoring pipeline. PRISM ingests candidate profiles and job descriptions, runs deep semantic filtering, evaluates candidates across multiple technical and behavioral dimensions, runs multi-agent panel debates, and serves a high-fidelity interactive dashboard for recruiters.

🔗 **Live Production URL**: [https://hiremind-u4do.onrender.com/](https://hiremind-u4do.onrender.com/)

---

## 🚀 Key Features

*   **Multi-Agent Recruiting Panel**: Leverages specialized AI agents (Senior AI Recruiter, Devil's Advocate, and Debate Consensus Panel) to generate critical evaluations, spot red flags, and highlight hidden talents.
*   **Dimensional Scoring Engine**: Computes structured metrics across 6 key axes: Capability Fit, Career Trajectory, Past Impact, Hidden Talent, Evidence Confidence, and Behavioral alignment.
*   **Adaptive Role Calibration**: Dynamically adjusts weights and scoring criteria depending on the target role seniority and profile (e.g., Senior IC vs. Engineering Lead vs. Startup Engineer).
*   **Interactive Web UI Dashboard**: A dark-mode recruiter workspace supporting drag-and-drop file uploads, real-time pipeline computation progress bars, candidate tier filtering (A/B/C/D), expandable summaries, and interview loops.
*   **Automated Interview Prep Generator**: Automatically creates tailored Technical, Gap Probing, and Behavioral interview question templates based on a candidate's specific background and resume.
*   **Spotlight Shortlist Report**: Computes comparison matrices and identifies specialized spotlights:
    *   **Safe Pick**: Candidate with the highest evidence confidence and solid profile.
    *   **Growth Pick**: Candidate with the steepest career trajectory.
    *   **Dark Horse**: Candidate with exceptional hidden talents or non-obvious strengths.
    *   **Fast Learner**: Candidate with high behavioral agility and rapid progression.

---

## 📁 Repository Structure

```tree
HireMind/
├── main.py                 # FastAPI backend, routing, upload APIs, and co-hosted frontend endpoints
├── precompute.py           # Offline pipeline orchestrator (Fast vector filter -> Deep LLM parsing -> Score calibration)
├── rank.py                 # Scoring engine calibrator & final leaderboard ranking generator
├── config.py               # Pydantic v2 settings management (weights, models, limits)
├── schemas.py              # Pydantic schema definitions for candidates, profiles, and scoring models
├── requirements.txt        # Backend dependencies (FastAPI, Anthropic, docx parser, etc.)
├── agents/                 # Multi-Agent evaluation engines
│   ├── base.py             # Base agent orchestration class
│   ├── recruiter.py        # Technical Recruiter Agent
│   ├── devils_advocate.py  # Critical review / Gap probing Agent
│   └── debate.py           # Consensus and debate orchestrator Agent
├── engines/                # Modular scoring engines (Phase 2)
│   ├── input_engine.py     # Resumes & document parsing
│   └── scoring/            # Metric computation modules
│       ├── capability.py   # Skill matching and technical capability scorer
│       ├── trajectory.py   # Job duration and career progression scorer
│       ├── impact.py       # Quantified impact & achievements scorer
│       ├── behavioral.py   # Culture fit & soft skills scorer
│       └── confidence.py   # Verifiability & evidence scorer
├── data/                   # Default schemas, docs, and upload target folders
│   ├── uploads/            # Location for files uploaded via configure screen (git-ignored)
│   ├── sample_candidates.json # Pre-loaded list of 100+ candidates for testing
│   └── capability_graph.json # Standard capability dictionary mapping
├── static/                 # Co-hosted recruiter interface web pages (Phase 5)
│   ├── upload_configure.html  # Config landing page, drag-and-drop file uploader & status progress bar
│   ├── main_dashboard.html    # Core recruiter workspace with candidate cards, filters, and expanders
│   ├── candidate_detail.html  # Detailed candidate overview, scores, consensus, and interview tabs
│   └── shortlist_report.html  # Final comparison matrix, export tool, and spotlight highlights
└── results/                # Output destination for precomputed features & final CSV leaderboards
```

---

## 🛠️ Step-by-Step Implementation Phases

PRISM was constructed incrementally across 5 distinct phases to guarantee robustness and separation of concerns:

### 🔹 Phase 1: Input Parsing & Document Processing
*   **Resume Parser**: Built extractors in [input_engine.py](file:///C:/Users/Krish%20Maheshwari/OneDrive/Desktop/HireMind/engines/input_engine.py) to ingest Word/Docx, PDF, Markdown, and raw Text documents, handling layout normalization.
*   **Schema Enforcement**: Normalizes raw resume inputs into the standardized, strict candidate format defined in [schemas.py](file:///C:/Users/Krish%20Maheshwari/OneDrive/Desktop/HireMind/schemas.py).

### 🔹 Phase 2: Dimensional Scoring Engines
*   Created five independent scoring engines in [engines/scoring/](file:///C:/Users/Krish%20Maheshwari/OneDrive/Desktop/HireMind/engines/scoring/) measuring different candidate signals:
    1.  **Capability Scorer**: Matches skills against the Job Description using semantic weights.
    2.  **Trajectory Scorer**: Evaluates duration at previous roles, graduation dates, and promotion velocity.
    3.  **Impact Scorer**: Scores candidate's direct achievements using action verbs and metrics.
    4.  **Behavioral Scorer**: Evaluates leadership, ownership, and adaptability indicators.
    5.  **Evidence Scorer**: Evaluates confidence levels in claims based on external references, links, and detailed achievements.
*   Implemented **Calibrated Scoring** in [rank.py](file:///C:/Users/Krish%20Maheshwari/OneDrive/Desktop/HireMind/rank.py) to normalize scores and apply dynamic weights per role profile.

### 🔹 Phase 3: Multi-Agent Recruiter Panel
*   Developed a collaborative LLM agent network:
    *   **Senior AI Recruiter**: Synthesizes the core profile highlights, strengths, and reasons to hire.
    *   **Devil's Advocate**: Acts as a critical reviewer, probing gaps, highlighting red flags, and designing targeted verification questions.
    *   **Debate Consensus Panel**: Reconciles conflicting opinions, creates consensus summaries, and generates a structured candidate tier (A, B, C, or D).

### 🔹 Phase 4: API Backend & Robustness
*   Designed the FastAPI app in [main.py](file:///C:/Users/Krish%20Maheshwari/OneDrive/Desktop/HireMind/main.py) to handle jobs, run status, candidate lookups, and leaderboard extraction.
*   **Atomic Writing for State Persistence**: Solved file lock/truncation race conditions during polling by writing status updates to a `.tmp` file and executing atomic `os.replace` operations, backed by a retry-loop reader.
*   Added comprehensive verification scripts (`verify_setup.py`, `verify_scoring.py`, `verify_agents.py`, `verify_api.py`) for automated quality assurance.

### 🔹 Phase 5: Web UI & Dashboard Integration
*   Co-hosted the front-end web templates inside the FastAPI app served at clean, user-friendly endpoints (`/configure`, `/dashboard`, `/candidate`, `/report`).
*   Configured **Simulation Mode Pacing** in [precompute.py](file:///C:/Users/Krish%20Maheshwari/OneDrive/Desktop/HireMind/precompute.py) (0.05s wait delay per candidate during mock runs) so recruiters get high-fidelity step-by-step progress spinner feedback even when running without a live Anthropic API key.

---

## ⚡ Quickstart Guide

### 1. Installation
Clone the repository and install all required dependencies:
```bash
git clone https://github.com/101krish/HireMind.git
cd HireMind
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# If ANTHROPIC_API_KEY is omitted or set to 'mock_key', the system running precompute 
# will automatically fallback to simulation mode with high-fidelity pacing.
```

### 3. Running the Server
Start the local FastAPI development server:
```bash
python main.py
```
By default, the server runs on `http://127.0.0.1:8000/`.

### 4. Interactive Web Endpoints
Open your browser and navigate to:
*   **Configure Upload**: [http://127.0.0.1:8000/configure](http://127.0.0.1:8000/configure) — Start here! Upload candidate datasets and Job Descriptions.
*   **Recruiter Dashboard**: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) — View candidate listing, sorting, and expand summaries.
*   **Candidate Profile**: [http://127.0.0.1:8000/candidate](http://127.0.0.1:8000/candidate) — Deep dive into scores, agent debates, and interview loops.
*   **Shortlist Matrix**: [http://127.0.0.1:8000/report](http://127.0.0.1:8000/report) — View role spotlights and download the exportable CSV ranking list.

### 5. Running Verification Suites
To execute the automated end-to-end integration and API tests:
```bash
python verify_setup.py
python verify_scoring.py
python verify_agents.py
python verify_api.py
```

---

*Developed by Krish Maheshwari. Made with 💖 and advanced agentic intelligence.*
