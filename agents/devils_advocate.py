import sys
import json
import anthropic
from pathlib import Path

# Add project root to sys.path so we can run this module directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings
from schemas import DevilsAdvocate, InterviewQuestions, safe_validate

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

DEVILS_ADVOCATE_PROMPT = """
You are a highly analytical, skeptical recruiter playing the "Devil's Advocate".
Your objective is to scrutinize this candidate's profile to identify potential hiring risks, anomalies, or gaps.

Job Requirements:
{role_requirements}

Candidate's Structured Profile:
{candidate_profile}

Please perform the following:
1. Identify all key career risks (e.g. short stints, sudden domain shifts, title/skill mismatches, lack of production scaling, etc.).
2. Set an overall risk level (low, medium, high).
3. Suggest a mitigation strategy.
4. Formulate specific, tough interview questions:
   - Technical questions probing potential skill weaknesses or depth.
   - Gap probing questions probing tenure gaps or jumps.
   - Behavioral questions probing potential teamwork or ownership weaknesses.

Return ONLY a valid JSON dictionary matching this exact schema:
{{
  "risks": ["risk 1", "risk 2"],  // identified risks/concerns
  "risk_level": "low|medium|high", // string indicating risk level
  "mitigation": "detailed explanation of how to verify/mitigate these risks",
  "interview_questions": {{
    "technical": ["question 1", "question 2"],
    "gap_probing": ["question 1", "question 2"],
    "behavioral": ["question 1", "question 2"]
  }}
}}

Return ONLY valid JSON. No markdown formatting, no explanations, no code fences.
"""

def analyze_risks(role: dict, candidate: dict) -> dict:
    """
    Analyzes profile risks and generates targeted interview questions.
    If the API key is not configured, runs in programmatic simulation mode.
    """
    if settings.anthropic_api_key == "mock_key":
        return simulate_risks(role, candidate)
        
    print(f"[Devil's Advocate] Scrutinizing profile risks for {candidate.get('name', 'Unknown')}...")
    
    prompt = DEVILS_ADVOCATE_PROMPT.format(
        role_requirements=json.dumps(role, indent=2),
        candidate_profile=json.dumps(candidate, indent=2)
    )
    
    try:
        response = client.messages.create(
            model=settings.model_name,
            max_tokens=settings.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.removeprefix("```json").removeprefix("```")
        raw = raw.removesuffix("```").strip()
        
        raw_result = json.loads(raw)
    except Exception as e:
        print(f"[ERROR] Devil's Advocate LLM execution failed: {e}. Falling back to simulation.")
        return simulate_risks(role, candidate)
        
    # Split results into devils_advocate and interview_questions models
    da_data = {
        "risks": raw_result.get("risks", []),
        "risk_level": raw_result.get("risk_level", "low"),
        "mitigation": raw_result.get("mitigation", "")
    }
    iq_data = raw_result.get("interview_questions", {})
    
    validated_da = safe_validate(da_data, DevilsAdvocate)
    validated_iq = safe_validate(iq_data, InterviewQuestions)
    
    if validated_da is None or validated_iq is None:
        print("[ERROR] Devil's Advocate response failed schema validation. Falling back to simulation.")
        return simulate_risks(role, candidate)
        
    return {
        "devils_advocate": validated_da,
        "interview_questions": validated_iq
    }

def simulate_risks(role: dict, candidate: dict) -> dict:
    """
    Programmatic simulation fallback for Devil's Advocate assessment.
    """
    risks = []
    technical_questions = []
    gap_questions = []
    behavioral_questions = []
    
    # 1. Look for tenure issues in timeline
    timeline = candidate.get("timeline", [])
    has_short_stint = False
    for job in timeline:
        start = job.get("year_start")
        end = job.get("year_end")
        if start and end:
            duration = end - start
            if duration <= 1:
                has_short_stint = True
                risks.append(f"Short employment duration ({duration} year) at {job.get('company', 'Unknown')}.")
                
    # 2. Check for capability alignment risks
    cap_map = candidate.get("capability_map", {})
    role_needs = role.get("capability_needs", {})
    for cap, need in role_needs.items():
        if need in ["high", "critical"]:
            c_cap = cap_map.get(cap, {})
            if not c_cap or c_cap.get("level") in ["none", "low"]:
                risks.append(f"Gap in critical capability area: {cap} is required but candidate level is '{c_cap.get('level', 'none')}'.")
                technical_questions.append(f"Explain your hands-on experience and architecture decisions regarding {cap.replace('_', ' ').title()}.")

    # 3. Overall Risk Level
    if len(risks) >= 3:
        risk_level = "high"
        mitigation = "Perform a deep-dive technical assessment and thorough reference checks to verify tenure and skill levels."
    elif len(risks) >= 1:
        risk_level = "medium"
        mitigation = "Validate capability gaps during interview probes and verify reason for shorter stints."
    else:
        risk_level = "low"
        mitigation = "Standard screening and onboarding checks."
        
    # Target standard questions if list is empty
    if not technical_questions:
        technical_questions.append("Can you explain how you designed and deployed the core retrieval logic in your vector search system?")
        technical_questions.append("What caching and indexing optimization steps do you take when managing high-traffic databases?")
        
    if has_short_stint:
        gap_questions.append("You spent a relatively short time at your previous role. What prompted that transition?")
    else:
        gap_questions.append("What are the core parameters you look for in a team and company context when planning your long-term growth?")
        
    behavioral_questions.append("Describe a project where you took end-to-end technical ownership. How did you handle conflicts and delivery timelines?")
    behavioral_questions.append("Tell me about a time when you made a wrong technical decision. How did you diagnose and remediate it?")
    
    return {
        "devils_advocate": {
            "risks": risks if risks else ["No significant anomalies detected in core timeline or capability map."],
            "risk_level": risk_level,
            "mitigation": mitigation
        },
        "interview_questions": {
            "technical": technical_questions,
            "gap_probing": gap_questions,
            "behavioral": behavioral_questions
        }
    }

# Self-Test block
if __name__ == "__main__":
    print("[Devil's Advocate] Running self-test...")
    
    mock_role = {
        "role_title": "Senior AI Engineer",
        "capability_needs": {
            "backend_engineering": "critical",
            "ml_engineering": "critical"
        }
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "timeline": [
            {"year_start": 2022, "year_end": 2023, "company": "QuickCorp", "title": "SDE2"}
        ],
        "capability_map": {
            "backend_engineering": {"level": "high"}
        }
    }
    
    res = analyze_risks(mock_role, mock_candidate)
    print(f"Result: {json.dumps(res, indent=2)}")
    
    assert "devils_advocate" in res
    assert "interview_questions" in res
    assert "risks" in res["devils_advocate"]
    assert "technical" in res["interview_questions"]
    
    print("[Devil's Advocate] Self-test passed successfully!")
