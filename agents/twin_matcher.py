import sys
import json
import anthropic
from pathlib import Path

# Add the project root to sys.path so we can run this module directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings
from schemas import ScoreResult, safe_validate

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

TWIN_MATCHER_PROMPT = """
You are a senior recruiter evaluating the alignment between a candidate's digital twin persona and the job role's target persona.

Job Role Target Persona:
{role_persona}

Candidate's Digital Twin Persona:
{candidate_twin}

Evaluate the match quality across the following dimensions:
1. Archetype Alignment (e.g. Systems Builder, Product Specialist, Generalist)
2. Working Environment Fit (e.g. startup_scaleup vs. enterprise)
3. Execution Expectation (e.g. high ownership, structured delivery)
4. Leadership Alignment (e.g. emerging leader, individual contributor)
5. Learning Speed & Tech Ceiling vs. Role Needs

Determine a score (0 to 100) indicating the match quality. Be objective. If there are mismatch signals (e.g. candidate is structured/enterprise-oriented, but the role needs a scrappy, high-ownership startup engineer), deduct scores accordingly.

Return ONLY a valid JSON dictionary matching this exact schema:
{{
  "score": number,  // 0.0 to 100.0 alignment score
  "rationale": ["bullet point 1", "bullet point 2"]  // specific evidence-grounded match points or gaps
}}

Return ONLY valid JSON. No markdown formatting, no explanations, no code fences.
"""

def score_twin_match(role_persona: dict, candidate_twin: dict) -> dict:
    """
    Evaluates the match between the candidate twin and role persona using Claude.
    If the API key is not configured, runs in programmatic simulation mode.
    """
    # 1. Fallback Simulation Mode
    if settings.anthropic_api_key == "mock_key":
        return simulate_twin_match(role_persona, candidate_twin)
        
    print(f"[Twin Matcher] Evaluating match using Claude...")
    prompt = TWIN_MATCHER_PROMPT.format(
        role_persona=json.dumps(role_persona, indent=2),
        candidate_twin=json.dumps(candidate_twin, indent=2)
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
        print(f"[ERROR] Twin Matcher LLM execution failed: {e}. Falling back to simulation.")
        return simulate_twin_match(role_persona, candidate_twin)
        
    # Validate against ScoreResult schema
    validated = safe_validate(raw_result, ScoreResult)
    if validated is None:
        print("[ERROR] Twin Matcher response failed schema validation. Falling back to simulation.")
        return simulate_twin_match(role_persona, candidate_twin)
        
    return validated

def simulate_twin_match(role_persona: dict, candidate_twin: dict) -> dict:
    """
    Programmatic simulation fallback for twin matching.
    """
    score = 75.0
    rationale = []
    
    # 1. Archetype alignment
    c_arch = candidate_twin.get("archetype", "").lower()
    r_arch = role_persona.get("archetype", "").lower()
    if c_arch and r_arch:
        if c_arch == r_arch or (c_arch in r_arch) or (r_arch in c_arch):
            score += 10.0
            rationale.append(f"Strong archetype alignment on '{candidate_twin.get('archetype')}'.")
        else:
            score -= 10.0
            rationale.append(f"Archetype mismatch: Candidate is a '{candidate_twin.get('archetype')}' while role expects '{role_persona.get('archetype')}'.")
            
    # 2. Environment alignment
    c_env = candidate_twin.get("environment_fit", "").lower()
    r_env = role_persona.get("environment", "").lower()
    if c_env and r_env:
        if ("startup" in c_env and "startup" in r_env) or ("enterprise" in c_env and "enterprise" in r_env):
            score += 5.0
            rationale.append(f"Environment fit aligns with targeted working context ('{candidate_twin.get('environment_fit')}').")
        elif "startup" in r_env and "enterprise" in c_env:
            score -= 15.0
            rationale.append(f"Environment risk: Candidate profile is enterprise-oriented, which may conflict with startup scale needs.")
            
    # 3. Execution expectations
    c_exec = candidate_twin.get("execution_style", "").lower()
    r_exec = role_persona.get("execution_expectation", "").lower()
    if c_exec and r_exec:
        if "ownership" in c_exec or "autonomous" in c_exec:
            score += 5.0
            rationale.append("Candidate demonstrates high autonomy/ownership style matching role expectation.")
            
    # 4. Learning and growth potential
    learning_speed = candidate_twin.get("learning_speed", "").lower()
    if learning_speed == "high":
        score += 5.0
        rationale.append("High learning velocity supports rapid onboarding on domain skills.")
        
    score = max(0.0, min(100.0, score))
    
    if not rationale:
        rationale.append("Candidate digital twin has moderate situational alignment with role expectations.")
        
    return {
        "score": score,
        "rationale": rationale
    }

# Self-Test block
if __name__ == "__main__":
    print("[Twin Matcher] Running self-test...")
    
    mock_role = {
        "archetype": "Systems Builder",
        "environment": "startup_scaleup",
        "execution_expectation": "high_ownership",
        "leadership_expected": "emerging",
        "learning_required": "high"
    }
    
    mock_candidate_twin = {
        "archetype": "Systems Builder",
        "risk_profile": "low",
        "learning_speed": "high",
        "execution_style": "high_ownership",
        "environment_fit": "startup_to_scaleup",
        "leadership": "emerging",
        "ceiling": "staff_engineer"
    }
    
    res = score_twin_match(mock_role, mock_candidate_twin)
    print(f"Result: {json.dumps(res, indent=2)}")
    assert "score" in res
    assert "rationale" in res
    assert isinstance(res["score"], (int, float))
    assert 0 <= res["score"] <= 100
    print("[Twin Matcher] Self-test passed successfully!")
