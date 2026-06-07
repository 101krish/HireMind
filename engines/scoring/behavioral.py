import anthropic
import json
from config import settings
from schemas import ScoreResult, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

BEHAVIORAL_PROMPT = """
You are an expert executive talent assessor evaluating a candidate's behavioral fit, environment alignment, and execution style for a role.

Job Role Persona & Culture Signals:
{role_persona}

Candidate's Twin Characteristics, Timeline Leadership & Communication Evidence:
{candidate_behavioral_info}

Evaluate:
1. Persona alignment: match of candidate archetype (e.g., Systems Builder) with expected role archetype.
2. Execution style fit: e.g., startup high ownership vs. enterprise directed execution.
3. Environment fit: how well does the candidate's career environment history (startup, scaleup, enterprise) align with the role environment?
4. Leadership and Communication signals.

Rules/constraints:
- Every score must be based on specific evidence from the profile provided. Do not infer information not present.
- If information is insufficient to score a dimension, return 50 with confidence: 'low'.

Return ONLY valid JSON matching this exact schema:
{{
  "score": number,  // 0-100 score representing behavioral/team fit
  "rationale": ["bullet point 1", "bullet point 2"]  // specific evidence-grounded reasons
}}

Return ONLY valid JSON. No markdown, no explanation, no code fences. JSON only.
"""

def call_llm(prompt: str, model: str, max_tokens: int) -> dict:
    """
    Calls the Anthropic Claude LLM and handles JSON parsing securely.
    """
    raw = ""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.removeprefix("```json").removeprefix("```")
        raw = raw.removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[ERROR] LLM JSON parse error: {e}\nRaw: {raw[:200]}")
        return {}
    except Exception as e:
        print(f"[ERROR] LLM call error: {e}")
        return {}

def score_candidate(role: dict, candidate: dict) -> dict:
    """
    Scores the candidate's behavioral alignment and environment fit.
    Returns a dictionary containing 'score' (0-100) and 'rationale' list.
    """
    candidate_name = candidate.get("name", "Unknown Candidate")
    print(f"[Scoring] Signal 6 - Behavioral & Team Fit: Assessing {candidate_name}")
    
    role_persona = {
        "role_title": role.get("role_title", ""),
        "culture_signals": role.get("culture_signals", []),
        "role_persona": role.get("role_persona", {})
    }
    
    candidate_behavioral_info = {
        "candidate_twin": candidate.get("candidate_twin", {}),
        "leadership_signals": candidate.get("leadership_signals", []),
        "communication_signals": candidate.get("communication_signals", []),
        "timeline_companies": [
            {"company": t.get("company", ""), "size": t.get("company_size", "")}
            for t in candidate.get("timeline", []) if t.get("company")
        ]
    }
    
    prompt = BEHAVIORAL_PROMPT.format(
        role_persona=json.dumps(role_persona, indent=2),
        candidate_behavioral_info=json.dumps(candidate_behavioral_info, indent=2)
    )
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Provide fallback if LLM call fails
    if not raw_result:
        raw_result = {
            "score": 50.0,
            "rationale": ["Insufficient information to assess behavioral fit due to LLM error."]
        }
        
    # Validate against ScoreResult schema
    validated = safe_validate(raw_result, ScoreResult)
    if validated is None:
        print(f"[ERROR] Behavioral score failed validation for candidate: {candidate_name}")
        return {
            "score": 50.0,
            "rationale": ["Failed to validate behavioral score structure."]
        }
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Scoring - Behavioral] Self-Test Running...")
    
    mock_role = {
        "role_title": "Senior Software Engineer",
        "culture_signals": ["High ownership", "Fast-paced startup"],
        "role_persona": {
            "archetype": "Systems Builder",
            "environment": "startup_scaleup",
            "execution_expectation": "high_ownership"
        }
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "candidate_twin": {
            "archetype": "Systems Builder",
            "execution_style": "high_ownership",
            "environment_fit": "startup_to_scaleup"
        },
        "leadership_signals": ["Led team of 3 engineers"],
        "communication_signals": ["Presented tech architecture to leadership"],
        "timeline": [
            {"company": "FinStartup", "company_size": "series_b"}
        ]
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Scoring - Behavioral] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = score_candidate(mock_role, mock_candidate)
            print(f"[Scoring - Behavioral] Output: {json.dumps(result, indent=2)}")
            assert "score" in result
            assert "rationale" in result
            assert 0 <= result["score"] <= 100
            print("[Scoring - Behavioral] Self-Test Passed!")
        except Exception as err:
            print(f"[Scoring - Behavioral] Self-Test raised unexpected error: {err}")
