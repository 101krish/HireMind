import anthropic
import json
from config import settings
from schemas import ScoreResult, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

TRAJECTORY_PROMPT = """
You are a senior technical recruiter evaluating a candidate's career trajectory, momentum, and growth velocity.

Job Description Role Title & Seniority:
{role_info}

Candidate's Reconstructed Career Timeline & Twin Arc:
{candidate_timeline}

Evaluate:
1. Promotion velocity: how quickly have they grown into senior roles (e.g. SDE 1 -> SDE 2 -> Senior)?
2. Tenure stability: check for short-term hops vs. sustained growth at companies.
3. Title progression and role complexity over time.
4. Momentum: is their career accelerating, steady, transitioning, or decelerating?

Rules/constraints:
- Every score must be based on specific evidence from the profile provided. Do not infer information not present.
- If information is insufficient to score a dimension, return 50 with confidence: 'low'.

Return ONLY valid JSON matching this exact schema:
{{
  "score": number,  // 0-100 score indicating trajectory momentum
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
    Scores the candidate's career trajectory and growth momentum.
    Returns a dictionary containing 'score' (0-100) and 'rationale' list.
    """
    candidate_name = candidate.get("name", "Unknown Candidate")
    print(f"[Scoring] Signal 2 - Trajectory: Assessing {candidate_name}")
    
    role_info = {
        "title": role.get("role_title", ""),
        "seniority_band": role.get("seniority_band", ""),
        "domain": role.get("domain", "")
    }
    
    candidate_timeline = {
        "timeline": candidate.get("timeline", []),
        "narrative_arc": candidate.get("narrative_arc", ""),
        "candidate_twin": candidate.get("candidate_twin", {})
    }
    
    prompt = TRAJECTORY_PROMPT.format(
        role_info=json.dumps(role_info, indent=2),
        candidate_timeline=json.dumps(candidate_timeline, indent=2)
    )
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Provide fallback if LLM call fails
    if not raw_result:
        raw_result = {
            "score": 50.0,
            "rationale": ["Insufficient information to assess career trajectory due to LLM error."]
        }
        
    # Validate against ScoreResult schema
    validated = safe_validate(raw_result, ScoreResult)
    if validated is None:
        print(f"[ERROR] Trajectory score failed validation for candidate: {candidate_name}")
        return {
            "score": 50.0,
            "rationale": ["Failed to validate trajectory score structure."]
        }
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Scoring - Trajectory] Self-Test Running...")
    
    mock_role = {
        "role_title": "Senior Software Engineer",
        "seniority_band": "senior",
        "domain": "Fintech"
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "narrative_arc": "accelerator",
        "candidate_twin": {
            "archetype": "Systems Builder",
            "learning_speed": "high",
            "ceiling": "staff_engineer"
        },
        "timeline": [
            {"year_start": 2019, "year_end": 2021, "company": "TechCorp", "title": "SDE1", "promoted": False},
            {"year_start": 2021, "year_end": 2023, "company": "TechCorp", "title": "SDE2", "promoted": True, "months_to_promote": 14},
            {"year_start": 2023, "year_end": None, "company": "FinStartup", "title": "Senior Engineer", "promoted": True}
        ]
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Scoring - Trajectory] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = score_candidate(mock_role, mock_candidate)
            print(f"[Scoring - Trajectory] Output: {json.dumps(result, indent=2)}")
            assert "score" in result
            assert "rationale" in result
            assert 0 <= result["score"] <= 100
            print("[Scoring - Trajectory] Self-Test Passed!")
        except Exception as err:
            print(f"[Scoring - Trajectory] Self-Test raised unexpected error: {err}")
