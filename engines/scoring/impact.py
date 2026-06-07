import anthropic
import json
from config import settings
from schemas import ScoreResult, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

IMPACT_PROMPT = """
You are a senior technical hiring manager evaluating a candidate's engineering impact and execution depth.

Job Description Role requirements:
{role_info}

Candidate's Achievement Signals, Ownership Language Score, and Timeline:
{candidate_impact_info}

Evaluate:
1. Achievement density: the volume and frequency of high-impact outcomes.
2. Scale metrics: evidence of working with scale (e.g. transactions per day, latency reductions, user counts, data sizes).
3. Complexity level of the systems or solutions they built.
4. Ownership language: does the timeline reflect a "doer" who owns products/features or someone who just participates? (Ownership Language Score: {ownership_score}).

Rules/constraints:
- Every score must be based on specific evidence from the profile provided. Do not infer information not present.
- If information is insufficient to score a dimension, return 50 with confidence: 'low'.

Return ONLY valid JSON matching this exact schema:
{{
  "score": number,  // 0-100 score indicating engineering impact
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
    Scores the candidate's engineering impact and scale metrics.
    Returns a dictionary containing 'score' (0-100) and 'rationale' list.
    """
    candidate_name = candidate.get("name", "Unknown Candidate")
    print(f"[Scoring] Signal 3 - Impact: Assessing {candidate_name}")
    
    role_info = {
        "title": role.get("role_title", ""),
        "hard_requirements": role.get("hard_requirements", []),
        "domain": role.get("domain", "")
    }
    
    ownership_score = candidate.get("ownership_language_score", 0.5)
    
    candidate_impact_info = {
        "timeline": candidate.get("timeline", []),
        "achievement_signals": candidate.get("achievement_signals", []),
        "ownership_language_score": ownership_score
    }
    
    prompt = IMPACT_PROMPT.format(
        role_info=json.dumps(role_info, indent=2),
        candidate_impact_info=json.dumps(candidate_impact_info, indent=2),
        ownership_score=ownership_score
    )
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Provide fallback if LLM call fails
    if not raw_result:
        raw_result = {
            "score": 50.0,
            "rationale": ["Insufficient information to assess engineering impact due to LLM error."]
        }
        
    # Validate against ScoreResult schema
    validated = safe_validate(raw_result, ScoreResult)
    if validated is None:
        print(f"[ERROR] Impact score failed validation for candidate: {candidate_name}")
        return {
            "score": 50.0,
            "rationale": ["Failed to validate impact score structure."]
        }
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Scoring - Impact] Self-Test Running...")
    
    mock_role = {
        "role_title": "Senior Software Engineer",
        "hard_requirements": ["Scale distributed systems"]
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "ownership_language_score": 0.82,
        "timeline": [
            {
                "company": "FinStartup",
                "title": "Senior Engineer",
                "complexity_level": 8,
                "ownership_signals": ["Built", "Led"],
                "metrics": ["Reduced P99 latency by 40%", "2M tx/day"]
            }
        ],
        "achievement_signals": [
            {"text": "Reduced P99 latency by 40%", "type": "performance_metric", "strength": "strong"},
            {"text": "2M transactions per day", "type": "scale_metric", "strength": "strong"}
        ]
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Scoring - Impact] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = score_candidate(mock_role, mock_candidate)
            print(f"[Scoring - Impact] Output: {json.dumps(result, indent=2)}")
            assert "score" in result
            assert "rationale" in result
            assert 0 <= result["score"] <= 100
            print("[Scoring - Impact] Self-Test Passed!")
        except Exception as err:
            print(f"[Scoring - Impact] Self-Test raised unexpected error: {err}")
