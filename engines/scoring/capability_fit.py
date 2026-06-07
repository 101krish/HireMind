import anthropic
import json
from config import settings
from schemas import ScoreResult, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

CAPABILITY_FIT_PROMPT = """
You are a senior technical assessor evaluating a candidate's technical capability fit for a specific role.

Job Role Capability Needs:
{role_needs}

Candidate's Mapped Capabilities:
{candidate_capabilities}

Assess how well the candidate's capabilities match the required capability levels for the role.
Evaluate:
1. The overlaps (where capability matches or exceeds expectations).
2. The gaps (where candidate is deficient in critical/high need areas).
3. The depth of evidence.

Rules/constraints:
- Every score must be based on specific evidence from the profile provided. Do not infer information not present.
- If information is insufficient to score a dimension, return 50 with confidence: 'low'.

Return ONLY valid JSON matching this exact schema:
{{
  "score": number,  // 0-100 score indicating fit
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
    Scores the candidate's capability fit for the role.
    Returns a dictionary containing 'score' (0-100) and 'rationale' list.
    """
    candidate_name = candidate.get("name", "Unknown Candidate")
    print(f"[Scoring] Signal 1 - Capability Fit: Assessing {candidate_name}")
    
    role_needs = role.get("capability_needs", {})
    candidate_capabilities = candidate.get("capability_map", {})
    
    prompt = CAPABILITY_FIT_PROMPT.format(
        role_needs=json.dumps(role_needs, indent=2),
        candidate_capabilities=json.dumps(candidate_capabilities, indent=2)
    )
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Provide fallback if LLM call fails
    if not raw_result:
        raw_result = {
            "score": 50.0,
            "rationale": ["Insufficient information to assess capability fit due to LLM error."]
        }
        
    # Validate against ScoreResult schema
    validated = safe_validate(raw_result, ScoreResult)
    if validated is None:
        print(f"[ERROR] Capability fit score failed validation for candidate: {candidate_name}")
        return {
            "score": 50.0,
            "rationale": ["Failed to validate capability fit score structure."]
        }
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Scoring - Capability Fit] Self-Test Running...")
    
    mock_role = {
        "capability_needs": {
            "backend_engineering": "critical",
            "cloud_infrastructure": "high",
            "distributed_systems": "high"
        }
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "capability_map": {
            "backend_engineering": {"level": "high", "evidence_tier": 1, "evidence": "Built distributed payments at scale", "confidence": 0.95},
            "cloud_infrastructure": {"level": "high", "evidence_tier": 2, "evidence": "AWS production deployment", "confidence": 0.90},
            "distributed_systems": {"level": "high", "evidence_tier": 1, "evidence": "2M tx/day payment system", "confidence": 0.95}
        }
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Scoring - Capability Fit] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = score_candidate(mock_role, mock_candidate)
            print(f"[Scoring - Capability Fit] Output: {json.dumps(result, indent=2)}")
            assert "score" in result
            assert "rationale" in result
            assert 0 <= result["score"] <= 100
            print("[Scoring - Capability Fit] Self-Test Passed!")
        except Exception as err:
            print(f"[Scoring - Capability Fit] Self-Test raised unexpected error: {err}")
