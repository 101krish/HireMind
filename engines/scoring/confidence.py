import anthropic
import json
from config import settings
from schemas import ScoreResult, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

CONFIDENCE_PROMPT = """
You are a senior technical diligence assessor verifying the credibility and depth of evidence in a candidate profile.

Job Requirements:
{role_info}

Candidate's Mapped Capabilities & Timeline Evidence:
{candidate_evidence}

Evaluate:
1. Validity & Depth of evidence: Tier 1 (prod at scale) is high, Tier 5-6 (academic/listed only) is low.
2. Skill validation: are their claims backed by actual projects, detailed descriptions, or certifications?
3. Red flags / gaps: vague project descriptions, title-to-skill discrepancies, or lack of context on claims.
4. Consistency: does the work timeline align with the project stack and descriptions?

Rules/constraints:
- Every score must be based on specific evidence from the profile provided. Do not infer information not present.
- If information is insufficient to score a dimension, return 50 with confidence: 'low'.

Return ONLY valid JSON matching this exact schema:
{{
  "score": number,  // 0-100 score representing evidence confidence
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
    Scores the confidence in the candidate's claims based on profile details.
    Returns a dictionary containing 'score' (0-100) and 'rationale' list.
    """
    candidate_name = candidate.get("name", "Unknown Candidate")
    print(f"[Scoring] Signal 4 - Evidence Confidence: Assessing {candidate_name}")
    
    role_info = {
        "title": role.get("role_title", ""),
        "hard_requirements": role.get("hard_requirements", []),
        "domain": role.get("domain", "")
    }
    
    candidate_evidence = {
        "capability_map": candidate.get("capability_map", {}),
        "timeline": candidate.get("timeline", []),
        "certifications": candidate.get("original_profile", {}).get("certifications", []),
        "projects": candidate.get("original_profile", {}).get("projects", [])
    }
    
    prompt = CONFIDENCE_PROMPT.format(
        role_info=json.dumps(role_info, indent=2),
        candidate_evidence=json.dumps(candidate_evidence, indent=2)
    )
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Provide fallback if LLM call fails
    if not raw_result:
        raw_result = {
            "score": 50.0,
            "rationale": ["Insufficient information to assess evidence confidence due to LLM error."]
        }
        
    # Validate against ScoreResult schema
    validated = safe_validate(raw_result, ScoreResult)
    if validated is None:
        print(f"[ERROR] Confidence score failed validation for candidate: {candidate_name}")
        return {
            "score": 50.0,
            "rationale": ["Failed to validate confidence score structure."]
        }
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Scoring - Evidence Confidence] Self-Test Running...")
    
    mock_role = {
        "role_title": "Senior Software Engineer",
        "hard_requirements": ["Scale distributed systems"]
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "capability_map": {
            "backend_engineering": {"level": "high", "evidence_tier": 1, "evidence": "Built distributed payments at scale", "confidence": 0.95},
            "cloud_infrastructure": {"level": "high", "evidence_tier": 2, "evidence": "AWS production deployment", "confidence": 0.90}
        },
        "timeline": [
            {
                "company": "FinStartup",
                "title": "Senior Engineer",
                "complexity_level": 8,
                "ownership_signals": ["Built"]
            }
        ],
        "original_profile": {
            "certifications": ["AWS Solutions Architect"],
            "projects": [
                {"name": "Payment Gateway", "description": "High throughput system"}
            ]
        }
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Scoring - Evidence Confidence] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = score_candidate(mock_role, mock_candidate)
            print(f"[Scoring - Evidence Confidence] Output: {json.dumps(result, indent=2)}")
            assert "score" in result
            assert "rationale" in result
            assert 0 <= result["score"] <= 100
            print("[Scoring - Evidence Confidence] Self-Test Passed!")
        except Exception as err:
            print(f"[Scoring - Evidence Confidence] Self-Test raised unexpected error: {err}")
