import anthropic
import json
from config import settings
from schemas import ScoreResult, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

HIDDEN_TALENT_PROMPT = """
You are a senior technical talent partner specialized in identifying hidden talents, transferable skills, and title-to-role mismatches.

Job Description requirements:
{role_info}

Candidate's Mapped Capabilities, Hidden Capabilities (from assessment), and Timeline:
{candidate_info}

Evaluate:
1. Mismatches: instances where their title (e.g., SDE 2) is lower than the actual complexity of work they own (e.g., architecting full distributed systems).
2. Transferable skills: cross-domain capabilities that are highly relevant to this role (e.g., data pipeline expertise for a backend role).
3. Hidden capabilities: skills or capabilities shown in projects or work history that are NOT explicitly listed as their main skills or title.
4. Assess the strength and value of these hidden talents to the target role.

Rules/constraints:
- Every score must be based on specific evidence from the profile provided. Do not infer information not present.
- If information is insufficient to score a dimension (e.g., no hidden talents are detected), return a base score of 50 or 70 with an explanation that no significant title mismatches or hidden talents were found.

Return ONLY valid JSON matching this exact schema:
{{
  "score": number,  // 0-100 score representing the hidden talent index
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
    Scores the candidate's hidden or transferable talents relevant to the role.
    Returns a dictionary containing 'score' (0-100) and 'rationale' list.
    """
    candidate_name = candidate.get("name", "Unknown Candidate")
    print(f"[Scoring] Signal 5 - Hidden Talent: Assessing {candidate_name}")
    
    role_info = {
        "title": role.get("role_title", ""),
        "domain": role.get("domain", ""),
        "capability_needs": role.get("capability_needs", {})
    }
    
    candidate_info = {
        "capability_map": candidate.get("capability_map", {}),
        "hidden_capabilities": candidate.get("hidden_capabilities", []),
        "timeline": candidate.get("timeline", []),
        "skills": candidate.get("original_profile", {}).get("skills", [])
    }
    
    prompt = HIDDEN_TALENT_PROMPT.format(
        role_info=json.dumps(role_info, indent=2),
        candidate_info=json.dumps(candidate_info, indent=2)
    )
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Provide fallback if LLM call fails
    if not raw_result:
        raw_result = {
            "score": 50.0,
            "rationale": ["Insufficient information to assess hidden talents due to LLM error."]
        }
        
    # Validate against ScoreResult schema
    validated = safe_validate(raw_result, ScoreResult)
    if validated is None:
        print(f"[ERROR] Hidden talent score failed validation for candidate: {candidate_name}")
        return {
            "score": 50.0,
            "rationale": ["Failed to validate hidden talent score structure."]
        }
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Scoring - Hidden Talent] Self-Test Running...")
    
    mock_role = {
        "role_title": "Senior Software Engineer",
        "domain": "Fintech",
        "capability_needs": {"backend_engineering": "critical", "ml_engineering": "medium"}
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "capability_map": {
            "backend_engineering": {"level": "high", "evidence_tier": 1, "evidence": "Built payments at scale", "confidence": 0.95}
        },
        "hidden_capabilities": [
            {
                "capability": "ml_engineering",
                "evidence": "Built data pipelines and ML models for fraud detection",
                "title_mismatch": True,
                "note": "Has ML capability despite standard SDE title"
            }
        ],
        "timeline": [
            {"company": "FinStartup", "title": "Senior Engineer"}
        ]
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Scoring - Hidden Talent] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = score_candidate(mock_role, mock_candidate)
            print(f"[Scoring - Hidden Talent] Output: {json.dumps(result, indent=2)}")
            assert "score" in result
            assert "rationale" in result
            assert 0 <= result["score"] <= 100
            print("[Scoring - Hidden Talent] Self-Test Passed!")
        except Exception as err:
            print(f"[Scoring - Hidden Talent] Self-Test raised unexpected error: {err}")
