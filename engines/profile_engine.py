import anthropic
import json
from config import settings
from schemas import StructuredCandidate, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

PROFILE_DECOMPOSITION_PROMPT = """
You are an expert technical recruiter. Analyze this candidate profile deeply.
Extract explicit signals, implicit signals, and reconstruct their career timeline.

Candidate Profile:
{profile_json}

Return ONLY valid JSON matching this exact schema:
{{
  "candidate_id": "string",
  "name": "string",
  "timeline": [
    {{
      "year_start": number or null,
      "year_end": number or null,
      "company": "string or null",
      "title": "string or null",
      "company_size": "startup|series_a|series_b|enterprise|unknown or null",
      "domain": "string or null",
      "skills_demonstrated": ["list of skills"],
      "complexity_level": number (1-10),
      "ownership_signals": ["action verbs used: e.g., Built, Led, Designed"],
      "metrics": ["any quantified achievements"],
      "promoted": boolean,
      "months_to_promote": number or null,
      "year": number or null,
      "event": "string or null (e.g. graduation, certification)",
      "skills_gained": ["list of skills gained"]
    }}
  ],
  "achievement_signals": [
    {{
      "text": "exact achievement text",
      "type": "performance_metric|scale_metric|business_impact|leadership",
      "strength": "weak|medium|strong"
    }}
  ],
  "ownership_language_score": number (0.0-1.0),
  "leadership_signals": ["list of leadership evidence"],
  "communication_signals": ["list of communication evidence"],
  "narrative_arc": "specialist|generalist_builder|transition|accelerator|late_bloomer|steady",
  "candidate_twin": {{
    "archetype": "string (e.g., Systems Builder, Product Generalist, Infrastructure Specialist)",
    "risk_profile": "low|medium|high",
    "learning_speed": "low|medium|high",
    "execution_style": "high_ownership|collaborative|directed",
    "environment_fit": "startup|scaleup|enterprise|any",
    "leadership": "none|emerging|moderate|strong",
    "ceiling": "senior|staff|principal|vp_engineering"
  }}
}}

If information is insufficient for any field, use reasonable default values or empty arrays/null.
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

def decompose_profile(profile: dict) -> dict:
    """
    Decomposes a raw candidate profile dictionary into a structured StructuredCandidate dictionary.
    """
    candidate_name = profile.get("name", "Unknown Candidate")
    candidate_id = profile.get("id", "unknown_id")
    print(f"[Profile Engine] Parsing: {candidate_name}")
    
    prompt = PROFILE_DECOMPOSITION_PROMPT.format(profile_json=json.dumps(profile, indent=2))
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Ensure candidate_id and name are populated correctly
    if isinstance(raw_result, dict):
        raw_result["candidate_id"] = candidate_id
        if "name" not in raw_result or not raw_result["name"]:
            raw_result["name"] = candidate_name
        # Ensure capability_map is empty initially as it will be filled by capability_engine
        if "capability_map" not in raw_result:
            raw_result["capability_map"] = {}
    else:
        raw_result = {
            "candidate_id": candidate_id,
            "name": candidate_name,
            "timeline": [],
            "achievement_signals": [],
            "ownership_language_score": 0.5,
            "leadership_signals": [],
            "communication_signals": [],
            "narrative_arc": "generalist_builder",
            "candidate_twin": {
                "archetype": "General SDE",
                "risk_profile": "low",
                "learning_speed": "medium",
                "execution_style": "collaborative",
                "environment_fit": "any",
                "leadership": "none",
                "ceiling": "senior"
            },
            "capability_map": {}
        }
        
    # Validate against StructuredCandidate schema
    validated = safe_validate(raw_result, StructuredCandidate)
    if validated is None:
        print(f"[ERROR] Decomposed profile failed schema validation for candidate: {candidate_name}")
        return {}
        
    # Inject original raw profile under a separate key if downstream logic needs it
    validated["original_profile"] = profile
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Profile Engine] Self-Test Running...")
    
    sample_profile = {
        "id": "candidate_001",
        "name": "Priya Sharma",
        "current_title": "Senior Software Engineer",
        "total_experience_years": 5,
        "education": [
            {
                "degree": "B.Tech Computer Science",
                "institution": "IIT Bombay",
                "year": 2019
            }
        ],
        "work_history": [
            {
                "company": "FinStartup",
                "title": "Senior Engineer",
                "start_year": 2023,
                "end_year": None,
                "is_current": True,
                "description": "Built distributed payment processing system handling 2M tx/day. Reduced P99 latency by 40%. Led team of 3 engineers.",
                "company_size": "series_b",
                "domain": "fintech"
            }
        ],
        "skills": ["Python", "AWS", "Kafka", "Kubernetes", "PostgreSQL"],
        "projects": [
            {
                "name": "Payment Processing System",
                "description": "Real-time payment system with 99.9% uptime",
                "tech_stack": ["Python", "Kafka", "AWS", "PostgreSQL"],
                "metrics": ["2M tx/day", "99.9% uptime", "40% latency reduction"]
            }
        ],
        "certifications": ["AWS Solutions Architect"],
        "platforms": {
            "github": "github.com/priya",
            "blog": "medium.com/@priya"
        }
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Profile Engine] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = decompose_profile(sample_profile)
            if result:
                print(f"[Profile Engine] Structured Profile: {json.dumps(result, indent=2)}")
                assert result["name"] == "Priya Sharma"
                assert "candidate_twin" in result
                assert "timeline" in result
                print("[Profile Engine] Self-Test Passed!")
            else:
                print("[Profile Engine] Self-Test Failed: Decomposed output was empty.")
        except Exception as err:
            print(f"[Profile Engine] Self-Test raised unexpected error: {err}")
