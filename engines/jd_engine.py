import anthropic
import json
from config import settings
from schemas import RoleObject, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

JD_DECOMPOSITION_PROMPT = """
You are an expert technical recruiter with 15 years of experience.
Analyze this job description and extract the REAL requirements — 
not just what's written, but what's actually needed.

Job Description:
{jd_text}

Return ONLY valid JSON matching this exact schema:
{{
  "jd_id": "string (the input ID or a generated unique ID)",
  "role_title": "string",
  "seniority_band": "junior|mid|senior|staff|principal",
  "domain": "string",
  "hard_requirements": ["list of non-negotiable requirements"],
  "soft_requirements": ["list of preferred requirements"],
  "implied_needs": ["list of requirements NOT written but clearly needed given role/company context"],
  "culture_signals": ["list of work environment/culture indicators extracted from language"],
  "capability_needs": {{
    "backend_engineering": "none|low|medium|high|critical",
    "cloud_infrastructure": "none|low|medium|high|critical",
    "distributed_systems": "none|low|medium|high|critical",
    "databases": "none|low|medium|high|critical",
    "ml_engineering": "none|low|medium|high|critical",
    "frontend_engineering": "none|low|medium|high|critical",
    "data_engineering": "none|low|medium|high|critical",
    "leadership": "none|low|medium|high|critical"
  }},
  "role_persona": {{
    "archetype": "string (e.g. Systems Builder, Product Engineer, ML Researcher)",
    "environment": "startup|scaleup|enterprise|research",
    "execution_expectation": "high_ownership|collaborative|directed",
    "leadership_expected": "none|emerging|moderate|strong",
    "learning_required": "low|medium|high"
  }},
  "contradictions_detected": ["list any contradictions found, empty array if none"],
  "inflation_adjusted": true,
  "inflation_notes": ["list any inflated requirements that seem unrealistic, empty if none"],
  "disqualifiers": ["list any negative signals or disqualifier patterns extracted from requirements, e.g., consulting_only_career, cv_speech_robotics_only, no_production_deployment, title_chaser_pattern"]
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

def decompose_jd(jd_id: str, jd_text: str) -> dict:
    """
    Decomposes a raw job description string into a structured RoleObject dictionary.
    """
    print(f"[JD Engine] Decomposing: {jd_id}")
    prompt = JD_DECOMPOSITION_PROMPT.format(jd_text=jd_text)
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    # Ensure jd_id is populated from parameters if missing or incorrect
    if isinstance(raw_result, dict):
        raw_result["jd_id"] = jd_id
    else:
        raw_result = {"jd_id": jd_id}

    # Validate LLM output against the RoleObject schema
    validated = safe_validate(raw_result, RoleObject)
    if validated is None:
        print(f"[ERROR] Decomposed JD failed schema validation for ID: {jd_id}")
        return {}
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[JD Engine] Self-Test Running...")
    
    sample_jd = """
    We are looking for a Senior Backend Engineer to join our fast-growing Series A FinTech startup.
    You will own the payment integration systems, scale database throughput, and help design
    our microservices architecture.
    Requirements:
    - 5+ years building backend systems using Python
    - Strong SQL and database optimization skills (PostgreSQL)
    - Experience deploying to AWS using Terraform
    """
    
    # Running test. If API key is invalid/missing, it will log an error rather than crash the test.
    if settings.anthropic_api_key == "mock_key":
        print("[JD Engine] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = decompose_jd("test_jd_001", sample_jd)
            if result:
                print(f"[JD Engine] Decomposed successfully: {json.dumps(result, indent=2)}")
                assert result["role_title"] != ""
                assert "jd_id" in result
                assert "capability_needs" in result
                print("[JD Engine] Self-Test Passed!")
            else:
                print("[JD Engine] Self-Test Failed: Decomposed output was empty.")
        except Exception as err:
            print(f"[JD Engine] Self-Test raised unexpected error: {err}")
