import json
from pathlib import Path
import anthropic
from config import settings
from schemas import StructuredCandidate, safe_validate

# Initialize the Anthropic client using the key from config settings
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

def load_capability_graph() -> dict:
    """
    Loads the capability ontology graph from data/capability_graph.json.
    Resolves the path relative to the file location to prevent FileNotFoundError.
    """
    try:
        path = Path(__file__).parent.parent / "data" / "capability_graph.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] Capability graph file not found: {e}")
        return {"capabilities": {}}
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse capability graph JSON: {e}")
        return {"capabilities": {}}

CAPABILITY_MAPPING_PROMPT = """
You are a technical capability assessor. 
Given this candidate's structured profile, map their capabilities.

Capability Graph (available domains):
{capability_graph}

Candidate Profile:
{structured_profile}

For each capability domain, assess:
1. What evidence exists (explicit mention, implicit from projects, inferred)
2. Evidence tier: 1=prod_at_scale, 2=production, 3=certification, 4=side_project, 5=academic, 6=listed_only
3. Confidence: 0.0-1.0
4. Source: "explicit|implicit|inferred"

Return ONLY valid JSON matching this exact schema:
{{
  "capability_map": {{
    "backend_engineering": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }},
    "cloud_infrastructure": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }},
    "distributed_systems": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }},
    "databases": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }},
    "ml_engineering": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }},
    "frontend_engineering": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }},
    "data_engineering": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }},
    "security_engineering": {{
      "level": "none|low|medium|high|expert",
      "evidence_tier": 1-6,
      "evidence": "specific evidence from profile",
      "confidence": 0.0-1.0,
      "source": "explicit|implicit|inferred"
    }}
  }},
  "hidden_capabilities": [
    {{
      "capability": "domain name",
      "evidence": "why we think they have this",
      "title_mismatch": true/false,
      "note": "explanation"
    }}
  ]
}}

If information is insufficient to score a dimension, return level "none" with evidence tier 6, confidence 0.0, source "inferred", and evidence "No evidence found".
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

def map_capabilities(structured_profile: dict) -> dict:
    """
    Maps the capabilities of a structured profile using the ontology graph.
    Updates the capability_map and hidden_capabilities fields on the structured profile.
    """
    candidate_name = structured_profile.get("name", "Unknown Candidate")
    print(f"[Capability Engine] Mapping capabilities: {candidate_name}")
    
    graph = load_capability_graph()
    
    prompt = CAPABILITY_MAPPING_PROMPT.format(
        capability_graph=json.dumps(graph, indent=2),
        structured_profile=json.dumps(structured_profile, indent=2)
    )
    
    # Execute the LLM call using settings parameters
    raw_result = call_llm(prompt, settings.model_name, settings.max_tokens)
    
    if not isinstance(raw_result, dict):
        raw_result = {}
        
    capability_map = raw_result.get("capability_map", {})
    hidden_capabilities = raw_result.get("hidden_capabilities", [])
    
    # Merge the parsed capability_map back into the structured profile
    structured_profile["capability_map"] = capability_map
    structured_profile["hidden_capabilities"] = hidden_capabilities
    
    # Validate the final merged profile
    validated = safe_validate(structured_profile, StructuredCandidate)
    if validated is None:
        print(f"[ERROR] Structured profile with capabilities failed schema validation for candidate: {candidate_name}")
        return structured_profile
        
    # Retain the hidden_capabilities and original_profile on the output dictionary
    validated["hidden_capabilities"] = hidden_capabilities
    if "original_profile" in structured_profile:
        validated["original_profile"] = structured_profile["original_profile"]
        
    return validated

# Self-Test block
if __name__ == "__main__":
    print("[Capability Engine] Self-Test Running...")
    
    # Mock decomposed profile (matching StructuredCandidate structure)
    mock_structured_profile = {
        "candidate_id": "candidate_001",
        "name": "Priya Sharma",
        "timeline": [
            {
                "year_start": 2023,
                "year_end": None,
                "company": "FinStartup",
                "title": "Senior Engineer",
                "skills_gained": ["Python", "AWS", "Kafka", "PostgreSQL"],
                "complexity_level": 7,
                "ownership_signals": ["Built", "Led"]
            }
        ],
        "achievement_signals": [
            {
                "text": "Reduced P99 latency by 40%",
                "type": "performance_metric",
                "strength": "strong"
            }
        ],
        "ownership_language_score": 0.82,
        "leadership_signals": ["led team of 3 engineers"],
        "communication_signals": [],
        "narrative_arc": "generalist_builder",
        "candidate_twin": {
            "archetype": "Systems Builder",
            "risk_profile": "low",
            "learning_speed": "high",
            "execution_style": "high_ownership",
            "environment_fit": "startup_to_scaleup",
            "leadership": "emerging",
            "ceiling": "staff_engineer"
        },
        "capability_map": {}
    }
    
    if settings.anthropic_api_key == "mock_key":
        print("[Capability Engine] Skipping live LLM call during self-test because ANTHROPIC_API_KEY is not set.")
    else:
        try:
            result = map_capabilities(mock_structured_profile)
            print(f"[Capability Engine] Mapped Profile: {json.dumps(result, indent=2)}")
            assert "capability_map" in result
            assert "backend_engineering" in result["capability_map"]
            assert "hidden_capabilities" in result
            print("[Capability Engine] Self-Test Passed!")
        except Exception as err:
            print(f"[Capability Engine] Self-Test raised unexpected error: {err}")
