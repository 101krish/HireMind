import sys
import json
import anthropic
from pathlib import Path

# Add project root to sys.path so we can run this module directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings
from schemas import DebateScores, safe_validate

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

DEBATE_PANEL_PROMPT = """
You are simulating a recruiter hiring panel debating a candidate's alignment with a role.
The panel consists of three distinct personas:
1. Technical Recruiter: Emphasizes concrete skills, domain expertise, project scale, and deployment.
2. Hiring Manager: Emphasizes team fit, execution style, archetype alignment, and practical delivery.
3. Growth/Talent Partner: Emphasizes trajectory velocity, leadership potential, ceiling, and learning speed.

Job Description & Requirements:
{role_requirements}

Candidate's Decomposed Profile & Capability Mapping:
{candidate_profile}

Please simulate their debate regarding the candidate. Each panelist will give their individual score (0 to 100), and together they will align on a final consensus score (0 to 100).
Provide 3 key debate takeaways or bullet points summarizing the consensus or key concerns discussed by the panelists.

Return ONLY a valid JSON dictionary matching this exact schema:
{{
  "technical_agent": number,  // Technical Recruiter score (0.0 to 100.0)
  "manager_agent": number,    // Hiring Manager score (0.0 to 100.0)
  "growth_agent": number,     // Growth Partner score (0.0 to 100.0)
  "consensus": number,        // Final resolved consensus score (0.0 to 100.0)
  "rationale": ["bullet point 1", "bullet point 2", "bullet point 3"]  // key discussion takeaways
}}

Return ONLY valid JSON. No markdown formatting, no explanations, no code fences.
"""

def run_debate(role: dict, candidate: dict) -> dict:
    """
    Simulates a debate panel for a candidate.
    If the API key is not configured, runs in programmatic simulation mode.
    """
    if settings.anthropic_api_key == "mock_key":
        return simulate_debate(role, candidate)
        
    print(f"[Debate Panel] Simulating panel debate for {candidate.get('name', 'Unknown')}...")
    
    prompt = DEBATE_PANEL_PROMPT.format(
        role_requirements=json.dumps(role, indent=2),
        candidate_profile=json.dumps(candidate, indent=2)
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
        print(f"[ERROR] Debate Panel LLM execution failed: {e}. Falling back to simulation.")
        return simulate_debate(role, candidate)
        
    # Validate scores matching the schemas
    scores_dict = {
        "technical_agent": raw_result.get("technical_agent", 50.0),
        "manager_agent": raw_result.get("manager_agent", 50.0),
        "growth_agent": raw_result.get("growth_agent", 50.0),
        "consensus": raw_result.get("consensus", 50.0)
    }
    validated_scores = safe_validate(scores_dict, DebateScores)
    if validated_scores is None:
        print("[ERROR] Debate scores failed validation. Falling back to simulation.")
        return simulate_debate(role, candidate)
        
    return {
        "debate_scores": validated_scores,
        "rationale": raw_result.get("rationale", ["Takeaway resolved by the panel."])
    }

def simulate_debate(role: dict, candidate: dict) -> dict:
    """
    Programmatic simulation fallback for the recruiter debate panel.
    """
    # 1. Tech score evaluation
    tech_score = 65.0
    skills = [s.get("name", "").lower() for s in candidate.get("original_profile", {}).get("skills", [])]
    if not skills:
        # Check from capability map
        skills = list(candidate.get("capability_map", {}).keys())
        
    critical_skills = ["python", "embedding", "vector", "rag", "retrieval", "ml", "search"]
    matches = sum(1 for cs in critical_skills if any(cs in sk for sk in skills))
    tech_score += min(30.0, matches * 7.5)
    
    # 2. Manager score evaluation
    manager_score = 70.0
    c_twin = candidate.get("candidate_twin", {})
    r_persona = role.get("role_persona", {})
    if c_twin.get("archetype") == r_persona.get("archetype"):
        manager_score += 15.0
    if c_twin.get("execution_style") == "high_ownership":
        manager_score += 5.0
        
    # 3. Growth score evaluation
    growth_score = 60.0
    # Assess via timeline/experience length
    timeline = candidate.get("timeline", [])
    promoted = any(t.get("promoted", False) for t in timeline)
    if promoted:
        growth_score += 15.0
    if c_twin.get("learning_speed") == "high":
        growth_score += 10.0
        
    # Clamping
    tech_score = max(0.0, min(100.0, tech_score))
    manager_score = max(0.0, min(100.0, manager_score))
    growth_score = max(0.0, min(100.0, growth_score))
    
    # Consensus is a weighted combination
    consensus = (tech_score * 0.4) + (manager_score * 0.4) + (growth_score * 0.2)
    consensus = round(consensus, 2)
    
    rationale = [
        f"Technical Recruiter: Strong background in engineering with key alignment on critical tech stacks.",
        f"Hiring Manager: Executive persona match is positive, demonstrating compatibility with team expectations.",
        f"Growth Partner: Promising upward career velocity and high capability ceilings observed."
    ]
    
    return {
        "debate_scores": {
            "technical_agent": tech_score,
            "manager_agent": manager_score,
            "growth_agent": growth_score,
            "consensus": consensus
        },
        "rationale": rationale
    }

# Self-Test block
if __name__ == "__main__":
    print("[Debate Panel] Running self-test...")
    
    mock_role = {
        "role_title": "Senior AI Engineer",
        "role_persona": {"archetype": "Systems Builder"}
    }
    
    mock_candidate = {
        "name": "Priya Sharma",
        "candidate_twin": {
            "archetype": "Systems Builder",
            "execution_style": "high_ownership",
            "learning_speed": "high"
        },
        "timeline": [
            {"year_start": 2021, "year_end": 2023, "company": "TechCorp", "promoted": True}
        ],
        "capability_map": {
            "backend_engineering": {"level": "high"}
        },
        "original_profile": {
            "skills": [{"name": "Python"}, {"name": "Vector Databases"}]
        }
    }
    
    res = run_debate(mock_role, mock_candidate)
    print(f"Result: {json.dumps(res, indent=2)}")
    
    assert "debate_scores" in res
    assert "rationale" in res
    assert "consensus" in res["debate_scores"]
    assert len(res["rationale"]) > 0
    
    print("[Debate Panel] Self-test passed successfully!")
