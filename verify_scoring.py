import sys
import json
from config import settings
from engines.scoring.capability_fit import score_candidate as score_capability
from engines.scoring.trajectory import score_candidate as score_trajectory
from engines.scoring.impact import score_candidate as score_impact
from engines.scoring.confidence import score_candidate as score_confidence
from engines.scoring.hidden_talent import score_candidate as score_hidden_talent
from engines.scoring.behavioral import score_candidate as score_behavioral

def get_test_data() -> tuple:
    """Returns a mock role and a mock candidate structured profile for testing."""
    role = {
        "role_title": "Senior Python Backend Engineer",
        "seniority_band": "senior",
        "domain": "Fintech",
        "hard_requirements": ["Python", "AWS", "Kafka", "PostgreSQL"],
        "culture_signals": ["High ownership", "Fast-paced startup"],
        "capability_needs": {
            "backend_engineering": "critical",
            "cloud_infrastructure": "high",
            "distributed_systems": "high",
            "databases": "high",
            "frontend_engineering": "none",
            "ml_engineering": "none",
            "data_engineering": "low",
            "leadership": "medium"
        },
        "role_persona": {
            "archetype": "Systems Builder",
            "environment": "startup_scaleup",
            "execution_expectation": "high_ownership",
            "leadership_expected": "emerging",
            "learning_required": "medium"
        }
    }
    
    candidate = {
        "candidate_id": "candidate_001",
        "name": "Priya Sharma",
        "ownership_language_score": 0.82,
        "timeline": [
            {
                "year_start": 2023,
                "year_end": None,
                "company": "FinStartup",
                "title": "Senior Engineer",
                "skills_gained": ["Python", "AWS", "Kafka", "PostgreSQL"],
                "complexity_level": 8,
                "ownership_signals": ["Built", "Led"]
            },
            {
                "year_start": 2021,
                "year_end": 2023,
                "company": "TechCorp",
                "title": "SDE 2",
                "skills_gained": ["Python", "AWS", "MySQL"],
                "complexity_level": 6,
                "ownership_signals": ["Designed"]
            }
        ],
        "achievement_signals": [
            {"text": "Reduced latency by 40%", "type": "performance_metric", "strength": "strong"},
            {"text": "2M transactions per day", "type": "scale_metric", "strength": "strong"}
        ],
        "leadership_signals": ["Led team of 3 engineers"],
        "communication_signals": ["Presented tech architecture to leadership"],
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
        "capability_map": {
            "backend_engineering": {"level": "high", "evidence_tier": 1, "evidence": "Built distributed payments at scale", "confidence": 0.95},
            "cloud_infrastructure": {"level": "high", "evidence_tier": 2, "evidence": "AWS production deployment", "confidence": 0.90},
            "distributed_systems": {"level": "high", "evidence_tier": 1, "evidence": "2M tx/day payment system", "confidence": 0.95}
        },
        "hidden_capabilities": [
            {
                "capability": "ml_engineering",
                "evidence": "Built fraud models and data pipelines",
                "title_mismatch": True,
                "note": "ML capabilities found on backend profile"
            }
        ],
        "original_profile": {
            "certifications": ["AWS Solutions Architect"],
            "projects": [
                {"name": "Payment Gateway", "description": "High throughput system"}
            ]
        }
    }
    
    return role, candidate

def run_live_scoring(role: dict, candidate: dict) -> dict:
    """Runs all six signals using live LLM calls."""
    return {
        "capability_fit": score_capability(role, candidate),
        "trajectory": score_trajectory(role, candidate),
        "impact": score_impact(role, candidate),
        "evidence_confidence": score_confidence(role, candidate),
        "hidden_talent": score_hidden_talent(role, candidate),
        "behavioral": score_behavioral(role, candidate)
    }

def run_mock_scoring(role: dict, candidate: dict) -> dict:
    """Simulates the scorecard results for testing environment without API key."""
    print("[verify_scoring.py] Running scoring simulation (Mock LLM scores)...")
    return {
        "capability_fit": {"score": 91.0, "rationale": ["Direct match for critical backend need", "High evidence tier for AWS and Kafka"]},
        "trajectory": {"score": 88.0, "rationale": ["Consistent promotions from SDE2 to Senior in 4 years", "Strong tenure length"]},
        "impact": {"score": 85.0, "rationale": ["Quantified 40% latency reduction", "Handled 2M transactions per day scale"]},
        "evidence_confidence": {"score": 92.0, "rationale": ["Strong evidence from production systems", "AWS Solutions Architect certificate"]},
        "hidden_talent": {"score": 70.0, "rationale": ["ML engineering capability found on backend engineer profile"]},
        "behavioral": {"score": 78.0, "rationale": ["Twin archetype aligns with Systems Builder", "Experience in high ownership series_b startup"]}
    }

if __name__ == "__main__":
    print("=== STARTING PHASE 2 SCORING ENGINES TEST ===")
    role, candidate = get_test_data()
    
    try:
        if settings.anthropic_api_key == "mock_key":
            results = run_mock_scoring(role, candidate)
        else:
            print("[verify_scoring.py] Running full live scoring engines...")
            results = run_live_scoring(role, candidate)
            
        print("\n================== SCORECARD ==================")
        print(f"Candidate: {candidate['name']}")
        print(f"Role: {role['role_title']}")
        print("-----------------------------------------------")
        for signal, res in results.items():
            print(f"[Scoring] {signal.upper()}: Score = {res['score']}")
            for r in res.get("rationale", []):
                print(f"  - {r}")
            print("-----------------------------------------------")
            assert 0 <= res["score"] <= 100
            assert isinstance(res["rationale"], list)
            
        print("===============================================")
        print("=== PHASE 2 SCORING ENGINES TEST PASSED ===")
    except Exception as e:
        print(f"[ERROR] Scoring integration test failed: {e}")
        sys.exit(1)
