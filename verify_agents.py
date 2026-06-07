import sys
import json
from pathlib import Path

# Add project root to sys.path so we can run this module directly
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings
from agents.twin_matcher import score_twin_match
from agents.debate_panel import run_debate
from agents.devils_advocate import analyze_risks
from agents.calibration_agent import calibrate_candidate_pool

def run_agent_integration_tests():
    print("=== STARTING AGENT INTEGRATION TESTS ===")
    
    mock_role = {
        "role_title": "Senior AI Engineer",
        "seniority_band": "senior",
        "domain": "Fintech",
        "capability_needs": {
            "backend_engineering": "critical",
            "ml_engineering": "critical"
        },
        "role_persona": {
            "archetype": "Systems Builder",
            "environment": "startup_scaleup",
            "execution_expectation": "high_ownership",
            "leadership_expected": "emerging",
            "learning_required": "high"
        }
    }
    
    mock_candidate = {
        "candidate_id": "CAND_1111111",
        "name": "Priya Sharma",
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
            "backend_engineering": {"level": "high", "evidence_tier": 1, "evidence": "Built payments at scale", "confidence": 0.95},
            "ml_engineering": {"level": "high", "evidence_tier": 2, "evidence": "NLP RAG search pipelines", "confidence": 0.90}
        },
        "timeline": [
            {"year_start": 2021, "year_end": 2023, "company": "TechCorp", "promoted": True, "duration_months": 24}
        ],
        "original_profile": {
            "skills": [{"name": "Python"}, {"name": "Vector Search"}]
        },
        "scores": {
            "capability_fit": 92.0,
            "trajectory": 85.0,
            "impact": 88.0,
            "evidence_confidence": 90.0,
            "hidden_talent": 70.0,
            "behavioral": 80.0
        }
    }
    
    # 1. Test Twin Matcher
    print("[Test] Running Twin Matcher...")
    twin_res = score_twin_match(mock_role["role_persona"], mock_candidate["candidate_twin"])
    print(f"Twin Matcher Output: {json.dumps(twin_res, indent=2)}")
    assert "score" in twin_res
    assert isinstance(twin_res["rationale"], list)
    
    # 2. Test Debate Panel
    print("[Test] Running Debate Panel...")
    debate_res = run_debate(mock_role, mock_candidate)
    print(f"Debate Panel Output: {json.dumps(debate_res, indent=2)}")
    assert "debate_scores" in debate_res
    assert "consensus" in debate_res["debate_scores"]
    
    # 3. Test Devil's Advocate
    print("[Test] Running Devil's Advocate...")
    da_res = analyze_risks(mock_role, mock_candidate)
    print(f"Devil's Advocate Output: {json.dumps(da_res, indent=2)}")
    assert "devils_advocate" in da_res
    assert "interview_questions" in da_res
    
    # 4. Test Calibration Agent
    print("[Test] Running Calibration Agent...")
    mock_pool = [
        {
            "candidate_id": "CAND_001",
            "scores": {"capability_fit": 95, "trajectory": 90, "impact": 95, "evidence_confidence": 95, "hidden_talent": 70, "behavioral": 90}
        },
        {
            "candidate_id": "CAND_002",
            "scores": {"capability_fit": 60, "trajectory": 60, "impact": 60, "evidence_confidence": 60, "hidden_talent": 50, "behavioral": 60}
        }
    ]
    role_weights = {
        "capability_fit": 0.30,
        "trajectory": 0.15,
        "impact": 0.25,
        "evidence_confidence": 0.15,
        "hidden_talent": 0.05,
        "behavioral": 0.10
    }
    calibrated = calibrate_candidate_pool(mock_pool, role_weights)
    print(f"Calibrated Pool Sample: {json.dumps(calibrated, indent=2)}")
    assert len(calibrated) == 2
    assert calibrated[0]["rank"] == 1
    assert calibrated[0]["weighted_score"] == 95.0
    assert calibrated[1]["weighted_score"] == 55.0
    assert calibrated[0]["tier"] == "A"
    assert calibrated[1]["tier"] == "C"
    
    print("=== AGENT INTEGRATION TESTS PASSED SUCCESSFULLY ===")

def verify_precomputed_output_file():
    print("[Test] Checking generated precomputed features file...")
    path = Path("results/precomputed_features.json")
    assert path.exists(), f"Output file does not exist: {path}"
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} candidates from precomputed JSON.")
    assert len(data) > 0
    
    # Check first candidate fields
    cand = data[0]
    required_keys = [
        "candidate_id", "name", "scores", "capability_map",
        "twin_match_score", "debate_scores", "devils_advocate",
        "interview_questions", "weighted_score", "current_fit",
        "future_fit", "confidence_band", "tier", "recruiter_verdict", "rank"
    ]
    for key in required_keys:
        assert key in cand, f"Missing key in precomputed output candidate: '{key}'"
        
    print(f"[Test] Candidate rank 1 '{cand['name']}' ({cand['candidate_id']}) verification passed.")
    print("[Test] Precomputed output file verification passed successfully!")

if __name__ == "__main__":
    run_agent_integration_tests()
    print("")
    verify_precomputed_output_file()
    print("\n=== ALL VERIFICATION CHECKS PASSED ===")
