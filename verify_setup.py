import json
from pathlib import Path
from config import settings
from schemas import (
    JobDescriptionInput,
    CandidateProfile,
    RoleObject,
    StructuredCandidate,
    ScoredCandidate,
    safe_validate,
    Education,
    WorkHistory,
    Project
)

def test_config_loading() -> None:
    """Verifies that the Settings model correctly loads configurations and default values."""
    print("[verify_setup.py] Testing configuration loading...")
    assert settings.model_name is not None
    assert settings.max_tokens > 0
    assert "senior_ic" in settings.weights
    print("[verify_setup.py] Config verification passed.")

def test_capability_graph_loading() -> None:
    """Verifies that the capability graph JSON file exists, is valid JSON, and matches expectations."""
    print("[verify_setup.py] Testing capability graph loading...")
    path = Path("data/capability_graph.json")
    assert path.exists(), f"Capability graph file not found at {path}"
    
    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    
    assert "capabilities" in graph
    assert "backend_engineering" in graph["capabilities"]
    assert "skills" in graph["capabilities"]["backend_engineering"]
    print("[verify_setup.py] Capability graph verification passed.")

def test_schema_validations() -> None:
    """Verifies that key schemas validate valid structures and reject invalid ones."""
    print("[verify_setup.py] Testing schema validations...")
    
    # 1. Test JobDescriptionInput
    jd_data = {
        "id": "jd_001",
        "raw_text": "We are looking for a backend engineer with AWS experience",
        "title": "Backend Engineer",
        "company": "TestCorp",
        "domain": "Fintech"
    }
    validated_jd = safe_validate(jd_data, JobDescriptionInput)
    assert validated_jd is not None
    assert validated_jd["id"] == "jd_001"
    
    # 2. Test CandidateProfile
    candidate_data = {
        "id": "candidate_001",
        "name": "John Doe",
        "current_title": "SDE 1",
        "total_experience_years": 2.5,
        "education": [
            {"degree": "BS CS", "institution": "State Univ", "year": 2021}
        ],
        "work_history": [
            {
                "company": "OldCorp",
                "title": "Software Engineer",
                "start_year": 2021,
                "end_year": 2023,
                "is_current": False,
                "description": "Wrote APIs in Python",
                "company_size": "enterprise",
                "domain": "retail"
            }
        ],
        "skills": ["Python", "Flask", "AWS"],
        "projects": [
            {
                "name": "API Refactor",
                "description": "Refactored legacy API",
                "tech_stack": ["Python", "Flask"],
                "metrics": ["50% faster latency"]
            }
        ],
        "certifications": ["AWS Practitioner"],
        "platforms": {"github": "github.com/johndoe"}
    }
    validated_candidate = safe_validate(candidate_data, CandidateProfile)
    assert validated_candidate is not None
    assert validated_candidate["name"] == "John Doe"
    assert len(validated_candidate["work_history"]) == 1
    
    print("[verify_setup.py] Schema verification passed.")

if __name__ == "__main__":
    print("=== STARTING PHASE 0 VERIFICATION ===")
    try:
        test_config_loading()
        test_capability_graph_loading()
        test_schema_validations()
        print("=== PHASE 0 VERIFICATION COMPLETED: ALL TESTS PASSED ===")
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        exit(1)
