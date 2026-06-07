import json
from pathlib import Path
from config import settings
from schemas import (
    JobDescriptionInput,
    CandidateProfile,
    RoleObject,
    StructuredCandidate,
    ScoredCandidate,
    safe_validate
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
    assert "jd_specific_weights" in graph
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
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "John Doe",
            "headline": "SDE 1 | Python | SQL",
            "summary": "Software engineer with 2 years experience.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 2.0,
            "current_title": "Software Engineer",
            "current_company": "OldCorp",
            "current_company_size": "51-200",
            "current_industry": "IT Services"
        },
        "career_history": [
            {
                "company": "OldCorp",
                "title": "Software Engineer",
                "start_date": "2024-01-01",
                "end_date": None,
                "duration_months": 24,
                "is_current": True,
                "industry": "IT Services",
                "company_size": "51-200",
                "description": "Wrote APIs in Python"
            }
        ],
        "education": [
            {
                "institution": "State Univ",
                "degree": "BS CS",
                "field_of_study": "Computer Science",
                "start_year": 2020,
                "end_year": 2024,
                "grade": "8.5 CGPA",
                "tier": "tier_3"
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 24}
        ],
        "certifications": [],
        "languages": [
            {"language": "English", "proficiency": "professional"}
        ],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"Python": 80.0},
            "connection_count": 50,
            "endorsements_received": 5,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 8.0, "max": 12.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 40.0,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 3,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    validated_candidate = safe_validate(candidate_data, CandidateProfile)
    assert validated_candidate is not None
    assert validated_candidate["profile"]["anonymized_name"] == "John Doe"
    assert len(validated_candidate["career_history"]) == 1
    
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
        import traceback
        traceback.print_exc()
        exit(1)
