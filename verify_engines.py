import sys
import json
from pathlib import Path
from config import settings
from schemas import JobDescriptionInput, CandidateProfile
from engines.jd_engine import decompose_jd
from engines.profile_engine import decompose_profile
from engines.capability_engine import map_capabilities
from engines.embedding_engine import fast_filter

def get_sample_jd() -> dict:
    """Returns a mock job description input dictionary."""
    return {
        "id": "jd_001",
        "title": "Senior Python Backend Engineer",
        "company": "FintechCorp",
        "domain": "Fintech",
        "raw_text": (
            "We are looking for a Senior Backend Engineer with 5+ years of experience in Python.\n"
            "You will build scale systems, manage distributed systems using Kafka, and deploy on AWS.\n"
            "Requirements:\n"
            "- Python expertise, FastAPI/Django\n"
            "- Database experience (PostgreSQL, Redis)\n"
            "- Distributed queuing (Kafka/RabbitMQ)\n"
            "- Cloud Infrastructure (AWS, Terraform)\n"
        )
    }

def get_sample_candidates() -> list:
    """Returns a list of raw candidate profile input dictionaries."""
    return [
        {
            "id": "candidate_001",
            "name": "Priya Sharma",
            "current_title": "Senior Software Engineer",
            "total_experience_years": 5.5,
            "education": [
                {"degree": "B.Tech Computer Science", "institution": "IIT Bombay", "year": 2019}
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
                },
                {
                    "company": "TechCorp",
                    "title": "SDE 2",
                    "start_year": 2019,
                    "end_year": 2023,
                    "is_current": False,
                    "description": "Developed REST APIs using Python and Django. Deployed microservices to AWS.",
                    "company_size": "enterprise",
                    "domain": "ecommerce"
                }
            ],
            "skills": ["Python", "AWS", "Kafka", "PostgreSQL", "Terraform", "Django"],
            "projects": [
                {
                    "name": "Payment Gateway Integration",
                    "description": "High-throughput payment service",
                    "tech_stack": ["Python", "Kafka", "AWS", "PostgreSQL"],
                    "metrics": ["2M tx/day", "40% latency reduction"]
                }
            ],
            "certifications": ["AWS Solutions Architect"],
            "platforms": {"github": "github.com/priya"}
        },
        {
            "id": "candidate_002",
            "name": "Alex Smith",
            "current_title": "Frontend SDE 2",
            "total_experience_years": 3.0,
            "education": [
                {"degree": "BS Information Technology", "institution": "State University", "year": 2022}
            ],
            "work_history": [
                {
                    "company": "SaaS Corp",
                    "title": "Frontend Engineer",
                    "start_year": 2022,
                    "end_year": None,
                    "is_current": True,
                    "description": "Developed React dashboard and components. Improved SEO and UI accessibility.",
                    "company_size": "startup",
                    "domain": "marketing"
                }
            ],
            "skills": ["JavaScript", "React", "HTML", "CSS", "TypeScript"],
            "projects": [
                {
                    "name": "Dashboard Refactor",
                    "description": "Refactored UI components to Tailwind",
                    "tech_stack": ["React", "TypeScript", "Tailwind CSS"],
                    "metrics": ["Page speed score increase of 15%"]
                }
            ],
            "certifications": [],
            "platforms": {"github": "github.com/alex"}
        }
    ]

def run_mock_pipeline(jd_input: dict, raw_candidates: list) -> None:
    """
    Runs a mock simulation of the Phase 1 pipeline for test environments
    where ANTHROPIC_API_KEY is not configured.
    """
    print("[verify_engines.py] Run simulation: Mocking LLM outputs for testing...")
    
    # 1. Mock Role Object
    mock_role = {
        "jd_id": jd_input["id"],
        "role_title": jd_input["title"],
        "seniority_band": "senior",
        "domain": jd_input["domain"],
        "hard_requirements": ["Python", "AWS", "Kafka"],
        "soft_requirements": ["FastAPI", "PostgreSQL"],
        "implied_needs": ["Ownership mindset", "Scale operations"],
        "culture_signals": ["Fast-moving startup"],
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
            "environment": "startup",
            "execution_expectation": "high_ownership",
            "leadership_expected": "emerging",
            "learning_required": "medium"
        },
        "contradictions_detected": [],
        "inflation_adjusted": False
    }
    
    # 2. Mock Decomposed & Mapped Candidates
    structured_candidates = []
    for raw in raw_candidates:
        if raw["id"] == "candidate_001":
            # Priya (Strong backend fit)
            mock_cand = {
                "candidate_id": raw["id"],
                "name": raw["name"],
                "timeline": [
                    {
                        "year_start": 2023,
                        "year_end": None,
                        "company": "FinStartup",
                        "title": "Senior Engineer",
                        "skills_gained": ["Python", "AWS", "Kafka", "PostgreSQL"],
                        "complexity_level": 8,
                        "ownership_signals": ["Built", "Led"]
                    }
                ],
                "achievement_signals": [
                    {"text": "Reduced latency by 40%", "type": "performance_metric", "strength": "strong"}
                ],
                "ownership_language_score": 0.82,
                "leadership_signals": ["Led team of 3"],
                "communication_signals": [],
                "narrative_arc": "generalist_builder",
                "candidate_twin": {
                    "archetype": "Systems Builder",
                    "risk_profile": "low",
                    "learning_speed": "high",
                    "execution_style": "high_ownership",
                    "environment_fit": "startup",
                    "leadership": "emerging",
                    "ceiling": "staff_engineer"
                },
                "capability_map": {
                    "backend_engineering": {"level": "high", "evidence_tier": 1, "evidence": "Built payments at scale", "confidence": 0.95},
                    "cloud_infrastructure": {"level": "high", "evidence_tier": 2, "evidence": "AWS microservices", "confidence": 0.90},
                    "distributed_systems": {"level": "high", "evidence_tier": 1, "evidence": "Kafka queues", "confidence": 0.95}
                },
                "hidden_capabilities": [],
                "original_profile": raw
            }
        else:
            # Alex (Frontend - Poor backend fit)
            mock_cand = {
                "candidate_id": raw["id"],
                "name": raw["name"],
                "timeline": [
                    {
                        "year_start": 2022,
                        "year_end": None,
                        "company": "SaaS Corp",
                        "title": "Frontend Engineer",
                        "skills_gained": ["React", "TypeScript", "Tailwind CSS"],
                        "complexity_level": 5,
                        "ownership_signals": ["Developed"]
                    }
                ],
                "achievement_signals": [],
                "ownership_language_score": 0.5,
                "leadership_signals": [],
                "communication_signals": [],
                "narrative_arc": "specialist",
                "candidate_twin": {
                    "archetype": "Frontend Specialist",
                    "risk_profile": "low",
                    "learning_speed": "medium",
                    "execution_style": "collaborative",
                    "environment_fit": "enterprise",
                    "leadership": "none",
                    "ceiling": "senior"
                },
                "capability_map": {
                    "frontend_engineering": {"level": "high", "evidence_tier": 2, "evidence": "React development", "confidence": 0.90}
                },
                "hidden_capabilities": [],
                "original_profile": raw
            }
        structured_candidates.append(mock_cand)
        
    # 3. Perform Fast Filtering
    ranked = fast_filter(mock_role, structured_candidates, top_n=2)
    
    print("[verify_engines.py] Ranked results from simulation:")
    for r in ranked:
        print(f" - {r['name']}: Similarity = {r['fast_filter_score']:.4f}")
        
    assert len(ranked) == 2
    assert ranked[0]["candidate_id"] == "candidate_001"
    print("[verify_engines.py] Pipeline simulation completed successfully!")

def run_real_pipeline(jd_input: dict, raw_candidates: list) -> None:
    """
    Runs the full live Phase 1 pipeline using the actual Anthropic API.
    """
    print("[verify_engines.py] Run pipeline: Decomposing Job Description...")
    role_object = decompose_jd(jd_input["id"], jd_input["raw_text"])
    assert role_object
    print(f"[verify_engines.py] Job Description Decomposed: {role_object['role_title']}")
    
    structured_candidates = []
    for raw in raw_candidates:
        print(f"[verify_engines.py] Decomposing candidate: {raw['name']}")
        decomposed = decompose_profile(raw)
        assert decomposed
        
        print(f"[verify_engines.py] Mapping capabilities for candidate: {raw['name']}")
        mapped = map_capabilities(decomposed)
        assert mapped
        
        structured_candidates.append(mapped)
        
    print("[verify_engines.py] Performing fast filtering on candidates...")
    ranked = fast_filter(role_object, structured_candidates, top_n=2)
    
    print("[verify_engines.py] Ranked results from live pipeline:")
    for r in ranked:
        print(f" - {r['name']}: Similarity = {r['fast_filter_score']:.4f}")
        
    assert len(ranked) > 0
    print("[verify_engines.py] Live pipeline run completed successfully!")

if __name__ == "__main__":
    print("=== STARTING PHASE 1 ENGINES INTEGRATION TEST ===")
    jd = get_sample_jd()
    candidates = get_sample_candidates()
    
    try:
        if settings.anthropic_api_key == "mock_key":
            print("[verify_engines.py] ANTHROPIC_API_KEY is not configured. Running pipeline simulation...")
            run_mock_pipeline(jd, candidates)
        else:
            print("[verify_engines.py] Running full live pipeline integration test...")
            run_real_pipeline(jd, candidates)
        print("=== PHASE 1 ENGINES INTEGRATION TEST PASSED ===")
    except Exception as e:
        print(f"[ERROR] Engines integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
