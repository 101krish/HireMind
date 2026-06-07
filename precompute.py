import json
import docx
from pathlib import Path
from config import settings
from engines.jd_engine import decompose_jd
from engines.profile_engine import decompose_profile
from engines.capability_engine import map_capabilities
from engines.embedding_engine import fast_filter, get_candidate_text
from engines.scoring.capability_fit import score_candidate as score_capability
from engines.scoring.trajectory import score_candidate as score_trajectory
from engines.scoring.impact import score_candidate as score_impact
from engines.scoring.confidence import score_candidate as score_confidence
from engines.scoring.hidden_talent import score_candidate as score_hidden_talent
from engines.scoring.behavioral import score_candidate as score_behavioral

def read_job_description(path: Path) -> str:
    """Reads the job description text from docx, txt, or md file formats."""
    if path.suffix == ".docx":
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def load_candidates(path: Path) -> list:
    """Loads candidates from a JSON array file or a JSON Lines (.jsonl) file."""
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
        
    if path.suffix == ".jsonl":
        candidates = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        return candidates
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def run_offline_pipeline(candidates_path: str, jd_path: str, output_features_path: str) -> None:
    """
    Offline Mode orchestrator.
    Runs fast filter first, then performs deep LLM analysis and scoring on the top candidates.
    Saves precomputed features to results/precomputed_features.json.
    """
    print("[Precompute] Starting offline pipeline run...")
    
    if settings.anthropic_api_key == "mock_key":
        print("[Precompute] ANTHROPIC_API_KEY is not configured. Running offline pipeline in SIMULATION mode...")
        # Create a mock role object matching the JD decomposition schema
        role_object = {
            "jd_id": "jd_001",
            "role_title": "Senior AI Engineer",
            "seniority_band": "senior",
            "domain": "Fintech",
            "hard_requirements": [
                "Production experience with embeddings-based retrieval systems",
                "Production experience with vector databases",
                "Strong Python",
                "Evaluation frameworks (NDCG, MRR, MAP)"
            ],
            "soft_requirements": [
                "LLM fine-tuning experience (LoRA, QLoRA)",
                "Learning-to-rank models"
            ],
            "implied_needs": ["Scrappy founding team attitude", "Ownership mindset"],
            "culture_signals": ["Fast-moving startup", "Direct and rapid decisions"],
            "capability_needs": {
                "backend_engineering": "critical",
                "cloud_infrastructure": "high",
                "distributed_systems": "high",
                "databases": "high",
                "ml_engineering": "critical",
                "frontend_engineering": "none",
                "data_engineering": "medium",
                "leadership": "medium"
            },
            "role_persona": {
                "archetype": "Systems Builder",
                "environment": "startup_scaleup",
                "execution_expectation": "high_ownership",
                "leadership_expected": "emerging",
                "learning_required": "high"
            },
            "contradictions_detected": [],
            "inflation_adjusted": False,
            "disqualifiers": ["consulting_only_career", "cv_speech_robotics_only", "no_production_deployment"]
        }
        
        # Load all candidates
        candidates_p = Path(candidates_path)
        raw_candidates = load_candidates(candidates_p)
        print(f"[Precompute] Loaded {len(raw_candidates)} candidates for simulation.")
        
        # Build mock precomputed features for all candidates in the dataset
        precomputed_features = []
        for i, raw_cand in enumerate(raw_candidates):
            cid = raw_cand.get("candidate_id")
            profile_sec = raw_cand.get("profile", {})
            name = profile_sec.get("anonymized_name", "Unknown")
            
            # Formulate simulated score based on AI/ML skills and experience
            skills = [s.get("name", "").lower() for s in raw_cand.get("skills", [])]
            is_ml = any("ml" in s or "machine learning" in s or "nlp" in s or "retrieval" in s or "embedding" in s or "vector" in s or "rag" in s for s in skills)
            is_python = any("python" in s for s in skills)
            years_exp = profile_sec.get("years_of_experience", 0.0)
            
            # Higher experience and skill alignment yields higher mock score
            cap_score = 65.0
            if is_ml and is_python:
                cap_score = 90.0 + min(5.0, years_exp / 2.0)
            elif is_python or is_ml:
                cap_score = 78.0
            
            # Factor in consulting disqualifiers
            cur_company = profile_sec.get("current_company", "").lower()
            consulting_companies = ["wipro", "tcs", "infosys", "cognizant", "tech mahindra", "hcl", "accenture", "capgemini"]
            is_consulting = any(c in cur_company for c in consulting_companies)
            
            if is_consulting:
                cap_score -= 15.0  # Consulting penalty
                
            scores = {
                "capability_fit": cap_score,
                "trajectory": 82.0 if years_exp > 4 else 70.0,
                "impact": 85.0 if years_exp > 5 else 72.0,
                "evidence_confidence": 88.0,
                "hidden_talent": 75.0 if is_ml else 60.0,
                "behavioral": 80.0
            }
            
            record = {
                "candidate_id": cid,
                "name": name,
                "scores": scores,
                "capability_map": {
                    "backend_engineering": {"level": "high" if is_python else "medium", "evidence_tier": 2, "evidence": "Production systems dev", "confidence": 0.85},
                    "ml_engineering": {"level": "high" if is_ml else "none", "evidence_tier": 2, "evidence": "NLP/Embeddings work", "confidence": 0.80}
                },
                "hidden_capabilities": [],
                "candidate_twin": {
                    "archetype": "Systems Builder" if is_python else "Product Specialist",
                    "learning_speed": "high" if years_exp < 7 else "medium",
                    "ceiling": "staff_engineer"
                },
                "redrob_signals": raw_cand.get("redrob_signals", {}),
                "reasoning": [
                    f"Candidate has {years_exp} years experience with strong backend capabilities.",
                    f"Skills match: Python={is_python}, ML/NLP={is_ml}.",
                    "Direct relevance to vector search and index operations."
                ],
                "raw_profile": raw_cand
            }
            precomputed_features.append(record)
            
        out_p = Path(output_features_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(precomputed_features, f, indent=2, ensure_ascii=False)
            
        print(f"[Precompute] Simulated precomputation complete. Saved {len(precomputed_features)} records to {output_features_path}")
        return
    
    # 1. Load Job Description and Decompose it
    jd_p = Path(jd_path)
    jd_text = read_job_description(jd_p)
    print(f"[Precompute] Job description loaded. Length: {len(jd_text)} characters.")
    
    role_object = decompose_jd("jd_001", jd_text)
    if not role_object:
        print("[ERROR] Decomposing Job Description failed. Exiting.")
        return
        
    print(f"[Precompute] Job Description Decomposed: {role_object['role_title']}")
    
    # 2. Load all candidates
    candidates_p = Path(candidates_path)
    raw_candidates = load_candidates(candidates_p)
    print(f"[Precompute] Loaded {len(raw_candidates)} candidates.")
    
    # 3. Perform Fast Filtering to reduce pool to top N (e.g. top 100 or 150)
    # We construct a mock structured candidate for fast filtering
    temp_candidates = []
    for cand in raw_candidates:
        temp_candidates.append({
            "name": cand.get("profile", {}).get("anonymized_name", "Unknown"),
            "candidate_id": cand.get("candidate_id", "unknown"),
            "narrative_arc": "generalist_builder",
            "candidate_twin": {"archetype": "SDE", "environment_fit": "any"},
            "timeline": [],
            "achievement_signals": [],
            "original_profile": cand
        })
        
    # We want to precompute features for all candidates in small pools, or top N in large pools
    top_n = min(150, len(raw_candidates))
    filtered_candidates = fast_filter(role_object, temp_candidates, top_n=top_n)
    
    # Extract the original candidate dictionaries for the selected top N candidates
    selected_ids = {c["candidate_id"] for c in filtered_candidates}
    selected_candidates = [c for c in raw_candidates if c.get("candidate_id") in selected_ids]
    
    print(f"[Precompute] Selected top {len(selected_candidates)} candidates for deep LLM scoring.")
    
    # 4. Run deep LLM analysis and scoring for selected candidates
    precomputed_features = []
    
    for i, raw_cand in enumerate(selected_candidates):
        cid = raw_cand.get("candidate_id")
        name = raw_cand.get("profile", {}).get("anonymized_name", "Unknown")
        print(f"\n[Precompute] Deep parsing candidate {i+1}/{len(selected_candidates)}: {name} ({cid})")
        
        # Decompose Profile
        decomposed = decompose_profile(raw_cand)
        if not decomposed:
            print(f"[ERROR] Failed to decompose candidate {name}. Skipping.")
            continue
            
        # Map Capabilities
        mapped = map_capabilities(decomposed)
        if not mapped:
            print(f"[ERROR] Failed to map capabilities for {name}. Skipping.")
            continue
            
        # Run 6 scoring signals
        scores = {}
        reasoning_bullets = []
        
        # Signal 1: Capability Fit
        cap_res = score_capability(role_object, mapped)
        scores["capability_fit"] = cap_res.get("score", 50.0)
        reasoning_bullets.extend(cap_res.get("rationale", []))
        
        # Signal 2: Trajectory
        traj_res = score_trajectory(role_object, mapped)
        scores["trajectory"] = traj_res.get("score", 50.0)
        reasoning_bullets.extend(traj_res.get("rationale", []))
        
        # Signal 3: Impact
        imp_res = score_impact(role_object, mapped)
        scores["impact"] = imp_res.get("score", 50.0)
        reasoning_bullets.extend(imp_res.get("rationale", []))
        
        # Signal 4: Evidence Confidence
        conf_res = score_confidence(role_object, mapped)
        scores["evidence_confidence"] = conf_res.get("score", 50.0)
        
        # Signal 5: Hidden Talent
        ht_res = score_hidden_talent(role_object, mapped)
        scores["hidden_talent"] = ht_res.get("score", 50.0)
        
        # Signal 6: Behavioral
        beh_res = score_behavioral(role_object, mapped)
        scores["behavioral"] = beh_res.get("score", 50.0)
        
        # Compile precomputed record
        record = {
            "candidate_id": cid,
            "name": name,
            "scores": scores,
            "capability_map": mapped.get("capability_map", {}),
            "hidden_capabilities": mapped.get("hidden_capabilities", []),
            "candidate_twin": mapped.get("candidate_twin", {}),
            "redrob_signals": raw_cand.get("redrob_signals", {}),
            "reasoning": reasoning_bullets[:4], # Keep top 4 reasoning statements
            "raw_profile": raw_cand
        }
        precomputed_features.append(record)
        
    # 5. Save precomputed features
    out_p = Path(output_features_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(precomputed_features, f, indent=2, ensure_ascii=False)
        
    print(f"\n[Precompute] Saved {len(precomputed_features)} precomputed candidates to {output_features_path}")
    print("[Precompute] Offline precomputation run complete.")

if __name__ == "__main__":
    # Self-test or execution script
    import argparse
    parser = argparse.ArgumentParser(description="PRISM Precomputations (Offline Mode)")
    parser.add_argument("--candidates", default="data/sample_candidates.json", help="Path to candidates dataset")
    parser.add_argument("--jd", default="data/job_description.docx", help="Path to Job Description")
    parser.add_argument("--out", default="results/precomputed_features.json", help="Output precomputed features JSON path")
    args = parser.parse_args()
    
    run_offline_pipeline(args.candidates, args.jd, args.out)
