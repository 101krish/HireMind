import numpy as np
from config import settings

# Global cached model instance to avoid reloading
_model = None

def get_embedding_model():
    """
    Lazy loads and caches the SentenceTransformer model defined in config settings.
    If sentence_transformers is not installed or fails, falls back to lightweight mode.
    """
    global _model
    if _model is None:
        model_name = settings.embedding_model
        print(f"[Embedding Engine] Loading local embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[ERROR] Failed to load SentenceTransformer model {model_name}: {e}")
            print("[Embedding Engine] Falling back to lightweight keyword-matching mode.")
            _model = "lightweight_fallback"
    return _model


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two 1D numpy arrays.
    Returns 0.0 if either vector is all zeros (sparse/empty inputs).
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def get_role_text(role: dict) -> str:
    """
    Constructs a detailed textual representation of a RoleObject for embedding.
    Handles missing or sparse fields gracefully.
    """
    title = role.get("role_title", "")
    seniority = role.get("seniority_band", "")
    domain = role.get("domain", "")
    
    hard_reqs = ", ".join(role.get("hard_requirements", []))
    soft_reqs = ", ".join(role.get("soft_requirements", []))
    implied_needs = ", ".join(role.get("implied_needs", []))
    
    role_persona = role.get("role_persona", {})
    archetype = role_persona.get("archetype", "")
    environment = role_persona.get("environment", "")
    
    text_parts = [
        f"Job Title: {title}",
        f"Seniority: {seniority}",
        f"Domain: {domain}",
        f"Archetype: {archetype} in {environment} environment",
        f"Hard Requirements: {hard_reqs}",
        f"Soft Requirements: {soft_reqs}",
        f"Implied Needs: {implied_needs}"
    ]
    return ". ".join([p for p in text_parts if p.strip()])

def get_candidate_text(candidate: dict) -> str:
    """
    Constructs a detailed textual representation of a StructuredCandidate for embedding.
    Handles missing or sparse fields gracefully.
    """
    name = candidate.get("name", "Unknown Candidate")
    narrative = candidate.get("narrative_arc", "")
    
    twin = candidate.get("candidate_twin", {})
    archetype = twin.get("archetype", "")
    environment_fit = twin.get("environment_fit", "")
    
    # Reconstruct timeline events summary
    timeline_summaries = []
    for event in candidate.get("timeline", []):
        company = event.get("company", "")
        title = event.get("title", "")
        skills = ", ".join(event.get("skills_gained", []) or event.get("skills_demonstrated", []) or [])
        if company and title:
            timeline_summaries.append(f"Worked at {company} as {title} developing skills: {skills}")
        elif event.get("event") == "graduation":
            timeline_summaries.append(f"Graduated in {event.get('year', '')}")
            
    timeline_str = "; ".join(timeline_summaries)
    
    # Extract explicit skills list from original profile if available
    skills_list = []
    if "original_profile" in candidate and isinstance(candidate["original_profile"], dict):
        skills_list = candidate["original_profile"].get("skills", [])
    if not skills_list:
        # Fallback to skills mentioned in timeline
        all_skills = set()
        for event in candidate.get("timeline", []):
            for s in (event.get("skills_gained", []) or event.get("skills_demonstrated", []) or []):
                all_skills.add(s)
        skills_list = list(all_skills)
        
    skills_str = ", ".join(skills_list)
    
    # Extract achievements
    achievements = []
    for ach in candidate.get("achievement_signals", []):
        ach_text = ach.get("text", "")
        if ach_text:
            achievements.append(ach_text)
    achievements_str = ". ".join(achievements)
    
    text_parts = [
        f"Candidate Name: {name}",
        f"Narrative Arc: {narrative}",
        f"Twin Archetype: {archetype}",
        f"Best Fit Environment: {environment_fit}",
        f"Skills: {skills_str}",
        f"Work History: {timeline_str}",
        f"Key Achievements: {achievements_str}"
    ]
    return ". ".join([p for p in text_parts if p.strip()])

def get_keyword_overlap_score(role_text: str, candidate_text: str) -> float:
    """
    Computes a simple, zero-dependency token-overlap score.
    Used as a lightweight fallback when sentence-transformers is not available.
    """
    role_words = set(role_text.lower().replace('.', ' ').replace(',', ' ').split())
    cand_words = set(candidate_text.lower().replace('.', ' ').replace(',', ' ').split())
    
    # Filter common English stopwords
    stopwords = {"and", "the", "a", "or", "in", "of", "to", "for", "with", "is", "on", "at", "by", "an", "as"}
    role_words = role_words - stopwords
    cand_words = cand_words - stopwords
    
    if not role_words:
        return 0.0
        
    intersection = role_words.intersection(cand_words)
    return float(len(intersection) / len(role_words))

def fast_filter(role: dict, candidates: list, top_n: int = 30) -> list:
    """
    Performs Phase 1 Fast Filtering.
    Embeds the role and candidates, calculates cosine similarities, 
    and returns the top_n candidates sorted by similarity score.
    Falls back to a lightweight keyword overlap model if sentence-transformers is unavailable.
    """
    if not candidates:
        print("[Embedding Engine] Empty candidate list provided to fast_filter.")
        return []
        
    print(f"[Embedding Engine] Performing fast filter on {len(candidates)} candidates, target: {top_n}")
    
    try:
        model = get_embedding_model()
    except Exception as e:
        print(f"[ERROR] Cannot perform fast filter: embedding model loading failed: {e}")
        model = "lightweight_fallback"
        
    role_text = get_role_text(role)
    
    if model == "lightweight_fallback":
        print("[Embedding Engine] Running fast filter in lightweight keyword-overlap fallback mode.")
        scored_candidates = []
        for cand in candidates:
            cand_text = get_candidate_text(cand)
            score = get_keyword_overlap_score(role_text, cand_text)
            cand_copy = dict(cand)
            cand_copy["fast_filter_score"] = float(score)
            scored_candidates.append(cand_copy)
        scored_candidates.sort(key=lambda x: x.get("fast_filter_score", 0.0), reverse=True)
        print(f"[Embedding Engine] Fast filtering (lightweight) complete. Top score: {scored_candidates[0]['fast_filter_score'] if scored_candidates else 0.0:.4f}")
        return scored_candidates[:top_n]
        
    # Generate role text and role embedding
    try:
        role_embedding = model.encode(role_text, convert_to_numpy=True)
    except Exception as e:
        print(f"[ERROR] Embedding generation failed for role: {e}")
        return candidates[:top_n]
        
    scored_candidates = []
    for cand in candidates:
        cand_text = get_candidate_text(cand)
        try:
            cand_embedding = model.encode(cand_text, convert_to_numpy=True)
            score = cosine_similarity(role_embedding, cand_embedding)
        except Exception as e:
            print(f"[ERROR] Embedding generation/similarity failed for candidate {cand.get('name', 'Unknown')}: {e}")
            score = 0.0
            
        # Add score to candidate dict copy to avoid side-effects
        cand_copy = dict(cand)
        cand_copy["fast_filter_score"] = float(score)
        scored_candidates.append(cand_copy)
        
    # Sort candidates by fast_filter_score descending
    scored_candidates.sort(key=lambda x: x.get("fast_filter_score", 0.0), reverse=True)
    
    print(f"[Embedding Engine] Fast filtering complete. Top score: {scored_candidates[0]['fast_filter_score'] if scored_candidates else 0.0:.4f}")
    return scored_candidates[:top_n]


# Self-Test block
if __name__ == "__main__":
    print("[Embedding Engine] Self-Test Running...")
    
    mock_role = {
        "role_title": "Senior Python SDE",
        "seniority_band": "senior",
        "domain": "finance",
        "hard_requirements": ["Python", "AWS", "SQL"],
        "role_persona": {"archetype": "Systems Builder", "environment": "startup"}
    }
    
    mock_candidates = [
        {
            "name": "Alice Developer",
            "narrative_arc": "generalist_builder",
            "candidate_twin": {"archetype": "Systems Builder", "environment_fit": "startup"},
            "timeline": [
                {"company": "FintechCorp", "title": "Senior Python SDE", "skills_gained": ["Python", "AWS", "PostgreSQL"]}
            ],
            "achievement_signals": [{"text": "Scaled DB write throughput by 200%"}],
            "original_profile": {"skills": ["Python", "AWS", "SQL"]}
        },
        {
            "name": "Bob Designer",
            "narrative_arc": "specialist",
            "candidate_twin": {"archetype": "UI Designer", "environment_fit": "agency"},
            "timeline": [
                {"company": "WebStudio", "title": "Lead UI Designer", "skills_gained": ["Figma", "CSS", "UI/UX"]}
            ],
            "achievement_signals": [{"text": "Redesigned landing pages increasing conversion by 20%"}],
            "original_profile": {"skills": ["Figma", "CSS"]}
        }
    ]
    
    try:
        ranked = fast_filter(mock_role, mock_candidates, top_n=2)
        print(f"[Embedding Engine] Self-Test Ranked Outputs:")
        for r in ranked:
            print(f" - {r['name']}: {r['fast_filter_score']:.4f}")
        assert len(ranked) == 2
        # Alice should score higher than Bob because she is a Python SDE, matching the role
        assert ranked[0]["name"] == "Alice Developer"
        assert ranked[0]["fast_filter_score"] > ranked[1]["fast_filter_score"]
        print("[Embedding Engine] Self-Test Passed!")
    except Exception as err:
        print(f"[Embedding Engine] Self-Test raised unexpected error: {err}")
