import sys
from pathlib import Path

# Add project root to sys.path so we can run this module directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings

def calibrate_candidate_pool(candidates: list, role_weights: dict) -> list:
    """
    Standardizes raw scores across the candidate pool using min-max normalization.
    Assigns talent tiers, current/future fit, recruiter verdicts, and confidence bands.
    """
    if not candidates:
        return []
        
    print(f"[Calibration Agent] Calibrating score distribution for {len(candidates)} candidates...")
    
    # 1. Calculate raw weighted scores for all candidates
    raw_scores = []
    for cand in candidates:
        scores = cand.get("scores", {})
        # Calculate raw weighted score from the 6 core signals
        raw_weighted = sum(
            scores.get(sig, 50.0) * role_weights.get(sig, 0.0)
            for sig in role_weights
        )
        cand["raw_weighted_score"] = raw_weighted
        raw_scores.append(raw_weighted)
        
    # 2. Apply min-max normalization to raw scores to fit a standard talent pool distribution
    # Target range for calibrated weighted score: [55.0, 95.0]
    min_target = 55.0
    max_target = 95.0
    
    min_raw = min(raw_scores)
    max_raw = max(raw_scores)
    raw_range = max_raw - min_raw
    
    for cand in candidates:
        raw_val = cand["raw_weighted_score"]
        if raw_range > 0:
            calibrated_score = min_target + ((raw_val - min_raw) / raw_range) * (max_target - min_target)
        else:
            calibrated_score = raw_val  # No variance, keep original
            
        # Final scores before behavioral multiplier (behavioral multiplier is applied online)
        cand["weighted_score"] = round(calibrated_score, 2)
        
        # Calculate current_fit and future_fit from core scores
        scores = cand.get("scores", {})
        cap_fit = scores.get("capability_fit", 50.0)
        beh_fit = scores.get("behavioral", 50.0)
        ev_conf = scores.get("evidence_confidence", 50.0)
        traj = scores.get("trajectory", 50.0)
        imp = scores.get("impact", 50.0)
        ht = scores.get("hidden_talent", 50.0)
        
        current_fit = (cap_fit * 0.5) + (beh_fit * 0.3) + (ev_conf * 0.2)
        future_fit = (traj * 0.4) + (imp * 0.4) + (ht * 0.2)
        
        cand["current_fit"] = round(current_fit, 2)
        cand["future_fit"] = round(future_fit, 2)
        
        # Calculate confidence band
        margin = max(1.0, round(5.0 - (ev_conf / 25.0), 2))
        cand["confidence_band"] = {
            "score": round(calibrated_score, 2),
            "margin": margin
        }
        
        # Determine Talent Tier and Recruiter Verdict based on calibrated score
        if calibrated_score >= 85.0:
            tier = "A"
            verdict = "STRONG_SHORTLIST"
        elif calibrated_score >= 70.0:
            tier = "B"
            verdict = "SHORTLIST"
        elif calibrated_score >= 50.0:
            tier = "C"
            verdict = "REVIEW"
        else:
            tier = "D"
            verdict = "REJECT"
            
        cand["tier"] = tier
        cand["recruiter_verdict"] = verdict
        
        # Clean up temporary key
        if "raw_weighted_score" in cand:
            del cand["raw_weighted_score"]
            
    # Sort candidates by calibrated score descending
    candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
    
    # Assign ranks in calibrated order
    for idx, cand in enumerate(candidates):
        cand["rank"] = idx + 1
        
    print(f"[Calibration Agent] Score calibration completed. Top candidate score: {candidates[0]['weighted_score']}")
    return candidates

# Self-Test block
if __name__ == "__main__":
    print("[Calibration Agent] Running self-test...")
    
    mock_weights = {
        "capability_fit": 0.30,
        "trajectory": 0.15,
        "impact": 0.25,
        "evidence_confidence": 0.15,
        "hidden_talent": 0.05,
        "behavioral": 0.10
    }
    
    mock_pool = [
        {
            "candidate_id": "CAND_001",
            "name": "Alice Smith",
            "scores": {
                "capability_fit": 90, "trajectory": 80, "impact": 85,
                "evidence_confidence": 95, "hidden_talent": 60, "behavioral": 80
            }
        },
        {
            "candidate_id": "CAND_002",
            "name": "Bob Jones",
            "scores": {
                "capability_fit": 70, "trajectory": 60, "impact": 65,
                "evidence_confidence": 70, "hidden_talent": 50, "behavioral": 70
            }
        },
        {
            "candidate_id": "CAND_003",
            "name": "Charlie Brown",
            "scores": {
                "capability_fit": 50, "trajectory": 50, "impact": 50,
                "evidence_confidence": 50, "hidden_talent": 50, "behavioral": 50
            }
        }
    ]
    
    calibrated = calibrate_candidate_pool(mock_pool, mock_weights)
    
    # Verify outputs
    assert len(calibrated) == 3
    assert calibrated[0]["rank"] == 1
    assert calibrated[0]["tier"] == "A"
    assert calibrated[2]["tier"] == "C"
    assert "current_fit" in calibrated[0]
    assert "future_fit" in calibrated[0]
    assert "confidence_band" in calibrated[0]
    
    print("[Calibration Agent] Self-test passed successfully!")
