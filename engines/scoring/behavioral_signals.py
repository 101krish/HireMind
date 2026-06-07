import json
from datetime import datetime
from config import settings

def calculate_behavioral_multiplier(candidate: dict, reference_date_str: str = "2026-06-07") -> dict:
    """
    Computes a score multiplier (0.50 to 1.15) based on candidate's platform activity signals.
    Returns:
        dict: {
            "behavioral_signal_multiplier": float,
            "availability_score": float,
            "engagement_score": float
        }
    """
    # Extract redrob_signals
    signals = candidate.get("redrob_signals") or candidate.get("original_profile", {}).get("redrob_signals", {})
    if not signals:
        # Fallback to neutral multiplier if no signals are available
        return {
            "behavioral_signal_multiplier": 1.0,
            "availability_score": 0.0,
            "engagement_score": 0.0
        }
        
    availability_adjustments = []
    engagement_adjustments = []
    
    # --- Availability Score Adjustments ---
    
    # 1. Open to work flag
    open_to_work = signals.get("open_to_work_flag", True)
    if not open_to_work:
        availability_adjustments.append(-0.20)
        
    # 2. Last active date > 6 months ago (180 days)
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
            ref_date = datetime.strptime(reference_date_str, "%Y-%m-%d")
            days_inactive = (ref_date - last_active).days
            if days_inactive > 180:
                availability_adjustments.append(-0.15)
        except ValueError:
            pass
            
    # 3. Notice period > 90 days
    notice_period = signals.get("notice_period_days", 0)
    if notice_period > 90:
        availability_adjustments.append(-0.05)
        
    # --- Engagement Score Adjustments ---
    
    # 4. Recruiter response rate < 0.10
    response_rate = signals.get("recruiter_response_rate", 1.0)
    if response_rate < 0.10:
        engagement_adjustments.append(-0.15)
        
    # 5. Interview completion rate < 0.50
    completion_rate = signals.get("interview_completion_rate", 1.0)
    if completion_rate < 0.50:
        engagement_adjustments.append(-0.10)
        
    # 6. GitHub activity score > 60 (JD-specific bonus)
    github_score = signals.get("github_activity_score", -1)
    if github_score > 60:
        engagement_adjustments.append(0.10)
        
    # 7. Verified email and verified phone
    email_ver = signals.get("verified_email", False)
    phone_ver = signals.get("verified_phone", False)
    if email_ver and phone_ver:
        engagement_adjustments.append(0.05)
        
    # Compute sums
    avail_score = sum(availability_adjustments)
    eng_score = sum(engagement_adjustments)
    
    # Calculate clamped final multiplier
    total_adjustments = avail_score + eng_score
    raw_multiplier = 1.0 + total_adjustments
    final_multiplier = max(0.50, min(1.15, raw_multiplier))
    
    return {
        "behavioral_signal_multiplier": round(final_multiplier, 4),
        "availability_score": round(avail_score, 4),
        "engagement_score": round(eng_score, 4)
    }

# Self-Test block
if __name__ == "__main__":
    print("[Behavioral Signals] Self-Test Running...")
    
    # Neutral profile test
    mock_candidate_neutral = {
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": "2026-05-20",
            "notice_period_days": 30,
            "recruiter_response_rate": 0.85,
            "interview_completion_rate": 0.90,
            "github_activity_score": 10.0,
            "verified_email": True,
            "verified_phone": True
        }
    }
    res = calculate_behavioral_multiplier(mock_candidate_neutral)
    # 1.0 + (0.05 for email+phone) = 1.05
    assert res["behavioral_signal_multiplier"] == 1.05
    assert res["availability_score"] == 0.0
    assert res["engagement_score"] == 0.05
    
    # Inactive & long notice candidate
    mock_candidate_inactive = {
        "redrob_signals": {
            "open_to_work_flag": False,          # -0.20
            "last_active_date": "2025-01-01",      # -0.15 (inactive over 6 months from 2026-06-07)
            "notice_period_days": 120,             # -0.05
            "recruiter_response_rate": 0.05,      # -0.15
            "interview_completion_rate": 0.30,    # -0.10
            "github_activity_score": 0.0,
            "verified_email": False,
            "verified_phone": False
        }
    }
    res_in = calculate_behavioral_multiplier(mock_candidate_inactive)
    # adjustments: -0.20 -0.15 -0.05 -0.15 -0.10 = -0.65. multiplier = 1.0 - 0.65 = 0.35 -> clamped to 0.50
    assert res_in["behavioral_signal_multiplier"] == 0.50
    assert res_in["availability_score"] == -0.40
    assert res_in["engagement_score"] == -0.25
    
    print("[Behavioral Signals] Self-Test Passed!")
