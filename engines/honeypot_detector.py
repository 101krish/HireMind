import json
from pathlib import Path
from datetime import datetime

# Standard founding years for companies appearing in the Redrob challenge dataset
COMPANY_FOUNDING_YEARS = {
    "CRED": 2018,
    "Razorpay": 2014,
    "Swiggy": 2014,
    "Flipkart": 2007,
    "Zomato": 2008,
    "Uber": 2009,
    "Ola": 2010,
    "Pied Piper": 2014,
    "Hooli": 1997,
    "Dunder Mifflin": 1949,
    "Stark Industries": 1939,
    "Initech": 1999,
    "Wayne Enterprises": 1939,
    "Acme Corp": 1920,
    "Globex Inc": 1996,
    "Mindtree": 1999,
    "Wipro": 1945,
    "TCS": 1968,
    "Infosys": 1981,
    "Cognizant": 1994,
    "HCL": 1976,
    "Tech Mahindra": 1986,
    "Mad Street Den": 2013
}

def detect_honeypot(candidate: dict) -> tuple:
    """
    Evaluates a candidate profile against 5 impossible profile (honeypot) rules.
    Returns:
        (is_honeypot: bool, flags: list[str])
    """
    flags = []
    
    # 1. Extract experience and education
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience") or candidate.get("total_experience_years") or 0.0
    total_exp_months = years_exp * 12
    
    # Rule 1 — Experience inflation:
    # duration_months at a company > company's possible age or start_date is before founding year
    for job in candidate.get("career_history", []):
        company = job.get("company", "")
        start_date_str = job.get("start_date", "")
        duration_months = job.get("duration_months") or job.get("duration") or 0
        
        # Check start date against founding year
        if start_date_str and company in COMPANY_FOUNDING_YEARS:
            try:
                start_year = datetime.strptime(start_date_str, "%Y-%m-%d").year
            except ValueError:
                try:
                    start_year = datetime.strptime(start_date_str, "%Y-%m").year
                except ValueError:
                    start_year = None
                    
            if start_year:
                founding_year = COMPANY_FOUNDING_YEARS[company]
                if start_year < founding_year:
                    flags.append(
                        f"Rule 1 Mismatch: Started at {company} in {start_year} before it was founded in {founding_year}"
                    )
                    
        # Check duration against possible age since founding to current year (2026)
        if company in COMPANY_FOUNDING_YEARS:
            max_possible_age_months = (2026 - COMPANY_FOUNDING_YEARS[company]) * 12
            if duration_months > max_possible_age_months:
                flags.append(
                    f"Rule 1 Mismatch: Job duration ({duration_months} months) at {company} exceeds its maximum possible age since founding ({max_possible_age_months} months)"
                )
                
    # Rule 2 — Skill duration fraud:
    # any skill duration_months > candidate's total_experience months
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        duration = skill.get("duration_months") or 0
        if duration > total_exp_months:
            flags.append(
                f"Rule 2 Mismatch: Skill '{name}' duration ({duration} months) exceeds total career experience ({total_exp_months:.1f} months)"
            )
            
    # Rule 3 — Expert with zero usage:
    # proficiency = "expert" AND duration_months = 0 (3+ occurrences)
    expert_zero_count = 0
    for skill in candidate.get("skills", []):
        prof = skill.get("proficiency", "")
        duration = skill.get("duration_months") or 0
        if prof == "expert" and duration == 0:
            expert_zero_count += 1
    if expert_zero_count >= 3:
        flags.append(
            f"Rule 3 Mismatch: Candidate has {expert_zero_count} skills listed as 'expert' with 0 months of experience (needs < 3)"
        )
        
    # Rule 4 — Experience vs graduation mismatch:
    # years_of_experience > (current_year - education.end_year + 2)
    current_year = 2026
    max_end_year = None
    for edu in candidate.get("education", []):
        end_year = edu.get("end_year") or edu.get("year")
        if end_year:
            if max_end_year is None or end_year > max_end_year:
                max_end_year = end_year
                
    if max_end_year is not None:
        max_valid_exp = current_year - max_end_year + 2
        if years_exp > max_valid_exp:
            flags.append(
                f"Rule 4 Mismatch: Career experience ({years_exp} years) exceeds time elapsed since graduation in {max_end_year} + 2 buffer years ({max_valid_exp} years)"
            )
            
    # Rule 5 — Assessment score vs proficiency mismatch:
    # skill_assessment_score < 30 AND proficiency = "expert" (3+ occurrences)
    signals = candidate.get("redrob_signals", {})
    scores = signals.get("skill_assessment_scores", {})
    assessment_mismatch_count = 0
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        score = scores.get(name)
        if score is not None and score < 30.0 and prof == "expert":
            assessment_mismatch_count += 1
            
    if assessment_mismatch_count >= 3:
        flags.append(
            f"Rule 5 Mismatch: Candidate has {assessment_mismatch_count} expert skills scoring < 30 in Redrob skill assessments (needs < 3)"
        )
        
    is_honeypot = len(flags) > 0
    return is_honeypot, flags

# Self-Test block
if __name__ == "__main__":
    print("[Honeypot Detector] Self-Test Running...")
    
    # Valid candidate test
    valid_cand = {
        "profile": {"years_of_experience": 5.0},
        "career_history": [{"company": "Swiggy", "start_date": "2020-01-01", "duration_months": 36}],
        "education": [{"end_year": 2019}],
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 60}],
        "redrob_signals": {"skill_assessment_scores": {"Python": 85.0}}
    }
    is_hp, flg = detect_honeypot(valid_cand)
    assert not is_hp
    
    # Mismatched candidate test (Rule 2)
    invalid_cand = {
        "profile": {"years_of_experience": 1.0},
        "skills": [{"name": "Java", "proficiency": "expert", "duration_months": 240}],
        "redrob_signals": {}
    }
    is_hp, flg = detect_honeypot(invalid_cand)
    assert is_hp
    assert any("Rule 2" in f for f in flg)
    
    print("[Honeypot Detector] Self-Test Passed!")
