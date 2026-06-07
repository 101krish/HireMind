from typing import List, Dict, Optional, Any, Type, TypeVar
from pydantic import BaseModel, Field, ValidationError

T = TypeVar('T', bound=BaseModel)

def safe_validate(data: dict, model_class: Type[T]) -> Optional[dict]:
    """
    Validates a data dictionary against a Pydantic model class safely.
    Returns the validated model dump as a dictionary, or None if validation fails.
    """
    try:
        return model_class(**data).model_dump()
    except ValidationError as e:
        print(f"[Validation error] {e}")
        return None


# ==========================================
# 1. CHALLENGE INPUT SCHEMAS (Section 3.2 / candidate_schema.json)
# ==========================================

class ProfileInfo(BaseModel):
    """General candidate profile metadata."""
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

class CareerHistoryStint(BaseModel):
    """A single job stint in the candidate's career history."""
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

class EducationChallenge(BaseModel):
    """Educational details matching the challenge schema."""
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str  # tier_1, tier_2, tier_3, tier_4, unknown

class SkillChallenge(BaseModel):
    """Raw skill endorsements and duration metadata."""
    name: str
    proficiency: str  # beginner, intermediate, advanced, expert
    endorsements: int
    duration_months: Optional[int] = 0

class CertificationChallenge(BaseModel):
    """Candidate certifications."""
    name: str
    issuer: str
    year: int

class LanguageChallenge(BaseModel):
    """Candidate languages spoken."""
    language: str
    proficiency: str  # basic, conversational, professional, native

class SalaryRange(BaseModel):
    """Min/Max expected salary range."""
    min: float
    max: float

class RedrobSignals(BaseModel):
    """Platform activity, engagement, and availability signals from Redrob."""
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str  # remote, hybrid, onsite, flexible
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

class CandidateProfile(BaseModel):
    """The full challenge candidate profile schema from candidate_schema.json."""
    candidate_id: str = Field(..., pattern=r"^CAND_[0-9]{7}$")
    profile: ProfileInfo
    career_history: List[CareerHistoryStint] = Field(default_factory=list)
    education: List[EducationChallenge] = Field(default_factory=list)
    skills: List[SkillChallenge] = Field(default_factory=list)
    certifications: List[CertificationChallenge] = Field(default_factory=list)
    languages: List[LanguageChallenge] = Field(default_factory=list)
    redrob_signals: RedrobSignals

class JobDescriptionInput(BaseModel):
    """The raw job description input model."""
    id: str
    raw_text: str
    title: str
    company: Optional[str] = ""
    domain: Optional[str] = ""


# ==========================================
# 2. PROCESSED / DECOMPOSED SCHEMAS
# ==========================================

class RolePersona(BaseModel):
    """The behavioral and situational persona extracted for the role."""
    archetype: str
    environment: str
    execution_expectation: str
    leadership_expected: str
    learning_required: str

class RoleObject(BaseModel):
    """The fully decomposed role requirements parsed from the Job Description."""
    jd_id: str
    role_title: str
    seniority_band: str
    domain: str
    hard_requirements: List[str] = Field(default_factory=list)
    soft_requirements: List[str] = Field(default_factory=list)
    implied_needs: List[str] = Field(default_factory=list)
    culture_signals: List[str] = Field(default_factory=list)
    capability_needs: Dict[str, str] = Field(default_factory=dict)
    role_persona: RolePersona
    contradictions_detected: List[str] = Field(default_factory=list)
    inflation_adjusted: bool = False
    inflation_notes: List[str] = Field(default_factory=list)
    disqualifiers: List[str] = Field(default_factory=list)  # Disqualifier keywords matching reweighting rules

class TimelineEvent(BaseModel):
    """An event reconstructed in the candidate's career timeline."""
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    company: Optional[str] = None
    title: Optional[str] = None
    company_size: Optional[str] = None
    domain: Optional[str] = None
    skills_demonstrated: Optional[List[str]] = Field(default_factory=list)
    complexity_level: Optional[int] = None
    ownership_signals: Optional[List[str]] = Field(default_factory=list)
    metrics: Optional[List[str]] = Field(default_factory=list)
    promoted: Optional[bool] = None
    months_to_promote: Optional[int] = None
    year: Optional[int] = None
    event: Optional[str] = None
    skills_gained: Optional[List[str]] = Field(default_factory=list)

class CapabilityAssessment(BaseModel):
    """Detailed evaluation of a single capability domain."""
    level: str
    evidence_tier: int
    evidence: str
    confidence: float
    source: Optional[str] = "explicit"

class AchievementSignal(BaseModel):
    """A quantified metric or strong performance indicator from work history."""
    text: str
    type: str
    strength: str

class CandidateTwin(BaseModel):
    """The digital twin matching descriptor of a candidate."""
    archetype: str
    risk_profile: str
    learning_speed: str
    execution_style: str
    environment_fit: str
    leadership: str
    ceiling: str

class StructuredCandidate(BaseModel):
    """The fully structured representation of a decomposed candidate profile."""
    candidate_id: str
    name: str
    timeline: List[TimelineEvent] = Field(default_factory=list)
    capability_map: Dict[str, CapabilityAssessment] = Field(default_factory=dict)
    achievement_signals: List[AchievementSignal] = Field(default_factory=list)
    ownership_language_score: float = 0.5
    leadership_signals: List[str] = Field(default_factory=list)
    communication_signals: List[str] = Field(default_factory=list)
    narrative_arc: str = "generalist_builder"
    candidate_twin: CandidateTwin


# ==========================================
# 3. SCORING & OUTPUT SCHEMAS
# ==========================================

class ScoreWeight(BaseModel):
    """A sub-score and its relative weight in the overall calculation."""
    score: float
    weight: float

class ScoreResult(BaseModel):
    """The output of an individual scoring engine module."""
    score: float
    rationale: List[str] = Field(default_factory=list)

class ConfidenceBand(BaseModel):
    """The estimated score band with margin of error."""
    score: float
    margin: float

class DebateScores(BaseModel):
    """Consensus and individual agent scores from the debate panel."""
    technical_agent: float
    manager_agent: float
    growth_agent: float
    consensus: float

class InterviewQuestions(BaseModel):
    """Targeted questions generated for different stages of candidate assessment."""
    technical: List[str] = Field(default_factory=list)
    gap_probing: List[str] = Field(default_factory=list)
    behavioral: List[str] = Field(default_factory=list)

class DevilsAdvocate(BaseModel):
    """Devil's Advocate risk evaluation and mitigation strategy."""
    risks: List[str] = Field(default_factory=list)
    risk_level: str = "low"
    mitigation: str = ""

class ScoredCandidate(BaseModel):
    """The final fully-evaluated candidate profile containing scores, multiplier, and honeypot flags."""
    candidate_id: str
    name: str
    rank: int
    tier: str
    is_honeypot: bool = False
    honeypot_flags: List[str] = Field(default_factory=list)
    scores: Dict[str, ScoreWeight] = Field(default_factory=dict)
    behavioral_signal_multiplier: float = 1.0
    availability_score: float = 0.0
    engagement_score: float = 0.0
    weighted_score: float
    twin_match_score: float
    current_fit: float
    future_fit: float
    confidence_band: ConfidenceBand
    debate_scores: Optional[DebateScores] = None
    final_score_before_multiplier: float
    final_score: float
    reasoning: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    hidden_talents: List[str] = Field(default_factory=list)
    interview_questions: InterviewQuestions
    devils_advocate: Optional[DevilsAdvocate] = None
    recruiter_verdict: str


# Self-Test block
if __name__ == "__main__":
    print("[Schemas] Self-Test Running...")
    
    # Test safe validation with basic challenge structures
    mock_salary = {"min": 10.0, "max": 20.0}
    validated_sal = safe_validate(mock_salary, SalaryRange)
    assert validated_sal is not None
    assert validated_sal["min"] == 10.0
    
    print("[Schemas] Self-Test Completed Successfully!")
