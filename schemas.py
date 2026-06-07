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
# 1. INPUT SCHEMAS
# ==========================================

class Education(BaseModel):
    """Represents a candidate's educational qualification."""
    degree: str
    institution: str
    year: Optional[int] = None

class WorkHistory(BaseModel):
    """Represents a single position/stint in the candidate's career history."""
    company: str
    title: str
    start_year: int
    end_year: Optional[int] = None
    is_current: Optional[bool] = False
    description: Optional[str] = ""
    company_size: Optional[str] = "unknown"
    domain: Optional[str] = ""

class Project(BaseModel):
    """Represents an engineering project completed by the candidate."""
    name: str
    description: Optional[str] = ""
    tech_stack: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)

class CandidateProfile(BaseModel):
    """The raw candidate profile as input to the Profile Engine."""
    id: str
    name: str
    current_title: Optional[str] = ""
    total_experience_years: Optional[float] = 0.0
    education: List[Education] = Field(default_factory=list)
    work_history: List[WorkHistory] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    platforms: Optional[Dict[str, str]] = Field(default_factory=dict)

class JobDescriptionInput(BaseModel):
    """The raw job description as input to the JD Engine."""
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
    """The final fully-evaluated candidate profile containing scores and agent consensus."""
    candidate_id: str
    name: str
    rank: int
    tier: str
    scores: Dict[str, ScoreWeight] = Field(default_factory=dict)
    weighted_score: float
    twin_match_score: float
    current_fit: float
    future_fit: float
    confidence_band: ConfidenceBand
    debate_scores: Optional[DebateScores] = None
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
    
    # Verify validation helper with mock data
    mock_edu = {"degree": "CS", "institution": "IIT", "year": 2020}
    validated_edu = safe_validate(mock_edu, Education)
    assert validated_edu is not None
    assert validated_edu["degree"] == "CS"
    
    mock_bad_edu = {"degree": 12345}  # missing institution
    assert safe_validate(mock_bad_edu, Education) is None
    
    print("[Schemas] Self-Test Completed Successfully!")
