import uuid
import json
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config import settings
from precompute import run_offline_pipeline
from rank import rank_candidates

app = FastAPI(
    title="PRISM API",
    description="AI-Powered Recruiter Intelligence & Scoring Machine API",
    version="1.1.0"
)

# Persistent status tracking helpers
def get_status_path(run_id: str) -> Path:
    return Path(f"results/{run_id}_status.json")

def get_features_path(run_id: str) -> Path:
    return Path(f"results/{run_id}_precomputed_features.json")

def get_submission_path(run_id: str) -> Path:
    return Path(f"results/{run_id}_submission.csv")

def save_run_status(run_id: str, status: str, error_msg: str = None):
    status_path = get_status_path(run_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_data = {"run_id": run_id, "status": status}
    if error_msg:
        status_data["error"] = error_msg
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

def read_run_status(run_id: str) -> dict:
    status_path = get_status_path(run_id)
    if not status_path.exists():
        return {"run_id": run_id, "status": "not_found"}
    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Request Models
class PrecomputeRequest(BaseModel):
    candidates_path: str = "data/sample_candidates.json"
    jd_path: str = "data/job_description.docx"

# Background worker
def execute_pipeline_task(run_id: str, candidates_path: str, jd_path: str):
    save_run_status(run_id, "running")
    try:
        features_out = str(get_features_path(run_id))
        run_offline_pipeline(candidates_path, jd_path, features_out)
        save_run_status(run_id, "completed")
    except Exception as e:
        print(f"[API Background Error] Run {run_id} failed: {e}")
        save_run_status(run_id, "failed", error_msg=str(e))

# Endpoints
@app.get("/")
@app.get("/health")
def health_check():
    """Health check route returning api state and models loaded."""
    return {
        "status": "healthy",
        "api_version": "1.1.0",
        "model_name": settings.model_name,
        "embedding_model": settings.embedding_model
    }

@app.post("/jobs/precompute", status_code=202)
def trigger_precomputation(req: PrecomputeRequest, background_tasks: BackgroundTasks):
    """
    Triggers the offline precomputation pipeline as a background task.
    Returns immediately with a run_id.
    """
    run_id = str(uuid.uuid4())
    
    # Verify paths exist before starting background task
    cand_p = Path(req.candidates_path)
    jd_p = Path(req.jd_path)
    if not cand_p.exists():
        raise HTTPException(status_code=400, detail=f"Candidates path does not exist: {req.candidates_path}")
    if not jd_p.exists():
        raise HTTPException(status_code=400, detail=f"Job description path does not exist: {req.jd_path}")
        
    save_run_status(run_id, "pending")
    
    background_tasks.add_task(
        execute_pipeline_task, 
        run_id, 
        req.candidates_path, 
        req.jd_path
    )
    
    return {"run_id": run_id, "status": "running"}

@app.get("/jobs/status/{run_id}")
def check_job_status(run_id: str):
    """Retrieves status of a candidate precomputation job."""
    status_info = read_run_status(run_id)
    if status_info["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Job run '{run_id}' not found.")
    return status_info

@app.get("/results/{run_id}/candidates")
def get_candidate_results(run_id: str):
    """Retrieves the list of fully scored and calibrated candidate profiles from a run."""
    status_info = read_run_status(run_id)
    if status_info["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Job run '{run_id}' not found.")
    if status_info["status"] != "completed":
        return {
            "run_id": run_id,
            "status": status_info["status"],
            "detail": "Results are not ready. Job is still running or failed."
        }
        
    features_file = get_features_path(run_id)
    if not features_file.exists():
        raise HTTPException(status_code=500, detail="Precomputed features file missing from disk.")
        
    with open(features_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/results/{run_id}/submission")
def download_submission_csv(run_id: str, candidates_path: str = "data/sample_candidates.json"):
    """
    Ranks the precomputed candidates online and returns the validator-ready submission CSV.
    Uses cached CSV if it was already generated for this run.
    """
    status_info = read_run_status(run_id)
    if status_info["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Job run '{run_id}' not found.")
    if status_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job run is not complete. Status: {status_info['status']}")
        
    csv_file = get_submission_path(run_id)
    if not csv_file.exists():
        features_file = get_features_path(run_id)
        if not features_file.exists():
            raise HTTPException(status_code=500, detail="Precomputed features missing from disk.")
        
        print(f"[API] Generating online ranking submission CSV for run {run_id}...")
        try:
            rank_candidates(
                candidates_path=candidates_path,
                precomputed_path=str(features_file),
                output_path=str(csv_file)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate submission: {e}")
            
    return FileResponse(
        path=csv_file,
        media_type="text/csv",
        filename=f"submission_{run_id}.csv"
    )

# Standard startup trigger
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
