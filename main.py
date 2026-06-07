import uuid
import json
import shutil
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import settings
from precompute import run_offline_pipeline
from rank import rank_candidates

app = FastAPI(
    title="PRISM API",
    description="AI-Powered Recruiter Intelligence & Scoring Machine API",
    version="1.1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Persistent status tracking helpers
def get_status_path(run_id: str) -> Path:
    return Path(f"results/{run_id}_status.json")

def get_features_path(run_id: str) -> Path:
    return Path(f"results/{run_id}_precomputed_features.json")

def get_submission_path(run_id: str) -> Path:
    return Path(f"results/{run_id}_submission.csv")

def save_run_status(run_id: str, status: str, error_msg: str = None, **kwargs):
    import os
    status_path = get_status_path(run_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_data = {"run_id": run_id, "status": status, **kwargs}
    if error_msg:
        status_data["error"] = error_msg
    # Merge with existing status if it exists, to preserve other keys
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                existing.update(status_data)
                status_data = existing
        except Exception:
            pass
            
    # Write to a temp file first, then replace atomically to prevent 0-byte reads
    temp_path = status_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)
        os.replace(temp_path, status_path)
    except Exception:
        # Fallback to direct write if atomic replace fails
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)

def read_run_status(run_id: str) -> dict:
    import time
    status_path = get_status_path(run_id)
    if not status_path.exists():
        return {"run_id": run_id, "status": "not_found"}
        
    # Retry loop to prevent concurrent read/write race conditions
    for attempt in range(5):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05) # Wait 50ms and retry
            
    # Final fallback if all retries fail
    return {"run_id": run_id, "status": "running", "step": "Updating status...", "percent": 50}

# Request Models
class PrecomputeRequest(BaseModel):
    candidates_path: str = "data/sample_candidates.json"
    jd_path: str = "data/job_description.docx"

# Background worker
def execute_pipeline_task(run_id: str, candidates_path: str, jd_path: str):
    save_run_status(run_id, "running", step="Initializing...", percent=0)
    
    def status_callback(step: str, percent: int, **kwargs):
        save_run_status(run_id, "running", step=step, percent=percent, **kwargs)
        
    try:
        features_out = str(get_features_path(run_id))
        run_offline_pipeline(candidates_path, jd_path, features_out, status_callback=status_callback)
        save_run_status(run_id, "completed", step="Analysis Complete.", percent=100)
    except Exception as e:
        print(f"[API Background Error] Run {run_id} failed: {e}")
        save_run_status(run_id, "failed", error_msg=str(e), step="Analysis Failed.", percent=100)

# Endpoints
# HTML Page serving routes
@app.get("/")
@app.get("/configure")
def serve_configure():
    return FileResponse("static/upload_configure.html")
    
@app.get("/dashboard")
def serve_dashboard(run_id: str = None):
    return FileResponse("static/main_dashboard.html")
    
@app.get("/candidate")
def serve_candidate(run_id: str = None, candidate_id: str = None):
    return FileResponse("static/candidate_detail.html")
    
@app.get("/report")
def serve_report(run_id: str = None):
    return FileResponse("static/shortlist_report.html")

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

class JDTextPayload(BaseModel):
    text: str
    
@app.post("/upload/candidates")
async def upload_candidates(file: UploadFile = File(...)):
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest_path = uploads_dir / f"candidates_{uuid.uuid4().hex}.json"
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"file_path": str(dest_path)}

@app.post("/upload/jd")
async def upload_jd(file: UploadFile = File(...)):
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix
    if ext.lower() not in [".docx", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Only .docx, .txt, or .md files are supported for job description.")
    dest_path = uploads_dir / f"jd_{uuid.uuid4().hex}{ext}"
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"file_path": str(dest_path)}

@app.post("/upload/jd-text")
def upload_jd_text(payload: JDTextPayload):
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest_path = uploads_dir / f"jd_{uuid.uuid4().hex}.txt"
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(payload.text)
    return {"file_path": str(dest_path)}

# Standard startup trigger
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
