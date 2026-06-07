import sys
import json
import subprocess
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path so we can run this module directly
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from main import app, get_status_path, get_features_path, get_submission_path

def test_prism_api():
    print("=== STARTING PRISM API INTEGRATION TESTS ===")
    
    # 1. Initialize TestClient
    client = TestClient(app)
    
    # 2. Check Health
    print("[Test] Calling health check...")
    res = client.get("/health")
    assert res.status_code == 200
    health = res.json()
    assert health["status"] == "healthy"
    print(f"Health API Version: {health['api_version']}")
    
    # 3. Trigger Precomputation (Simulation fallback runs since API key is not live)
    print("[Test] Triggering precomputation job...")
    req_payload = {
        "candidates_path": "data/sample_candidates.json",
        "jd_path": "data/job_description.docx"
    }
    res = client.post("/jobs/precompute", json=req_payload)
    assert res.status_code == 202
    job_start = res.json()
    run_id = job_start["run_id"]
    print(f"Generated Run ID: {run_id}")
    assert job_start["status"] == "running"
    
    # Note: In TestClient, background tasks run synchronously during the request/response cycle,
    # so the job will already be completed when the client receives the response.
    
    # 4. Check status
    print("[Test] Checking job status...")
    res = client.get(f"/jobs/status/{run_id}")
    assert res.status_code == 200
    status_info = res.json()
    print(f"Job Status: {status_info['status']}")
    assert status_info["status"] == "completed"
    
    # 5. Retrieve Candidate Results
    print("[Test] Fetching candidate results...")
    res = client.get(f"/results/{run_id}/candidates")
    assert res.status_code == 200
    candidates = res.json()
    print(f"Retrieved {len(candidates)} candidates.")
    assert len(candidates) > 0
    
    # Verify candidate keys
    cand = candidates[0]
    required_keys = ["candidate_id", "name", "scores", "weighted_score", "twin_match_score", "debate_scores", "devils_advocate", "interview_questions"]
    for k in required_keys:
        assert k in cand, f"Candidate card missing key: '{k}'"
        
    # 6. Retrieve CSV Submission
    print("[Test] Fetching submission CSV...")
    res = client.get(f"/results/{run_id}/submission")
    assert res.status_code == 200
    assert res.headers["content-type"] == "text/csv; charset=utf-8"
    
    # Save CSV locally for validation
    test_csv_path = Path("results/test_api_submission.csv")
    test_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_csv_path, "wb") as f:
        f.write(res.content)
    print(f"Downloaded test submission CSV to {test_csv_path}")
    
    # 7. Run Challenge Validator
    print("[Test] Running challenge validator script...")
    val_res = subprocess.run(
        ["python", "data/validate_submission.py", str(test_csv_path)],
        capture_output=True,
        text=True
    )
    print(f"Validator Output: {val_res.stdout.strip()}")
    assert val_res.returncode == 0, f"Validator failed: {val_res.stderr}"
    assert "Submission is valid." in val_res.stdout
    
    # 8. Cleanup test files
    print("[Test] Cleaning up generated files...")
    status_p = get_status_path(run_id)
    features_p = get_features_path(run_id)
    sub_p = get_submission_path(run_id)
    
    if status_p.exists(): status_p.unlink()
    if features_p.exists(): features_p.unlink()
    if sub_p.exists(): sub_p.unlink()
    if test_csv_path.exists(): test_csv_path.unlink()
    
    print("=== PRISM API INTEGRATION TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    try:
        test_prism_api()
    except Exception as e:
        print(f"[ERROR] API test suite failed: {e}")
        sys.exit(1)
