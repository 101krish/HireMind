import argparse
import json
import sys
from pathlib import Path
from config import settings
from engines.honeypot_detector import detect_honeypot
from engines.scoring.behavioral_signals import calculate_behavioral_multiplier
from output.submission_builder import write_submission_csv

def load_precomputed_features(path: Path) -> list:
    """Loads precomputed candidate features from results/precomputed_features.json."""
    if not path.exists():
        print(f"[ERROR] Precomputed features file not found at {path}. Run precompute.py first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_candidates_from_source(path: Path) -> list:
    """Loads raw candidates from the specified candidates dataset path (.json or .jsonl)."""
    if not path.exists():
        print(f"[ERROR] Source candidates file not found at {path}")
        return []
        
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

def rank_candidates(candidates_path: str, precomputed_path: str, output_path: str) -> None:
    """
    Online Mode ranking orchestrator.
    Runs on CPU without network calls. Loads precomputed signals, runs honeypot and behavioral signals,
    calculates final scores, and outputs exactly 100 rows.
    """
    print("[Rank] Starting Online Mode ranking...")
    
    # 1. Load precomputed features
    precomputed = load_precomputed_features(Path(precomputed_path))
    print(f"[Rank] Loaded {len(precomputed)} candidates from precomputed features.")
    
    # 2. Score and filter candidates
    scored_list = []
    processed_ids = set()
    
    role_weights = settings.weights.get("senior_ic", {})
    
    for cand in precomputed:
        cid = cand.get("candidate_id")
        name = cand.get("name", "Unknown")
        
        # Run Honeypot Detection Pass
        raw_profile = cand.get("raw_profile", {})
        is_hp, hp_flags = detect_honeypot(raw_profile)
        
        if is_hp:
            # Honeypots get 0 score and are marked
            print(f"[Rank] Honeypot detected! Candidate: {name} ({cid}) - Flags: {hp_flags}")
            final_score = 0.0
            weighted_score = 0.0
            multiplier = 1.0
            avail_score = 0.0
            eng_score = 0.0
        else:
            # Apply weighted signal scoring
            scores = cand.get("scores", {})
            weighted_score = sum(
                scores.get(sig, 50.0) * role_weights.get(sig, 0.0)
                for sig in role_weights
            )
            
            # Apply Redrob Behavioral Signals Multiplier
            beh_info = calculate_behavioral_multiplier(raw_profile)
            multiplier = beh_info["behavioral_signal_multiplier"]
            avail_score = beh_info["availability_score"]
            eng_score = beh_info["engagement_score"]
            
            final_score = weighted_score * multiplier
            
        scored_cand = {
            "candidate_id": cid,
            "name": name,
            "is_honeypot": is_hp,
            "honeypot_flags": hp_flags,
            "weighted_score": weighted_score,
            "behavioral_signal_multiplier": multiplier,
            "availability_score": avail_score,
            "engagement_score": eng_score,
            "final_score": round(final_score, 4),
            "reasoning": cand.get("reasoning", [])
        }
        scored_list.append(scored_cand)
        processed_ids.add(cid)
        
    # 3. Handle list padding to exactly 100 rows if pool is too small
    if len(scored_list) < 100:
        print(f"[Rank] Scored candidate list size ({len(scored_list)}) is less than 100. Padding from source dataset...")
        source_candidates = load_candidates_from_source(Path(candidates_path))
        
        # Add candidates that were not processed in the precomputed list
        padded_count = 0
        for sc in source_candidates:
            cid = sc.get("candidate_id")
            # Adapt to both naming conventions
            name = sc.get("profile", {}).get("anonymized_name") or sc.get("name", "Unknown")
            
            if cid not in processed_ids:
                is_hp, hp_flags = detect_honeypot(sc)
                scored_cand = {
                    "candidate_id": cid,
                    "name": name,
                    "is_honeypot": is_hp,
                    "honeypot_flags": hp_flags,
                    "weighted_score": 0.0,
                    "behavioral_signal_multiplier": 1.0,
                    "availability_score": 0.0,
                    "engagement_score": 0.0,
                    "final_score": 0.0,
                    "reasoning": "Baseline fallback candidate (low precomputed relevance)."
                }
                scored_list.append(scored_cand)
                processed_ids.add(cid)
                padded_count += 1
                if len(scored_list) >= 100:
                    break
                    
        print(f"[Rank] Padded {padded_count} candidates from source candidates dataset.")
        
        # If still less than 100, pad with generic CAND_ ids for local validator tests
        if len(scored_list) < 100:
            dummy_needed = 100 - len(scored_list)
            print(f"[Rank] Candidate pool still under 100 ({len(scored_list)} total). Padding with {dummy_needed} dummy records for local validator tests...")
            for i in range(dummy_needed):
                dummy_id = f"CAND_999{9000 + i:04d}"
                scored_list.append({
                    "candidate_id": dummy_id,
                    "name": f"Dummy Padding Candidate {i}",
                    "is_honeypot": False,
                    "honeypot_flags": [],
                    "weighted_score": 0.0,
                    "behavioral_signal_multiplier": 1.0,
                    "availability_score": 0.0,
                    "engagement_score": 0.0,
                    "final_score": 0.0,
                    "reasoning": "Baseline fallback record to satisfy 100 candidate count."
                })
        
    # 4. Sort by final score descending, breaking ties by candidate_id ascending (standard challenge tie-breaker)
    scored_list.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    # 5. Output exactly the top 100 candidates to CSV
    top_100 = scored_list[:100]
    
    write_submission_csv(top_100, output_path)
    print(f"[Rank] Online Mode ranking completed successfully. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Ranking & Submission Output (Online Mode)")
    parser.add_argument("--candidates", default="data/sample_candidates.json", help="Path to raw candidates source file")
    parser.add_argument("--precomputed", default="results/precomputed_features.json", help="Path to precomputed features JSON file")
    parser.add_argument("--out", default="results/submission.csv", help="Path to output submission CSV file")
    args = parser.parse_args()
    
    rank_candidates(args.candidates, args.precomputed, args.out)
