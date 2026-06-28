import argparse
import pandas as pd
import numpy as np
import faiss
import pickle
import xgboost as xgb
import csv
import json
from src.reasoner import generate_reasoning
from sentence_transformers import SentenceTransformer

# We will implement this in Phase 8. For now, a placeholder.
def generate_reasoning_placeholder(cid, score):
    return f"Candidate {cid} showed strong signals with a score of {score:.2f}."

def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60):
    scores = {}
    for rank, cid in enumerate(dense_ranks):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, cid in enumerate(sparse_ranks):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def run_ranking(candidates_file, output_file):
    print("Loading artifacts...")
    df = pd.read_parquet("artifacts/candidate_features.parquet")
    df.set_index('candidate_id', inplace=True)
    
    faiss_index = faiss.read_index("artifacts/faiss_index.bin")
    with open("artifacts/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
        
    ltr_model = xgb.Booster()
    ltr_model.load_model("artifacts/ltr_model.xgb")
    
    jd_vector = np.load("artifacts/jd_vector.npy").reshape(1, -1)
    faiss.normalize_L2(jd_vector)
    
    print("Running FAISS search...")
    # Get top 2000 indices
    _, dense_indices = faiss_index.search(jd_vector, 2000)
    dense_cids = df.iloc[dense_indices[0]].index.tolist()
    
    print("Running BM25 search...")
    # Get top 2000 by keyword
    jd_keywords = ["faiss", "pinecone", "qdrant", "weaviate", "ndcg", "mrr", "python", "ranking"]
    bm25_scores = bm25.get_scores(jd_keywords)
    sparse_indices = np.argsort(bm25_scores)[::-1][:2000]
    sparse_cids = df.iloc[sparse_indices].index.tolist()
    
    print("Applying Reciprocal Rank Fusion...")
    fused_cids = reciprocal_rank_fusion(dense_cids, sparse_cids)[:1000]
    
    print("Filtering and Re-scoring...")
    # Step 7.1.3 & 7.1.4 Apply Filters
    filtered_cids = []
    for cid in fused_cids:
        row = df.loc[cid]
        if row['is_honeypot'] or row['career_company_type_score'] == 0 or row['years_experience'] < 3 or row['availability_index'] < 0.05:
            continue
        filtered_cids.append(cid)
    
    # Step 7.1.5 LTR Re-scoring
    features_for_ltr = ['years_experience', 'availability_index', 'market_validation', 
                        'location_score', 'notice_period_modifier', 'skill_authenticity_score', 
                        'github_score', 'open_to_work', 'career_company_type_score']
    
    candidates_to_score = df.loc[filtered_cids]
    X_score = xgb.DMatrix(candidates_to_score[features_for_ltr].astype(float))
    ltr_scores = ltr_model.predict(X_score)
    
    # Tie breaking logic
    final_scores = ltr_scores + 0.0001 * candidates_to_score['github_score'].values
    
    # Pair IDs with scores and sort
    # Pair IDs with scores, round them to 4 decimals FIRST, then sort
    results = [(cid, round(float(score), 4)) for cid, score in zip(filtered_cids, final_scores)]
    # Sort by rounded score descending (-x[1]), then by candidate_id ascending (x[0])
    results.sort(key=lambda x: (-x[1], x[0]))
    top_100 = results[:100]
    
    print("Writing output and generating reasoning...")
    
    # We need the raw candidate dicts to generate reasoning, so we extract just the top 100 from the JSONL
    top_100_cids = {cid for cid, _ in top_100}
    raw_candidates = {}
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            if cand['candidate_id'] in top_100_cids:
                raw_candidates[cand['candidate_id']] = cand
                if len(raw_candidates) == 100:
                    break # Stop reading early once we have our 100

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        
        for rank, (cid, score) in enumerate(top_100, start=1):
            raw_cand = raw_candidates.get(cid, {'profile': {}, 'redrob_signals': {}})
            feature_row = df.loc[cid].to_dict()
            
            reasoning_str = generate_reasoning(raw_cand, feature_row, rank)
            writer.writerow([cid, rank, round(score, 4), reasoning_str])
            
    print(f"Ranking complete! Saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    
    run_ranking(args.candidates, args.out)