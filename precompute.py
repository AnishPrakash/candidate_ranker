import os
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.ingest import stream_candidates
from src.honeypot_filter import evaluate_honeypot
from src.skill_authenticator import calculate_authenticity_score
from src.behavioral_signals import availability_index, market_validation_index, location_score, notice_period_modifier

JD_QUERY = """
Senior AI engineer with production experience deploying embeddings-based retrieval 
and ranking systems at product companies. Built vector search infrastructure using 
FAISS, Qdrant, Pinecone, Weaviate, Milvus, or Elasticsearch in production. 
Designed offline evaluation frameworks using NDCG, MAP, MRR, A/B testing. 
Strong Python engineering. Shipped recommendation or search systems to real users 
at scale. Experience with hybrid retrieval combining dense and sparse search. 
LLM fine-tuning experience a plus. Located in India, preferably Pune or Noida. 
5-9 years total experience, 4+ years in applied ML at product companies, not consulting.
"""

def candidate_to_text(c):
    parts = []
    for role in c.get('career_history', [])[:5]:
        parts.append(f"{role.get('title', '')} at {role.get('company', '')}: {role.get('description', '')}")
    parts.append(c.get('profile', {}).get('summary', ''))
    parts.append(c.get('profile', {}).get('headline', ''))
    return " ".join(parts)

def run_precompute(filepath="candidates.jsonl"):
    os.makedirs("artifacts", exist_ok=True)
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Embed JD Query
    jd_vector = embedder.encode([JD_QUERY])[0]
    np.save("artifacts/jd_vector.npy", jd_vector)
    
    candidates = list(stream_candidates(filepath))
    
    print("Processing candidates for dense and sparse retrieval...")
    bm25_corpus = []
    dense_texts = []
    tabular_data = []
    
    for c in tqdm(candidates, desc="Extracting features & text"):
        # BM25 text
        skills_text = " ".join([s.get('name', '').lower() for s in c.get('skills', [])])
        career_text = " ".join([r.get('description', '').lower() for r in c.get('career_history', [])])
        bm25_corpus.append((skills_text + " " + career_text).split())
        
        # Dense text
        dense_texts.append(candidate_to_text(c))
        
        # Tabular Features
        profile = c.get('profile', {})
        signals = c.get('redrob_signals', {})
        
        is_hp, hp_pen = evaluate_honeypot(c)
        auth_score = calculate_authenticity_score(c)
        
        career_desc = " ".join([r.get('industry', '').lower() for r in c.get('career_history', [])])
        company_type = 0.3 if 'consulting' in career_desc else 1.0
        
        tabular_data.append({
            'candidate_id': c['candidate_id'],
            'years_experience': profile.get('years_of_experience', 0),
            'availability_index': availability_index(signals),
            'market_validation': market_validation_index(signals),
            'location_score': location_score(profile, signals),
            'notice_period_modifier': notice_period_modifier(signals.get('notice_period_days', 999)),
            'skill_authenticity_score': auth_score,
            'is_honeypot': is_hp,
            'open_to_work': signals.get('open_to_work_flag', False),
            'github_score': max(signals.get('github_activity_score', 0), 0),
            'career_company_type_score': company_type
        })
    
    # 2. Save Tabular Data
    df = pd.DataFrame(tabular_data)
    df.to_parquet("artifacts/candidate_features.parquet", index=False)
    
    # 3. Build BM25 Index
    print("Building BM25 Index...")
    bm25 = BM25Okapi(bm25_corpus)
    with open("artifacts/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
        
    # 4. Build FAISS Index
    print("Encoding dense vectors (this takes a few minutes)...")
    embeddings = embedder.encode(dense_texts, batch_size=128, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    print("Building FAISS Index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, "artifacts/faiss_index.bin")
    
    print("Phase 5 pre-computation complete! Artifacts saved.")

if __name__ == "__main__":
    run_precompute()