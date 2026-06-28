import pandas as pd
import numpy as np
import xgboost as xgb

def generate_synthetic_labels(features_df):
    """
    Generates synthetic target scores for Learning-to-Rank based on JD rules.
    """
    print("Generating synthetic LTR training data...")
    labels = []
    
    for _, row in features_df.iterrows():
        score = 0.0
        
        # Hard constraint: Honeypots get 0 relevance (completely irrelevant)
        if row['is_honeypot']:
            labels.append(0)
            continue
            
        # Additive scoring for synthetic labels
        if row['open_to_work']: 
            score += 2.0
        if 5 <= row['years_experience'] <= 9: 
            score += 3.0
        if row['career_company_type_score'] > 0.5: 
            score += 2.0
            
        score += float(row['availability_index']) * 2.0
        score += float(row['market_validation']) * 1.5
        score += float(row['skill_authenticity_score']) * 0.5
        
        # Convert to a positive integer (e.g., 0, 1, 2, 3...) for XGBoost rank:ndcg
        final_label = max(0, int(round(score)))
        labels.append(final_label)
        
    return np.array(labels)

def train_ltr_model(parquet_path="artifacts/candidate_features.parquet", model_out="artifacts/ltr_model.xgb"):
    print(f"Loading features from {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print("Error: Parquet file not found. Please run precompute.py first (Phase 5).")
        return
        
    labels = generate_synthetic_labels(df)
    
    # Ensure exact feature order that rank.py expects
    features_for_ltr = [
        'years_experience', 'availability_index', 'market_validation', 
        'location_score', 'notice_period_modifier', 'skill_authenticity_score', 
        'github_score', 'open_to_work', 'career_company_type_score'
    ]
    
    X = df[features_for_ltr].astype(float)
    
    print("Training XGBoost LambdaMART model...")
    dtrain = xgb.DMatrix(X, label=labels)
    
    # Rank:NDCG optimizes the tree specifically for top-K ranking
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@10',
        'learning_rate': 0.1,
        'max_depth': 6,
        'tree_method': 'hist',  # Fast CPU training
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    model.save_model(model_out)
    
    print(f"Phase 6 LTR training complete! Model saved to {model_out}.")

if __name__ == "__main__":
    train_ltr_model()