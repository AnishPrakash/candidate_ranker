import math

# Proficiency multiplier (alpha)
PROFICIENCY_MAP = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2
}

SKILL_CATEGORY_KEYWORDS = {
    "embedding_retrieval": ["embedding", "vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "sentence-transformer", "dense retrieval", "hybrid search", "elasticsearch", "opensearch"],
    "ml_production": ["mlflow", "model serving", "inference", "production", "a/b test", "feature store", "model deployment", "ranking", "recommendation", "retrieval"],
    "python_engineering": ["python", "fastapi", "flask", "django", "asyncio", "pydantic"],
    "llm_work": ["llm", "fine-tuning", "lora", "qlora", "peft", "rag", "langchain", "openai", "huggingface"],
    "evaluation_frameworks": ["ndcg", "mrr", "map", "precision@", "recall", "offline eval", "a/b testing"]
}

def calculate_authenticity_score(candidate: dict) -> float:
    """
    Calculates the Skill Authenticity Score based on career history corroboration.
    Penalizes keyword stuffers who claim skills without job experience.
    """
    skills = candidate.get('skills', [])
    career_history = candidate.get('career_history', [])
    
    total_authenticity_score = 0.0
    
    for skill in skills:
        skill_name = skill.get('name', '').lower()
        claimed_duration = skill.get('duration_months', 1) # default to 1 to avoid ZeroDivision
        if claimed_duration == 0:
            claimed_duration = 1
            
        proficiency = skill.get('proficiency', 'beginner').lower()
        alpha = PROFICIENCY_MAP.get(proficiency, 0.2)
        endorsements = skill.get('endorsements', 0)
        
        # Determine which categories this skill belongs to
        matched_categories = []
        for category, keywords in SKILL_CATEGORY_KEYWORDS.items():
            if any(kw in skill_name for kw in keywords):
                matched_categories.append(category)
                
        # If the skill isn't highly relevant to our JD constraints, skip heavy calculation
        if not matched_categories:
            continue
            
        # Calculate career match duration
        career_match_months = 0
        for role in career_history:
            desc = role.get('description', '').lower()
            title = role.get('title', '').lower()
            role_duration = role.get('duration_months', 0)
            
            # Check if role description/title contains keywords for the matched categories
            role_matches = False
            for category in matched_categories:
                for kw in SKILL_CATEGORY_KEYWORDS[category]:
                    if kw in desc or kw in title:
                        role_matches = True
                        break
                if role_matches:
                    break
                    
            if role_matches:
                career_match_months += role_duration
                
        # Calculate final weight for this specific skill
        ratio = min(1.0, career_match_months / claimed_duration)
        w_skill = alpha * ratio * math.log(1 + endorsements)
        
        total_authenticity_score += w_skill
        
    return total_authenticity_score