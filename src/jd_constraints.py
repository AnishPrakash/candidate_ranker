"""
Structured constraints extracted manually from the Redrob AI JD.
These define the scoring weights and required skill matches for the ranking engine.
"""

JD_CONSTRAINTS = {
    "hard_requirements": {
        "embedding_retrieval": ["sentence-transformers", "bge", "e5", "openai embeddings"],
        "vector_db_hybrid": ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch"],
        "core_language": ["python"],
        "evaluation_frameworks": ["ndcg", "mrr", "map", "a/b testing", "offline eval"],
        "production_experience": ["shipped", "production", "real users"]
    },
    
    "preferred_skills": {
        "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning"],
        "learning_to_rank": ["xgboost", "neural ltr", "lambdamart"],
        "domain_experience": ["hr-tech", "recruiting tech", "marketplace"],
        "infrastructure": ["distributed systems", "large-scale inference"],
        "open_source": ["open-source", "oss", "contributions"]
    },
    
    "hard_disqualifiers": [
        "Career entirely in pure research/academic roles with zero production deployment",
        "AI experience only recent LangChain-wrapper projects with no pre-LLM era ML work",
        "Senior/Staff/Principal engineer who hasn't written code in 18+ months",
        "Entire career at Big 4 consulting firms with zero product company experience",
        "Primary expertise is computer vision, speech, or robotics with zero NLP/IR",
        "All work on closed-source systems 5+ years with no external validation"
    ],
    
    "target_profile": {
        "total_experience_years": {
            "min": 5, 
            "max": 9, 
            "sweet_spot_min": 6, 
            "sweet_spot_max": 8
        },
        "applied_ml_product_years": {
            "min": 4
        },
        "location": {
            "preferred": ["pune", "noida"],
            "acceptable_country": "India",
            "relocation_acceptable": True
        },
        "notice_period_days": {
            "ideal_max": 30,
            "acceptable_max": 90
        }
    }
}