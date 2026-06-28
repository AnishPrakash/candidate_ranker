def generate_reasoning(candidate: dict, feature_row: dict, rank: int) -> str:
    """
    Generates a deterministic, hallucination-free reasoning string 
    based strictly on candidate data and extracted features.
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    career_history = candidate.get('career_history', [])
    
    positives = []
    concerns = []
    
    # 1. Experience
    yoe = profile.get('years_of_experience', 0)
    if 5 <= yoe <= 9:
        positives.append(f"{yoe:.1f} yrs experience (squarely in JD target band)")
        
    # 2. Career company type
    product_roles = [r for r in career_history 
                     if r.get('company_size') in ('11-50', '51-200', '201-500', '501-1000') 
                     and 'consulting' not in r.get('industry', '').lower()]
    if product_roles:
        most_recent = product_roles[0]
        positives.append(f"product-company experience ({most_recent.get('company', 'Unknown')})")
        
    # 3. Relevant skills in career descriptions
    key_skills_found = []
    for role in career_history:
        desc = role.get('description', '').lower()
        for skill_kw in ['faiss', 'vector', 'embedding', 'retrieval', 'ranking', 
                         'pinecone', 'qdrant', 'sentence-transformer', 'ndcg', 'elasticsearch']:
            if skill_kw in desc and skill_kw not in key_skills_found:
                key_skills_found.append(skill_kw)
                
    if key_skills_found:
        positives.append(f"demonstrated {', '.join(key_skills_found[:3])} in career history")
        
    # 4. Availability
    avail = feature_row.get('availability_index', 0)
    if avail > 0.6:
        positives.append(f"highly responsive (response rate {signals.get('recruiter_response_rate', 0):.0%})")
    elif avail < 0.2:
        concerns.append(f"low engagement ({signals.get('recruiter_response_rate', 0):.0%} response rate, last active {signals.get('last_active_date', 'Unknown')})")
        
    # 5. Notice period
    notice = signals.get('notice_period_days', 0)
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
        
    # 6. Location
    if profile.get('country') != 'India':
        concerns.append(f"based outside India ({profile.get('location', 'Unknown')}, {profile.get('country', 'Unknown')})")
        
    # Build the final string
    pos_str = "; ".join(positives[:2]) if positives else profile.get('current_title', 'Candidate')
    
    if rank <= 20:
        tone = "Strong fit"
    elif rank <= 50:
        tone = "Good fit"
    else:
        tone = "Marginal fit"
        
    reasoning = f"{tone}: {pos_str}."
    if concerns:
        reasoning += f" Concern(s): {'; '.join(concerns[:2])}."
        
    return reasoning