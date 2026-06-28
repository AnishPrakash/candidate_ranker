def generate_reasoning(candidate: dict, feature_row: dict, rank: int) -> str:
    """
    Generates a deterministic, hallucination-free reasoning string 
    based strictly on candidate data and extracted features.
    Prioritizes highly specific signals over generic ones.
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    career_history = candidate.get('career_history', [])
    
    positives = []
    concerns = []
    
    # --- 1. EXTRACT POSITIVE SIGNALS ---
    
    yoe = profile.get('years_of_experience', 0)
    if 5 <= yoe <= 9:
        positives.append(f"{yoe:.1f} yrs experience (squarely in JD target band)")
        
    product_roles = [r for r in career_history 
                     if r.get('company_size') in ('11-50', '51-200', '201-500', '501-1000') 
                     and 'consulting' not in r.get('industry', '').lower()]
    if product_roles:
        most_recent = product_roles[0]
        positives.append(f"product-company experience ({most_recent.get('company', 'Unknown')})")
        
    key_skills_found = []
    for role in career_history:
        desc = role.get('description', '').lower()
        for skill_kw in ['faiss', 'vector', 'embedding', 'retrieval', 'ranking', 
                         'pinecone', 'qdrant', 'sentence-transformer', 'ndcg', 'elasticsearch']:
            if skill_kw in desc and skill_kw not in key_skills_found:
                key_skills_found.append(skill_kw)
    if key_skills_found:
        positives.append(f"demonstrated {', '.join(key_skills_found[:3])} in career history")

    # New Behavioral & Technical Signals
    github = signals.get('github_activity_score', 0)
    if github >= 70:
        positives.append(f"strong GitHub activity score ({github:.0f}/100)")
    elif github >= 40:
        positives.append(f"active GitHub contributor ({github:.0f}/100)")

    assessments = signals.get('skill_assessment_scores', {})
    if assessments:
        top_assessment = max(assessments.items(), key=lambda x: x[1])
        if top_assessment[1] >= 80:
            positives.append(f"platform-verified {top_assessment[0]} score ({top_assessment[1]:.0f}/100)")

    saved = signals.get('saved_by_recruiters_30d', 0)
    if saved >= 10:
        positives.append(f"high recruiter demand ({saved} saves in 30d)")
    elif saved >= 5:
        positives.append(f"strong recruiter interest ({saved} saves in 30d)")

    oar = signals.get('offer_acceptance_rate', 0)
    if oar >= 0.8:
        positives.append(f"high offer acceptance rate ({oar:.0%}) — serious candidate")

    icr = signals.get('interview_completion_rate', 0)
    if icr >= 0.9:
        positives.append(f"excellent interview follow-through ({icr:.0%} completion rate)")

    apps = signals.get('applications_submitted_30d', 0)
    if apps >= 5:
        positives.append(f"actively job-hunting ({apps} applications in 30d)")

    pcs = signals.get('profile_completeness_score', 0)
    if pcs >= 95:
        positives.append(f"fully complete profile ({pcs:.0f}%)")

    title = profile.get('current_title', '').lower()
    if any(t in title for t in ['senior', 'staff', 'lead', 'principal']):
        if any(t in title for t in ['ml', 'ai', 'machine learning', 'data science', 'nlp', 'research']):
            positives.append(f"title directly matches JD seniority ({profile.get('current_title')})")

    industry = profile.get('current_industry', '').lower()
    if industry in ['hr tech', 'recruiting', 'talent', 'marketplace']:
        positives.append(f"direct domain experience ({profile.get('current_industry')} — JD's exact vertical)")

    # --- 2. SORT POSITIVES BY SPECIFICITY ---
    priority_order = [
        'platform-verified', 'github', 'saves in 30d', 'offer acceptance',
        'interview follow', 'applications', 'domain experience', 'directly matches',
        'demonstrated', 'product-company', 'squarely in JD'
    ]
    
    def priority(p):
        for i, kw in enumerate(priority_order):
            if kw in p:
                return i
        return len(priority_order)
        
    positives = sorted(positives, key=priority)
    top2 = positives[:2] if positives else [profile.get('current_title', 'Candidate')]

    # --- 3. EXTRACT CONCERNS ---
    avail = feature_row.get('availability_index', 0)
    if avail < 0.2:
        concerns.append(f"low engagement ({signals.get('recruiter_response_rate', 0):.0%} response rate, last active {signals.get('last_active_date', 'Unknown')})")
        
    notice = signals.get('notice_period_days', 0)
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
        
    if profile.get('country') != 'India':
        concerns.append(f"based outside India ({profile.get('location', 'Unknown')}, {profile.get('country', 'Unknown')})")

    # --- 4. BUILD THE STRING WITH TONE ADJUSTMENT ---
    # Adjusted tone thresholds to reflect the score cliff Claude identified
    if rank <= 20:
        tone = "Strong fit"
    elif rank <= 29:
        tone = "Good fit"
    else:
        tone = "Marginal fit"
        
    reasoning = f"{tone}: {'; '.join(top2)}."
    if concerns:
        reasoning += f" Concern(s): {'; '.join(concerns[:2])}."

    # --- 5. RICHNESS FALLBACK ---
    if len(reasoning) < 80:
        reasoning = (f"{tone}: {profile.get('current_title', 'Unknown')}, {yoe:.1f} yrs; "
                     f"response rate {signals.get('recruiter_response_rate', 0):.0%}, "
                     f"notice {notice}d.")

    return reasoning