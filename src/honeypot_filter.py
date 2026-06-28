from datetime import datetime
from dateutil import parser

def evaluate_honeypot(candidate: dict) -> tuple[bool, float]:
    """
    Evaluates a candidate against 4 chronological and biological paradoxes.
    Returns (is_honeypot: bool, penalty_score: float)
    """
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    education = candidate.get('education', [])
    
    claimed_yoe = profile.get('years_of_experience', 0)
    flags = 0
    
    # Check 1: Experience timeline impossibility
    total_career_months = sum(role.get('duration_months', 0) for role in career)
    total_career_years = total_career_months / 12.0
    
    if total_career_years < (claimed_yoe - 2) or total_career_years > (claimed_yoe + 2):
        flags += 1

    # Check 2: Company founding / Biological paradox
    # Flag if any single role duration exceeds 40 years (480 months) or implies working before birth
    for role in career:
        if role.get('duration_months', 0) > 480: 
            flags += 1
            break

    # Check 3: Skill duration paradox
    total_advanced_skill_months = sum(
        skill.get('duration_months', 0) for skill in skills 
        if skill.get('proficiency', '').lower() in ['expert', 'advanced']
    )
    
    if total_advanced_skill_months > (claimed_yoe * 12 * 5):
        flags += 1

    # Check 4: Education timeline sanity
    if education and career:
        earliest_grad_year = min((ed.get('end_year', 9999) for ed in education), default=9999)
        
        # Sort career history to find the earliest start date
        valid_starts = []
        for role in career:
            try:
                # Handle YYYY-MM-DD or similar standard formats
                start_date = parser.parse(role.get('start_date', ''))
                valid_starts.append(start_date.year)
            except (ValueError, TypeError):
                continue
                
        if valid_starts and earliest_grad_year != 9999:
            earliest_job_year = min(valid_starts)
            # If they started a mid/senior role >5 years before their earliest graduation
            if earliest_job_year < (earliest_grad_year - 5):
                flags += 1

    is_honeypot = flags >= 2
    penalty = -999.0 if is_honeypot else 0.0
    
    return is_honeypot, penalty