import math
from datetime import date

def availability_index(signals: dict, reference_date: date = date(2026, 6, 28)) -> float:
    """
    Returns float 0.0–1.0 representing hiring availability.
    A multiplicative score where low engagement collapses the overall score.
    """
    # Fallback to a very old date if last_active_date is missing
    last_active_str = signals.get('last_active_date', '2020-01-01')
    last_active = date.fromisoformat(last_active_str)
    delta_days = (reference_date - last_active).days
    
    R = signals.get('recruiter_response_rate', 0.0)   # 0.0–1.0
    lambda_decay = 0.01                               # Decay constant
    
    A = R * math.exp(-lambda_decay * delta_days)
    
    if not signals.get('open_to_work_flag', False):
        A *= 0.4   # Heavy penalty if not actively looking
        
    if signals.get('avg_response_time_hours', 0) > 72:
        A *= 0.8   # Slow responders get a mild penalty
        
    return min(max(A, 0.0), 1.0)


def market_validation_index(signals: dict) -> float:
    """
    Returns float 0.0–1.0 based on collaborative filtering signals.
    High saves and views indicate a strong profile regardless of text matching.
    """
    saved = signals.get('saved_by_recruiters_30d', 0)
    views = signals.get('profile_views_received_30d', 0)
    interview_rate = signals.get('interview_completion_rate', 0.0)
    github = signals.get('github_activity_score', -1)  # -1 means no GitHub
    
    # Normalize each component 0–1
    saved_norm = min(saved / 20.0, 1.0)    # 20+ saves = max score
    views_norm = min(views / 100.0, 1.0)   # 100+ views = max score
    
    github_norm = max(github, 0) / 100.0   # -1 maps to 0
    
    M = (0.3 * saved_norm + 0.2 * views_norm + 
         0.3 * interview_rate + 0.2 * github_norm)
    
    return min(max(M, 0.0), 1.0)


def location_score(profile: dict, signals: dict) -> float:
    """
    Scores geographical fitness based on JD constraints.
    Prefers Pune/Noida, accepts other major Indian hubs with relocation.
    """
    willing = signals.get('willing_to_relocate', False)
    
    if profile.get('country', '') == 'India':
        city = profile.get('location', '').lower()
        if any(c in city for c in ['pune', 'noida', 'delhi', 'gurugram', 'gurgaon']):
            return 1.0
        elif any(c in city for c in ['mumbai', 'hyderabad', 'bengaluru', 'bangalore']):
            return 0.9 if willing else 0.7
        else:
            return 0.8 if willing else 0.5
    else:
        return 0.3 if willing else 0.1


def notice_period_modifier(notice_days: int) -> float:
    """
    Penalizes candidates with excessive notice periods.
    """
    if notice_days <= 30:   
        return 1.0
    elif notice_days <= 60:  
        return 0.9
    elif notice_days <= 90:  
        return 0.75
    else:                    
        return 0.5   # 90+ days is a serious penalty