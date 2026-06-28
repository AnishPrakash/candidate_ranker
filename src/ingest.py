import json
from typing import Iterator, Dict, Any

def stream_candidates(filepath: str) -> Iterator[Dict[Any, Any]]:
    """
    Yields one candidate dictionary at a time from a JSONL file.
    Keeps memory footprint flat regardless of dataset size.
    
    Args:
        filepath (str): Path to the candidates.jsonl file
        
    Yields:
        dict: Parsed JSON object for a single candidate
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

if __name__ == "__main__":
    # Quick sanity check - only runs if executed directly
    try:
        count = 0
        for cand in stream_candidates("T:/Projects/RedRob/candidate_ranker/candidates.jsonl"):
            count += 1
            if count == 1:
                print(f"Successfully read first candidate: {cand.get('candidate_id', 'Unknown')}")
                break
    except FileNotFoundError:
        print("Dataset not found locally yet, but the parser is ready.")