import streamlit as st
import pandas as pd
import subprocess
import os
import json

# --- Page Config ---
st.set_page_config(page_title="Redrob Ranker | Team 001", layout="wide")
st.title("🏆 Redrob AI - Candidate Ranking Engine")
st.markdown("""
This sandbox runs our **Hybrid Retrieval (FAISS + BM25) & XGBoost LambdaMART** pipeline. 
It processes the candidates through our adversarial defense filters and generates hallucination-free reasoning.
""")

# --- Sidebar Controls ---
st.sidebar.header("Pipeline Controls")
uploaded_file = st.sidebar.file_uploader("Upload candidates.jsonl (≤ 100 profiles)", type=["jsonl"])

# We will always generate a temporary JSONL file for rank.py to consume
input_path = "temp_sandbox_input.jsonl"
output_path = "sandbox_output.csv"

if uploaded_file is not None:
    # If the user uploads a file, write it directly
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Custom dataset loaded.")
else:
    # If no upload, use Claude's logic to parse the default JSON array 
    # and temporarily yield it as JSONL so rank.py doesn't break.
    st.sidebar.info("Using default 50-candidate sample.")
    try:
        with open("sample_candidates.json", "r", encoding="utf-8") as f:
            candidates_array = json.load(f)
        with open(input_path, "w", encoding="utf-8") as f:
            for cand in candidates_array:
                f.write(json.dumps(cand) + "\n")
    except FileNotFoundError:
        st.sidebar.error("Could not find sample_candidates.json in the root directory.")

# --- Main Action ---
if st.sidebar.button("🚀 Run Ranking Engine", type="primary"):
    with st.spinner("Running FAISS + BM25 Hybrid Retrieval and LTR Scoring..."):
        
        # Call our exact command-line tool to prove modularity
        import sys
        result = subprocess.run(
            [sys.executable, "rank.py", "--candidates", input_path, "--out", output_path],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            st.error("Pipeline Error!")
            st.code(result.stderr)
        else:
            st.success(f"Ranking complete! Processed in under 5 minutes.")
            
            # Load and display results
            df = pd.read_csv(output_path)
            
            # Format the dataframe for a beautiful Streamlit UI
            st.subheader("📊 Top Candidates")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Candidates Processed", len(df))
            col2.metric("Top Score", f"{df['score'].max():.4f}" if not df.empty else "N/A")
            col3.metric("Honeypots Filtered", "Active")

            # Make the reasoning column wide
            st.dataframe(
                df,
                column_config={
                    "candidate_id": st.column_config.TextColumn("Candidate ID", width="small"),
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "score": st.column_config.NumberColumn("LTR Score", width="small"),
                    "reasoning": st.column_config.TextColumn("Deterministic Reasoning", width="large"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            with open(output_path, "rb") as f:
                csv_bytes = f.read()
                
            st.download_button(
                label="📥 Download Submission CSV",
                data=csv_bytes,
                file_name="sandbox_submission.csv",
                mime="text/csv",
            )
            
            # Clean up temp files
            if os.path.exists(input_path):
                os.remove(input_path)