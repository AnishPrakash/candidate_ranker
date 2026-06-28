import os
import sys

# ============================================================
# CRITICAL FIX: Set CWD to project root before anything else.
# app.py lives at candidate_ranker/sandbox/app.py
# All files (rank.py, artifacts/, sample_candidates.json) live
# at candidate_ranker/ (the root). Without this fix, every
# relative path resolves to sandbox/ and silently fails.
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import subprocess
import json

# --- Page Config ---
st.set_page_config(page_title="Redrob Ranker | Team 001", layout="wide")

st.title("🏆 Redrob AI — Candidate Ranking Engine")
st.markdown("""
This sandbox demonstrates our **Hybrid Retrieval (FAISS + BM25) + XGBoost LambdaMART** pipeline.  
Candidates are passed through adversarial defense filters (honeypot detection, skill authenticator)  
before LTR re-scoring and deterministic reasoning generation.

> ℹ️ **Sandbox mode:** runs on the pre-loaded 50-candidate sample (or your uploaded file).  
> Output will contain ≤ 50 rows. The full competition run on 100K candidates produces `team_001.csv`.
""")

# --- Paths (all relative to ROOT, which os.chdir set above) ---
SAMPLE_JSON = "sample_candidates.json"
input_path  = "temp_sandbox_input.jsonl"
output_path = "sandbox_output.csv"
RANK_PY     = "rank.py"

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Pipeline Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload candidates.jsonl (≤ 100 profiles)",
    type=["jsonl"],
    help="Optional. Leave empty to use the pre-loaded 50-candidate sample."
)

# --- Prepare input JSONL ---
sample_ready = False

if uploaded_file is not None:
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ Custom file loaded: {uploaded_file.name}")
    sample_ready = True
else:
    st.sidebar.info("📂 Using default 50-candidate sample.")
    if os.path.exists(SAMPLE_JSON):
        try:
            with open(SAMPLE_JSON, "r", encoding="utf-8") as f:
                candidates_array = json.load(f)
            with open(input_path, "w", encoding="utf-8") as f:
                for cand in candidates_array:
                    f.write(json.dumps(cand) + "\n")
            st.sidebar.success(f"✅ Sample loaded: {len(candidates_array)} candidates")
            sample_ready = True
        except Exception as e:
            st.sidebar.error(f"❌ Failed to parse sample_candidates.json: {e}")
    else:
        st.sidebar.error(
            f"❌ sample_candidates.json not found at:\n`{os.path.join(ROOT, SAMPLE_JSON)}`\n\n"
            "Make sure it exists in the project root."
        )

# --- Verify rank.py exists before showing the button ---
if not os.path.exists(RANK_PY):
    st.error(
        f"❌ `rank.py` not found at `{os.path.join(ROOT, RANK_PY)}`.\n\n"
        "The app must be run from inside the `candidate_ranker/` repo."
    )
    st.stop()

# --- Main Action ---
run_button = st.sidebar.button(
    "🚀 Run Ranking Engine",
    type="primary",
    disabled=not sample_ready
)

if run_button:
    with st.spinner("⏳ Running FAISS + BM25 Hybrid Retrieval and LTR Scoring..."):

        result = subprocess.run(
            [sys.executable, RANK_PY, "--candidates", input_path, "--out", output_path],
            capture_output=True,
            text=True,
            cwd=ROOT   # rank.py resolves artifacts/ relative to ROOT
        )

    # --- Error display ---
    if result.returncode != 0:
        st.error("❌ Pipeline returned a non-zero exit code.")
        with st.expander("🔍 Full stderr (click to expand)"):
            st.code(result.stderr, language="bash")
        if result.stdout:
            with st.expander("stdout"):
                st.code(result.stdout, language="bash")
        st.stop()

    # --- Success ---
    st.success("✅ Ranking complete!")

    if result.stdout:
        with st.expander("📋 Pipeline logs"):
            st.code(result.stdout, language="bash")

    # --- Load results ---
    if not os.path.exists(output_path):
        st.error(
            f"❌ rank.py exited successfully but `{output_path}` was not created. "
            "Check that rank.py writes its output to the path passed via --out."
        )
        st.stop()

    df = pd.read_csv(output_path)

    if df.empty:
        st.warning("⚠️ The output CSV is empty. No candidates were ranked.")
        st.stop()

    # --- Metrics row ---
    st.subheader("📊 Ranking Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates Ranked", len(df))
    col2.metric("Top LTR Score",  f"{df['score'].max():.4f}")
    col3.metric("Lowest Score",   f"{df['score'].min():.4f}")
    col4.metric("Honeypot Filter", "✅ Active")

    # --- Score distribution note ---
    if len(df) < 100:
        st.info(
            f"ℹ️ {len(df)} candidates ranked (input pool was {len(df)} — "
            "sandbox mode caps output at pool size, not 100). "
            "Full submission uses 100K pool → 100 ranked outputs."
        )

    # --- Results table ---
    st.dataframe(
        df,
        column_config={
            "candidate_id": st.column_config.TextColumn(
                "Candidate ID", width="small"
            ),
            "rank": st.column_config.NumberColumn(
                "Rank", width="small", format="%d"
            ),
            "score": st.column_config.NumberColumn(
                "LTR Score", width="small", format="%.4f"
            ),
            "reasoning": st.column_config.TextColumn(
                "Deterministic Reasoning", width="large"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    # --- Download ---
    with open(output_path, "rb") as f:
        csv_bytes = f.read()

    st.download_button(
        label="📥 Download Sandbox CSV",
        data=csv_bytes,
        file_name="sandbox_submission.csv",
        mime="text/csv",
    )

    # --- Show pre-computed full submission ---
    full_csv_path = "team_001.csv"
    if os.path.exists(full_csv_path):
        st.markdown("---")
        st.subheader("📁 Full Competition Submission (team_001.csv)")
        st.markdown(
            "This is the actual submission generated by running `rank.py` "
            "on the full 100,000-candidate pool."
        )
        df_full = pd.read_csv(full_csv_path)
        st.dataframe(
            df_full,
            column_config={
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="small"),
                "rank":         st.column_config.NumberColumn("Rank", width="small", format="%d"),
                "score":        st.column_config.NumberColumn("LTR Score", width="small", format="%.4f"),
                "reasoning":    st.column_config.TextColumn("Reasoning", width="large"),
            },
            use_container_width=True,
            hide_index=True,
        )
        with open(full_csv_path, "rb") as f:
            full_bytes = f.read()
        st.download_button(
            label="📥 Download team_001.csv (Full Submission)",
            data=full_bytes,
            file_name="team_001.csv",
            mime="text/csv",
        )

    # --- Cleanup temp input ---
    if os.path.exists(input_path):
        os.remove(input_path)