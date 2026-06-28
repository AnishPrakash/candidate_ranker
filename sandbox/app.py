import os
import sys

# ============================================================
# ROOT PATH DETECTION — works in both environments:
#
# Local dev:      app.py is at candidate_ranker/sandbox/app.py
#                 __file__ = .../candidate_ranker/sandbox/app.py
#                 _parent  = .../candidate_ranker/        <- rank.py is here
#                 ROOT = _parent  ✅
#
# HuggingFace:    app.py is at /app/app.py  (WORKDIR /app)
#                 __file__ = /app/app.py
#                 _parent  = /              <- rank.py is NOT here
#                 ROOT = _this_dir = /app   ✅
# ============================================================
_this_file = os.path.abspath(__file__)
_this_dir  = os.path.dirname(_this_file)
_parent    = os.path.dirname(_this_dir)

if os.path.exists(os.path.join(_parent, "rank.py")):
    ROOT = _parent       # local: candidate_ranker/
else:
    ROOT = _this_dir     # HuggingFace: /app

os.chdir(ROOT)
sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import subprocess
import json

# --- Constants ---
SAMPLE_JSON    = "sample_candidates.json"
FULL_CSV       = "Dev_NOVA.csv"
RANK_PY        = "rank.py"
SANDBOX_INPUT  = "temp_sandbox_input.jsonl"
SANDBOX_OUTPUT = "sandbox_output.csv"

# --- Page Config ---
st.set_page_config(page_title="Redrob Ranker | Dev NOVA", layout="wide")

st.title("🏆 Redrob AI — Candidate Ranking Engine")
st.markdown("""
This sandbox demonstrates our **Hybrid Retrieval (FAISS + BM25) + XGBoost LambdaMART** pipeline.  
Candidates are passed through adversarial defense filters (honeypot detection, skill authenticator)  
before LTR re-scoring and deterministic reasoning generation.
""")

# --- Sidebar ---
st.sidebar.header("⚙️ Pipeline Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload candidates.jsonl (≤ 100 profiles)",
    type=["jsonl"],
    help=(
        "Upload sandbox_candidates.jsonl (included in the repo) to see the full "
        "pipeline run with correct reasoning on all 100 ranked candidates. "
        "Leave empty to view the pre-computed Dev_NOVA.csv result directly."
    )
)

# ================================================================
# BRANCH A — No upload: display Dev_NOVA.csv directly.
# ================================================================
if uploaded_file is None:

    st.sidebar.info("📂 No file uploaded — showing pre-computed result.")

    st.info(
        "**Default mode:** Displaying `Dev_NOVA.csv` — the output of running "
        "`rank.py` on the full 100,000-candidate pool.  \n"
        "To run the pipeline live, upload `sandbox_candidates.jsonl` "
        "(included in the repo root) using the sidebar."
    )

    if not os.path.exists(FULL_CSV):
        st.error(
            f"❌ `{FULL_CSV}` not found at `{os.path.join(ROOT, FULL_CSV)}`.  \n"
            "Make sure the file is committed to the repo root."
        )
        st.stop()

    df_full = pd.read_csv(FULL_CSV)

    st.subheader("📊 Full Submission Results (100,000 → Top 100)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates Ranked",  len(df_full))
    col2.metric("Top LTR Score",      f"{df_full['score'].max():.4f}")
    col3.metric("Lowest Score",       f"{df_full['score'].min():.4f}")
    col4.metric("Honeypot Filter",    "✅ Active")

    st.dataframe(
        df_full,
        column_config={
            "candidate_id": st.column_config.TextColumn("Candidate ID",  width="small"),
            "rank":         st.column_config.NumberColumn("Rank",        width="small", format="%d"),
            "score":        st.column_config.NumberColumn("LTR Score",   width="small", format="%.4f"),
            "reasoning":    st.column_config.TextColumn("Reasoning",     width="large"),
        },
        use_container_width=True,
        hide_index=True,
    )

    with open(FULL_CSV, "rb") as f:
        full_bytes = f.read()

    st.download_button(
        label="📥 Download Dev_NOVA.csv (Full Submission)",
        data=full_bytes,
        file_name="Dev_NOVA.csv",
        mime="text/csv",
    )

# ================================================================
# BRANCH B — File uploaded: run rank.py live.
# ================================================================
else:

    if not os.path.exists(RANK_PY):
        st.error(
            f"❌ `rank.py` not found at `{os.path.join(ROOT, RANK_PY)}`.  \n"
            "The app must be run from inside the candidate_ranker repo."
        )
        st.stop()

    with open(SANDBOX_INPUT, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")

    st.info(
        f"**Live mode:** Running `rank.py` on `{uploaded_file.name}`.  \n"
        "The pipeline uses the pre-built FAISS + BM25 index and LTR model.  \n"
        "For correct reasoning on all 100 rows, upload `sandbox_candidates.jsonl`."
    )

    run_button = st.sidebar.button("🚀 Run Ranking Engine", type="primary")

    if run_button:

        with st.spinner("⏳ Running FAISS + BM25 Hybrid Retrieval and LTR Scoring..."):
            result = subprocess.run(
                [sys.executable, RANK_PY,
                 "--candidates", SANDBOX_INPUT,
                 "--out",        SANDBOX_OUTPUT],
                capture_output=True,
                text=True,
                cwd=ROOT
            )

        if result.returncode != 0:
            st.error("❌ Pipeline returned a non-zero exit code.")
            with st.expander("🔍 Full stderr"):
                st.code(result.stderr, language="bash")
            if result.stdout:
                with st.expander("stdout"):
                    st.code(result.stdout, language="bash")
            st.stop()

        st.success("✅ Ranking complete!")

        if result.stdout:
            with st.expander("📋 Pipeline logs"):
                st.code(result.stdout, language="bash")

        if not os.path.exists(SANDBOX_OUTPUT):
            st.error(
                f"❌ rank.py exited cleanly but `{SANDBOX_OUTPUT}` was not created.  \n"
                "Check that rank.py writes its output to the path given by --out."
            )
            st.stop()

        df = pd.read_csv(SANDBOX_OUTPUT)

        if df.empty:
            st.warning("⚠️ Output CSV is empty — no candidates were ranked.")
            st.stop()

        st.subheader("📊 Ranking Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates Ranked", len(df))
        col2.metric("Top LTR Score",     f"{df['score'].max():.4f}")
        col3.metric("Lowest Score",      f"{df['score'].min():.4f}")
        col4.metric("Honeypot Filter",   "✅ Active")

        st.info(
            "ℹ️ **About the output count:** The ranking pool is the pre-built FAISS index "
            "(all 100,000 candidates). The uploaded file is used for reasoning generation only. "
            "Output is always 100 rows regardless of upload size — "
            "this is by design for the competition format."
        )

        st.dataframe(
            df,
            column_config={
                "candidate_id": st.column_config.TextColumn("Candidate ID",            width="small"),
                "rank":         st.column_config.NumberColumn("Rank",                  width="small", format="%d"),
                "score":        st.column_config.NumberColumn("LTR Score",             width="small", format="%.4f"),
                "reasoning":    st.column_config.TextColumn("Deterministic Reasoning", width="large"),
            },
            use_container_width=True,
            hide_index=True,
        )

        with open(SANDBOX_OUTPUT, "rb") as f:
            csv_bytes = f.read()

        st.download_button(
            label="📥 Download Sandbox CSV",
            data=csv_bytes,
            file_name="sandbox_output.csv",
            mime="text/csv",
        )

        if os.path.exists(FULL_CSV):
            st.markdown("---")
            st.subheader("📁 Full Competition Submission (Dev_NOVA.csv)")
            st.markdown(
                "The actual submission produced by `rank.py` on the full 100,000-candidate pool."
            )
            df_full = pd.read_csv(FULL_CSV)
            st.dataframe(
                df_full,
                column_config={
                    "candidate_id": st.column_config.TextColumn("Candidate ID",  width="small"),
                    "rank":         st.column_config.NumberColumn("Rank",        width="small", format="%d"),
                    "score":        st.column_config.NumberColumn("LTR Score",   width="small", format="%.4f"),
                    "reasoning":    st.column_config.TextColumn("Reasoning",     width="large"),
                },
                use_container_width=True,
                hide_index=True,
            )
            with open(FULL_CSV, "rb") as f:
                full_bytes = f.read()
            st.download_button(
                label="📥 Download Dev_NOVA.csv (Full Submission)",
                data=full_bytes,
                file_name="Dev_NOVA.csv",
                mime="text/csv",
            )

        if os.path.exists(SANDBOX_INPUT):
            os.remove(SANDBOX_INPUT)