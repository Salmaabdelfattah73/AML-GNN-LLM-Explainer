import streamlit as st
import pandas as pd
from data_loader import resolve_data_dir, load_all_data

st.set_page_config(
    page_title="AML Investigation Assistant",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp { background-color: #f0f4f8; }

.header-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(15,52,96,0.3);
}
.header-card h1 {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}

.search-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #4a5568;
    margin-bottom: 0.3rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    background: #ffffff !important;
    font-size: 0.95rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #3182ce !important;
    box-shadow: 0 0 0 3px rgba(49,130,206,0.15) !important;
}

.details-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border-top: 4px solid #e53e3e;
}
.card-title-red {
    font-size: 0.75rem;
    font-weight: 700;
    color: #e53e3e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
}
.card-title-blue {
    font-size: 0.75rem;
    font-weight: 700;
    color: #3182ce;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
}
.txn-id-badge {
    background: #fff5f5;
    border: 1px solid #fed7d7;
    color: #c53030;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 1.2rem;
}
.report-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1.2rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border-top: 4px solid #3182ce;
}
.no-result {
    text-align: center;
    padding: 3rem;
    color: #a0aec0;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading pipeline artifacts...")
def get_data():
    data_dir = resolve_data_dir()
    if data_dir is None:
        return None, None
    return load_all_data(data_dir), data_dir


payload, data_dir = get_data()

if payload is None:
    st.error("Cannot find required data files. Run the LLM notebook first.")
    st.stop()

agg        = payload["agg"]
results_df = payload["results_df"]
reports    = payload["reports"]
lookup_df  = payload["lookup_df"]
txn_ids    = list(reports.keys())

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>🔍 AML Investigation Assistant</h1>
</div>
""", unsafe_allow_html=True)

# ── SEARCH ────────────────────────────────────────────────────────────────────
st.markdown('<div class="search-label">🔎 Search for Transaction ID</div>',
            unsafe_allow_html=True)

selected = st.selectbox(
    label="transaction_search",
    label_visibility="collapsed",
    options=txn_ids,
    index=0,
    placeholder="Select or type a Transaction ID...",
)

# ── Get row ───────────────────────────────────────────────────────────────────
row = pd.Series()
if not results_df.empty:
    row_df = results_df[results_df["transaction_id"] == selected]
    if not row_df.empty:
        row = row_df.iloc[0]

if row.empty:
    lkp = lookup_df[lookup_df["transaction_id"] == selected]
    if not lkp.empty:
        row = lkp.iloc[0]

# ── TRANSACTION DETAILS ───────────────────────────────────────────────────────
amount_paid = row.get("Amount Paid", "N/A")
amount_recv = row.get("Amount Received", "N/A")
if isinstance(amount_paid, float):
    amount_paid = f"{amount_paid:,.2f}"
if isinstance(amount_recv, float):
    amount_recv = f"{amount_recv:,.2f}"

st.markdown(f"""
<div class="details-card">
    <div class="card-title-red">📋 Transaction Details</div>
    <span class="txn-id-badge">{selected}</span>
</div>
""", unsafe_allow_html=True)

fields = {
    "Timestamp":          row.get("Timestamp", "N/A"),
    "Sender Account":     row.get("src_node", "N/A"),
    "Receiver Account":   row.get("dst_node", "N/A"),
    "Amount Paid":        amount_paid,
    "Amount Received":    amount_recv,
    "Payment Currency":   row.get("Payment Currency", "N/A"),
    "Receiving Currency": row.get("Receiving Currency", "N/A"),
    "Payment Format":     row.get("Payment Format", "N/A"),
}

for key, val in fields.items():
    col_k, col_v = st.columns([1, 2])
    with col_k:
        st.markdown(
            f"<span style='color:#718096; font-size:0.88rem;'>{key}</span>",
            unsafe_allow_html=True)
    with col_v:
        st.markdown(
            f"<span style='color:#2d3748; font-weight:600; font-size:0.88rem;'>{val}</span>",
            unsafe_allow_html=True)

# ── LLM REPORT ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="report-card">
    <div class="card-title-blue">🤖 LLM Investigation Summary</div>
</div>
""", unsafe_allow_html=True)

report_text = reports.get(selected, "")

if report_text:
    st.markdown(report_text)
    st.download_button(
        label="⬇️ Download Report (.md)",
        data=report_text,
        file_name=f"report_{selected}.md",
        mime="text/markdown",
        use_container_width=True,
    )
else:
    st.markdown("""
    <div class="no-result">No report available for this transaction.</div>
    """, unsafe_allow_html=True)
