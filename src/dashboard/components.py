import streamlit as st
import pandas as pd


def render_overview_metrics(agg):
    st.markdown("### 📊 Pipeline Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Reports Generated", agg["n_generated"])
    with c2:
        st.metric("Valid Reports", f"{agg['n_valid']}/{agg['n_generated']}")
    with c3:
        st.metric("ID Gate PASS", f"{agg['n_gate_clean']}/{agg['n_generated']}")
    with c4:
        st.metric("Aggressive Recs", agg["n_aggressive"])
    with c5:
        st.metric("Avg Word Count", f"{agg['avg_word_count']:.0f}")


def render_results_table(results_df):
    st.markdown("### 📋 All Reports — Summary Table")
    display_cols = [
        "transaction_id", "probability", "fidelity_plus",
        "fidelity_minus", "identifier_gate_clean",
        "is_valid", "word_count", "rubric_score", "y_true"
    ]
    available = [c for c in display_cols if c in results_df.columns]
    df_show = results_df[available].copy()

    if "identifier_gate_clean" in df_show.columns:
        df_show["identifier_gate_clean"] = df_show[
            "identifier_gate_clean"].apply(
            lambda v: "✅ PASS" if v else "❌ FAIL")
    if "is_valid" in df_show.columns:
        df_show["is_valid"] = df_show["is_valid"].apply(
            lambda v: "✅" if v else "❌")

    df_show = df_show.rename(columns={
        "transaction_id":        "Transaction ID",
        "probability":           "Model Prob",
        "fidelity_plus":         "Fidelity+",
        "fidelity_minus":        "Fidelity-",
        "identifier_gate_clean": "ID Gate",
        "is_valid":              "Valid",
        "word_count":            "Words",
        "rubric_score":          "Rubric",
        "y_true":                "Ground Truth",
    })

    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model Prob": st.column_config.ProgressColumn(
                "Model Prob", min_value=0.0, max_value=1.0, format="%.4f"),
            "Rubric": st.column_config.ProgressColumn(
                "Rubric", min_value=0.0, max_value=1.0, format="%.2f"),
        }
    )


def render_transaction_details(row):
    st.markdown("#### 🔍 Transaction Details")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Core Fields**")
        for k, v in {
            "Transaction ID":   row.get("transaction_id"),
            "Timestamp":        row.get("Timestamp"),
            "Sender Account":   row.get("src_node"),
            "Receiver Account": row.get("dst_node"),
            "Split":            row.get("split"),
        }.items():
            st.markdown(f"- **{k}:** `{v}`")

    with col_r:
        st.markdown("**Financial Fields**")
        for k, v in {
            "Amount Paid":        row.get("Amount Paid"),
            "Amount Received":    row.get("Amount Received"),
            "Payment Currency":   row.get("Payment Currency"),
            "Receiving Currency": row.get("Receiving Currency"),
            "Payment Format":     row.get("Payment Format"),
            "Ground Truth": (
                "🔴 Laundering"
                if row.get("Is Laundering") == 1 else "🟢 Normal"
            ),
        }.items():
            st.markdown(f"- **{k}:** `{v}`")


def render_model_metrics(row):
    st.markdown("#### 🤖 Model & Explanation Metrics")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        prob = float(row.get("probability", 0.0))
        st.metric("Model Probability", f"{prob:.4f}")
        st.progress(prob)
    with c2:
        st.metric("ID Gate",
                  "✅ PASS" if row.get("identifier_gate_clean") else "❌ FAIL")
    with c3:
        st.metric("Fidelity+", f"{row.get('fidelity_plus', 0.0):.4f}")
    with c4:
        st.metric("Fidelity-", f"{row.get('fidelity_minus', 0.0):.4f}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Rubric Score", f"{row.get('rubric_score', 0.0):.2f}")
    with c6:
        st.metric("Word Count", row.get("word_count", 0))
    with c7:
        st.metric("Valid Report",
                  "✅ Yes" if row.get("is_valid") else "❌ No")
    with c8:
        h = row.get("hallucinated_ids", [])
        st.metric("Hallucinated IDs",
                  len(h) if isinstance(h, list) else 0)


def render_llm_report(report_text, txn_id="report"):
    st.markdown("#### 📄 LLM Investigation Report")
    st.divider()
    if not report_text:
        st.warning("No report text available for this transaction.")
        return
    st.markdown(report_text)
    st.divider()
    st.download_button(
        label="⬇️ Download Report (.md)",
        data=report_text,
        file_name=f"report_{txn_id}.md",
        mime="text/markdown",
        use_container_width=True
    )
