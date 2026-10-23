import json
import pandas as pd
from pathlib import Path

CANDIDATE_DATA_DIRS = [
    Path("/home/jovyan/AML_Project/NoteBooks/Data/Bassam Data"),
    Path("/home/jovyan/AML_Project/Data/Bassam Data"),
    Path("/home/jovyan/AML_Project/NoteBooks/Bassam/Data/Bassam Data"),
]

REQUIRED_FILES = [
    "transactions_lookup.csv",
    "llm_reports/summary.json",
    "llm_reports/all_reports.json",
]


def resolve_data_dir():
    for cand in CANDIDATE_DATA_DIRS:
        if cand.exists() and all((cand / f).exists() for f in REQUIRED_FILES):
            return cand
    return None


def load_all_data(data_dir: Path) -> dict:
    # Load summary and reports (small files — fast)
    with open(data_dir / "llm_reports" / "summary.json") as f:
        summary = json.load(f)

    with open(data_dir / "llm_reports" / "all_reports.json") as f:
        all_reports_list = json.load(f)

    reports_dict = {
        r["transaction_id"]: r["report"]
        for r in all_reports_list
    }

    # Get only the transaction IDs we have reports for
    report_txn_ids = list(reports_dict.keys())

    # Load ONLY the rows we need from the huge CSV
    # First pass: find the row numbers quickly
    lookup_df = pd.read_csv(
        data_dir / "transactions_lookup.csv",
        usecols=["transaction_id", "Timestamp",
                 "src_node", "dst_node",
                 "Amount Paid", "Amount Received",
                 "Payment Currency", "Receiving Currency",
                 "Payment Format", "Is Laundering", "split"]
    )
    # Filter to only the 10 rows we need
    lookup_df = lookup_df[
        lookup_df["transaction_id"].isin(report_txn_ids)
    ].reset_index(drop=True)

    # Build results_df from summary
    results_df = pd.DataFrame(summary.get("results", []))

    if not results_df.empty:
        merge_cols = [
            c for c in [
                "transaction_id", "Timestamp",
                "src_node", "dst_node",
                "Amount Paid", "Amount Received",
                "Payment Currency", "Receiving Currency",
                "Payment Format", "Is Laundering", "split"
            ] if c in lookup_df.columns
        ]
        results_df = results_df.merge(
            lookup_df[merge_cols],
            on="transaction_id",
            how="left"
        )

    agg = {
        "model":          summary.get("model", "llama3.1:8b"),
        "temperature":    summary.get("temperature", 0.0),
        "n_generated":    summary.get("n_generated", 0),
        "n_valid":        summary.get("n_valid", 0),
        "n_gate_clean":   summary.get("n_gate_clean", 0),
        "n_aggressive":   summary.get("n_aggressive", 0),
        "avg_word_count": summary.get("avg_word_count", 0),
        "elapsed_min":    summary.get("elapsed_min", 0),
    }

    return {
        "summary":    summary,
        "reports":    reports_dict,
        "results_df": results_df,
        "lookup_df":  lookup_df,
        "agg":        agg,
    }
