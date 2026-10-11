#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script version of pgexplainer.ipynb
# Code cells are reproduced verbatim and in the original order; markdown cells are kept as comments.
# Cell markers (# %%) let VS Code / Spyder / Jupytext run the file cell by cell.

# %% [markdown]
# # PGExplainer -- Edge-Level Fraud Explanation for the Frozen PNA Classifier
#
# **Pipeline stage:** `preprocessing.ipynb` -> `aml_graph_construction.ipynb` ->
# `train_baseline_comparison.ipynb` -> **this notebook** -> LLM evidence-locked prompting
#
# **Goal:** given a transaction the frozen PNA classifier has flagged as laundering,
# produce a structured, evidence-locked explanation of *why* -- a ranked list of the
# other transactions (edges) in its local neighborhood that most shaped the model's
# decision, plus a feature-level breakdown of the flagged transaction's own fields.
# This structured JSON is what the LLM stage will be constrained to cite from --
# it never sees raw graph structure or invents evidence of its own.
#
# **Decisions this notebook implements (agreed before any code was written):**
#
# 1. **Implementation approach:** custom PGExplainer, not `torch_geometric.explain.PGExplainer`
#    off the shelf. The classifier's split `encode()`/`score()` design, the per-batch
#    `augment_subgraph()` step (self-loops + reverse edges applied *outside* the model),
#    and the target edge's own features bypassing message passing entirely in `score()`
#    don't match the single-`forward()` contract the built-in `Explainer` API expects.
#    We do reuse one low-level PyG utility (`set_masks`/`clear_masks` from
#    `torch_geometric.explain.algorithm.utils`) for the actual mask injection, since
#    that hooks into `MessagePassing.propagate()` at the base-class level and works
#    identically regardless of `PNAConv`'s specific `forward()` signature.
#
# 2. **Explanation target:** for a flagged transaction, output = (a) its own raw
#    features (always included, un-masked -- `score()` never routes the target edge
#    through message passing) + a feature-level Integrated Gradients breakdown of
#    those features (Decision 2/Scenario B), and (b) a ranked list of the *other*
#    transactions in its k-hop computational subgraph that shaped `h_src`/`h_dst`,
#    scored by a trained PGExplainer MLP (edge-level attribution only -- no IG on
#    neighbors, see Section 13).
#
# 3. **Forward/reverse merge + self-loop exclusion:** every real transaction enters
#    the augmented subgraph twice (original + `is_reverse=1` copy). Their importance
#    scores are merged by `max()` into one transaction-level score before ranking.
#    Synthetic self-loops added by `augment_subgraph` (all-zero `edge_attr`, distinct
#    from a genuine self-transaction which carries `is_self_loop=1` in its real
#    `edge_attr`) are excluded entirely -- they aren't real transactions.
#
# 4. **Training population:** the explainer MLP is trained/tuned/evaluated on
#    *predicted-positive* edges only (post-threshold, not ground-truth positives) --
#    matching the actual deployment use case, where explanations are only ever
#    requested for transactions the model already flagged. Phase-scoping is
#    preserved throughout (train predicted-positives use `train_base_ei/ea`, val use
#    `val_base_ei/ea`, test use `test_base_ei/ea` -- test touched once, at the end).
#
# 5. **Faithfulness metric:** Fidelity+ / Fidelity- (probability-drop when removing
#    vs. keeping only the explanation), with a random-k baseline for comparison.
#    Top-k chosen on val via a characterization score, confirmed once on test.
#
# 6. **Feature-level attribution (Scenario B):** Integrated Gradients, target
#    transaction only (not neighbors -- the neighbor GNN path is expensive to run IG
#    through and its importance is already covered by the PGExplainer edge score).
#    Baseline = 0 for continuous (a trained `StandardScaler`'s 0 *is* the train-set
#    mean) and binary features. Categorical columns (`Payment Format_enc` etc.) are
#    reported as plain facts, no IG score -- interpolating a categorical code has no
#    meaning, and the LLM stage doesn't need a numeric weight to use a categorical
#    fact correctly.
#
# **A note on what this notebook does NOT do:** it never re-trains or fine-tunes the
# PNA classifier itself (`best_model_pna.pt` stays frozen throughout), and it never
# uses `pattern_diagnostics.csv` / ground-truth pattern labels as a training signal --
# only as an optional, offline sanity check on the demo outputs at the very end
# (Section 15), consistent with the project's existing rule that pattern-type labels
# are diagnostics only.

# %% [markdown]
# ## 1. Imports & config

# %%
# --- Imports ---
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks

from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import auc as sk_auc
DATA_DIR = Path("/home/jovyan/AML_Project/NoteBooks/Data/Bassam Data")
assert DATA_DIR.exists(), f"{DATA_DIR} not found."

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %%
# --- Config ---
# Classifier-side settings mirror train_baseline_comparison.ipynb exactly -- these
# are NOT tunable here, they describe the frozen model we're loading.
NUM_NEIGHBORS_INFER = [10, 5]      # must match BEST_HP["num_neighbors"] used to
                                    # produce best_model_pna.pt -- overwritten below
                                    # once we load best_hyperparams_pna.json
INFER_BATCH_SIZE = 16384

# --- Explainer-side settings (new, tunable) ---
EXPLAINER_HIDDEN = 64              # MLP hidden width
EXPLAINER_EPOCHS = 10              # small MLP, few epochs -- watch val Fidelity,
                                    # raise if it's still improving
EXPLAINER_LR = 3e-3
EXPLAINER_BATCH_SIZE = 256         # number of TARGET transactions per batch (each
                                    # one pulls its own k-hop subgraph -- this is
                                    # not the same kind of batch as classifier
                                    # training, keep it small)
MASK_SPARSITY_COEF = 0.03          # penalizes the mean mask value -- encourages a
                                    # sparse (few important edges), not diffuse, mask.
                                    # FIX (first training run): loss_pred is trivially
                                    # minimized by mask=1 everywhere (no masking at all
                                    # exactly reproduces the original prediction), so
                                    # sparsity is the ONLY force pushing masks toward 0
                                    # -- at 5e-4 it was ~70x weaker than the converged
                                    # loss_pred (~0.036) and every mask saturated at 1.0.
                                    # 0.03 is a starting point, not a tuned value -- watch
                                    # the resulting mean mask value and raise further if
                                    # importance scores still don't differentiate.
MASK_ENTROPY_COEF = 1e-3           # penalizes masks that sit near 0.5 (encourages
                                    # confident 0/1-ish decisions, standard PGExplainer reg.)
                                    # Left unchanged for this run -- isolate the effect of
                                    # the sparsity fix first before touching this too.
TEMPERATURE = 1.0                  # Gumbel/concrete relaxation temperature (kept
                                    # fixed here; anneal 5.0 -> 1.0 across epochs if
                                    # masks look too noisy in practice)

TOP_K_CANDIDATES = [3, 5, 10, 15]  # swept on val to pick the final k (Decision 5)
IG_STEPS = 50                      # Integrated Gradients interpolation steps

RANDOM_SEED = 2000
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Config loaded.")

# %% [markdown]
# ## 2. Load `graph_data.pt`, metadata, encoders/scalers, and the human-readable lookup
#
# Everything here is read-only -- nothing in this section is fit on any data, it's
# all artifacts already produced and saved by the earlier notebooks.

# %%
data = torch.load(DATA_DIR / "graph_data.pt", weights_only=False)
print(data)

with open(DATA_DIR / "metadata.json") as f:
    metadata = json.load(f)

edge_cat_start = data.edge_cat_start if hasattr(data, "edge_cat_start") else metadata["edge_cat_start"]
EDGE_FEATURE_COLS = metadata["edge_feature_cols"]
CAT_COLS = ["Payment Format", "Payment Currency", "Receiving Currency"]
vocab_sizes = [metadata["vocab_sizes"][c] for c in CAT_COLS]

n_nodes = data.x.size(0)
in_dim = data.x.size(1)
print(f"edge_cat_start={edge_cat_start}  n_nodes={n_nodes:,}  in_dim={in_dim}")
print(f"EDGE_FEATURE_COLS ({len(EDGE_FEATURE_COLS)}): {EDGE_FEATURE_COLS}")

# Column index of "is_self_loop" within edge_attr -- needed in Section 6 to tell a
# genuine self-transaction (real edge_attr, is_self_loop=1) apart from a synthetic
# self-loop added by augment_subgraph (all-zero edge_attr, is_self_loop reads 0
# only because everything was zero-filled, not because it's a real flag).
IS_SELF_LOOP_COL = EDGE_FEATURE_COLS.index("is_self_loop")
print(f"is_self_loop column index: {IS_SELF_LOOP_COL}")

# %%
# --- Encoders / scalers (for decoding model-ready features back to human-readable
# values in the final JSON output -- the open item flagged from a prior session) ---
with open(DATA_DIR / "encoders.pkl", "rb") as f:
    cat_encoders = pickle.load(f)          # {col_name: {original_value: code}}

with open(DATA_DIR / "edge_scaler.pkl", "rb") as f:
    edge_scaler = pickle.load(f)           # fit on CONTINUOUS_EDGE_COLS, train-only

CONTINUOUS_EDGE_COLS = ["log_amount_paid", "log_amount_received", "log_amount_ratio",
                         "log_hours_since_last_tx", "log_tx_count_cumulative"]
# EDGE_FEATURE_COLS carries these as "scaled_<col>" -- map scaled name -> raw name
SCALED_TO_RAW = {f"scaled_{c}": c for c in CONTINUOUS_EDGE_COLS}

# Reverse category maps: code -> original string, per categorical column
cat_decoders = {col: {v: k for k, v in enc.items()} for col, enc in cat_encoders.items()}

# --- Human-readable transaction lookup + transaction_id <-> edge_idx mapping ---
transactions_lookup = pd.read_csv(DATA_DIR / "transactions_lookup.csv")
transaction_ids = np.load(DATA_DIR / "transaction_ids.npy", allow_pickle=True)  # (E,), same row order as edge_index/edge_attr

txn_id_to_edge_idx = {tid: i for i, tid in enumerate(transaction_ids)}
edge_idx_to_txn_id = {i: tid for tid, i in txn_id_to_edge_idx.items()}

# transaction_id -> which split it belongs to (train/val/test) -- needed later to
# pick the right phase-scoped base graph inside explain_transaction()
txn_id_to_split = dict(zip(transactions_lookup["transaction_id"], transactions_lookup["split"]))

print(f"transactions_lookup: {transactions_lookup.shape}")
print(f"{len(txn_id_to_edge_idx):,} transaction_id -> edge_idx mappings built.")

# %% [markdown]
# ## 3. Rebuild the frozen PNA `EdgeClassifier`
#
# **These class definitions are copied verbatim from `train_baseline_comparison.ipynb`
# (Sections 4/5) -- PNA only, since that's the winning/saved architecture.** If that
# notebook's `EdgeEncoder`/`NodeEncoderPNA`/`EdgeClassifier`/`augment_subgraph` ever
# change, this cell has to be updated to match, or `best_model_pna.pt`'s state_dict
# won't load. Nothing here is retrained -- we load the saved weights and freeze
# every parameter immediately after.

# %%
class EdgeEncoder(nn.Module):
    def __init__(self, edge_cat_start, vocab_sizes, emb_dim=8, out_dim=32):
        super().__init__()
        self.edge_cat_start = edge_cat_start
        n_continuous = edge_cat_start + 1  # +1 for is_reverse
        self.cont_proj = nn.Linear(n_continuous, out_dim)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(v, emb_dim) for v in vocab_sizes])
        self.out_proj = nn.Linear(out_dim + emb_dim * len(vocab_sizes), out_dim)

    def forward(self, edge_attr, is_reverse=None):
        if is_reverse is None:
            is_reverse = torch.zeros(edge_attr.size(0), 1, device=edge_attr.device, dtype=edge_attr.dtype)
        cont = torch.cat([edge_attr[:, :self.edge_cat_start], is_reverse], dim=1)
        cont_out = F.relu(self.cont_proj(cont))
        cat = edge_attr[:, self.edge_cat_start:].long()
        cat_out = torch.cat([emb(cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        return self.out_proj(torch.cat([cont_out, cat_out], dim=1))


class NodeEncoderPNA(nn.Module):
    def __init__(self, in_dim, hidden_dim, deg, n_layers=2, dropout=0.2, edge_dim=32, towers=2):
        super().__init__()
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        conv_kwargs = dict(aggregators=aggregators, scalers=scalers, deg=deg,
                            edge_dim=edge_dim, towers=towers, pre_layers=1, post_layers=1)
        self.convs = nn.ModuleList([PNAConv(in_dim, hidden_dim, **conv_kwargs)])
        for _ in range(n_layers - 1):
            self.convs.append(PNAConv(hidden_dim, hidden_dim, **conv_kwargs))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_embedding):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_embedding)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EdgeClassifier(nn.Module):
    def __init__(self, model_name, in_dim, edge_cat_start, vocab_sizes,
                 hidden_dim=32, n_layers=2, dropout=0.2, deg=None, pna_towers=2):
        super().__init__()
        assert model_name == "pna", "This notebook only rebuilds the PNA winner."
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.uses_edge_attr_in_mp = True

        self.edge_encoder = EdgeEncoder(edge_cat_start, vocab_sizes, emb_dim=8, out_dim=hidden_dim)
        assert deg is not None, "PNA requires a train-only degree histogram"
        self.node_encoder = NodeEncoderPNA(in_dim, hidden_dim, deg, n_layers, dropout,
                                            edge_dim=hidden_dim, towers=pna_towers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index, edge_attr, is_reverse=None):
        edge_emb = self.edge_encoder(edge_attr, is_reverse)
        return self.node_encoder(x, edge_index, edge_emb)

    def score(self, h_src, h_dst, target_edge_attr):
        target_edge_emb = self.edge_encoder(target_edge_attr)
        z = torch.cat([h_src, h_dst, target_edge_emb], dim=1)
        return self.head(z).squeeze(-1)


def augment_subgraph(edge_index, edge_attr, num_nodes):
    # Verbatim from train_baseline_comparison.ipynb Section 3 -- see that notebook
    # for the full explanation. Returns (edge_index, edge_attr, is_reverse).
    ei = edge_index
    ea = edge_attr
    is_reverse = torch.zeros(ei.size(1), 1, dtype=torch.float)

    non_self_loop = ei[0] != ei[1]
    rev_ei = ei[:, non_self_loop].flip(0)
    rev_ea = ea[non_self_loop]
    rev_is_reverse = torch.ones(rev_ei.size(1), 1, dtype=torch.float)

    ei = torch.cat([ei, rev_ei], dim=1)
    ea = torch.cat([ea, rev_ea], dim=0)
    is_reverse = torch.cat([is_reverse, rev_is_reverse], dim=0)

    n_before = ei.size(1)
    ei, ea = add_remaining_self_loops(ei, ea, fill_value=0.0, num_nodes=num_nodes)
    n_added = ei.size(1) - n_before
    if n_added > 0:
        is_reverse = torch.cat([is_reverse, torch.zeros(n_added, 1, dtype=torch.float)], dim=0)

    return ei, ea, is_reverse


def augment_subgraph_traceable(edge_index, edge_attr, e_id, num_nodes):
    # Same logic as augment_subgraph, but threads the ORIGINAL global edge id
    # (sub.e_id from NeighborLoader) through the reverse-copy and self-loop steps
    # too, so every row in the returned edge set can be traced back to a real
    # transaction_id (or flagged as synthetic with e_id = -1).
    #
    # BUG FIX (found while debugging the Section 6 assertion): add_remaining_self_loops
    # does NOT simply append new self-loops after the existing edges in place -- it
    # REMOVES every existing self-loop from its original position first
    # (internally: mask = edge_index[0] != edge_index[1]), then rebuilds ONE
    # self-loop block of size num_nodes, ordered by node index (not by original
    # edge order), copying the real edge_attr back in for nodes that already had a
    # self-loop and using fill_value for the rest. A naive "assume the last
    # (new_count - old_count) rows are synthetic" concat -- which is what this
    # function originally did, and what augment_subgraph (Section 3, copied
    # verbatim from train_baseline_comparison.ipynb) still does for is_reverse --
    # desyncs eid/is_reverse from edge_attr whenever a node in the subgraph has a
    # genuine self-transaction (is_self_loop=1), because that real self-loop gets
    # moved into the middle of the rebuilt block, not left in place. Fixed here by
    # replicating add_remaining_self_loops' exact internal logic by hand, so eid
    # (and is_reverse) can be threaded through it correctly instead of guessing
    # where rows landed afterward. See Section 6b for a diagnostic on how often
    # this actually matters for the classifier''s own (unfixed) augment_subgraph.
    ei = edge_index
    ea = edge_attr
    eid = e_id.clone()
    is_reverse = torch.zeros(ei.size(1), 1, dtype=torch.float)

    non_self_loop = ei[0] != ei[1]
    rev_ei = ei[:, non_self_loop].flip(0)
    rev_ea = ea[non_self_loop]
    rev_eid = eid[non_self_loop]              # same transaction_id as its forward copy
    rev_is_reverse = torch.ones(rev_ei.size(1), 1, dtype=torch.float)

    ei = torch.cat([ei, rev_ei], dim=1)
    ea = torch.cat([ea, rev_ea], dim=0)
    eid = torch.cat([eid, rev_eid], dim=0)
    is_reverse = torch.cat([is_reverse, rev_is_reverse], dim=0)

    # --- Hand-written equivalent of add_remaining_self_loops(fill_value=0.0),
    # extended to also carry eid and is_reverse through correctly ---
    mask = ei[0] != ei[1]                        # True for non-self-loop rows
    self_loop_mask = ~mask
    existing_loop_nodes = ei[0, self_loop_mask]   # node index of each pre-existing self-loop
    existing_loop_ea = ea[self_loop_mask]
    existing_loop_eid = eid[self_loop_mask]
    existing_loop_is_rev = is_reverse[self_loop_mask]  # always 0 for real self-loops

    loop_index = torch.arange(num_nodes, dtype=ei.dtype, device=ei.device)
    loop_ei = loop_index.unsqueeze(0).repeat(2, 1)
    loop_ea = ea.new_full((num_nodes,) + ea.size()[1:], 0.0)          # fill_value=0.0
    loop_eid = torch.full((num_nodes,), -1, dtype=eid.dtype, device=eid.device)
    loop_is_reverse = torch.zeros(num_nodes, 1, dtype=torch.float)

    loop_ea[existing_loop_nodes] = existing_loop_ea
    loop_eid[existing_loop_nodes] = existing_loop_eid
    loop_is_reverse[existing_loop_nodes] = existing_loop_is_rev

    final_ei = torch.cat([ei[:, mask], loop_ei], dim=1)
    final_ea = torch.cat([ea[mask], loop_ea], dim=0)
    final_eid = torch.cat([eid[mask], loop_eid], dim=0)
    final_is_reverse = torch.cat([is_reverse[mask], loop_is_reverse], dim=0)

    return final_ei, final_ea, final_is_reverse, final_eid

# %%
# --- Load the frozen checkpoint ---
with open(DATA_DIR / "best_hyperparams_pna.json") as f:
    best_hp = json.load(f)
NUM_NEIGHBORS_INFER = best_hp["num_neighbors"]
print(f"Loaded best_hyperparams_pna.json -- num_neighbors={NUM_NEIGHBORS_INFER}, "
      f"hidden_dim={best_hp['hidden_dim']}, n_layers={best_hp['n_layers']}")

checkpoint = torch.load(DATA_DIR / "best_model_pna.pt", weights_only=False)
assert checkpoint["model_name"] == "pna"
pna_deg = checkpoint["pna_deg_histogram"]

model = EdgeClassifier(
    "pna", in_dim, edge_cat_start, vocab_sizes,
    hidden_dim=checkpoint["hyperparameters"]["hidden_dim"],
    n_layers=checkpoint["hyperparameters"]["n_layers"],
    dropout=checkpoint["hyperparameters"]["dropout"],
    deg=pna_deg,
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Freeze -- this notebook trains only the small explainer MLP defined in Section 7,
# the classifier itself never gets a gradient update from here on.
for p in model.parameters():
    p.requires_grad_(False)

print(f"Loaded frozen PNA classifier from seed {checkpoint['seed']}. Params frozen: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 4. Recompute the locked decision threshold (Decision 4)
#
# `best_model_pna.pt` saves the winning seed's weights but **not** the val-locked
# threshold picked for it in Section 10 of `train_baseline_comparison.ipynb` --
# that value only ever lived in an in-memory variable during that run. The
# computation is deterministic (a function of the frozen weights + the val split),
# so we reproduce it here rather than re-deriving it from scratch: same
# `compute_node_embeddings_for` / `evaluate` logic, run once, on val only.

# %%
val_base_mask = data.train_mask | data.val_mask
val_base_ei = data.edge_index[:, val_base_mask]
val_base_ea = data.edge_attr[val_base_mask]

train_base_ei = data.edge_index[:, data.train_mask]
train_base_ea = data.edge_attr[data.train_mask]

test_base_ei = data.edge_index
test_base_ea = data.edge_attr

BASE_EDGES_BY_PHASE = {
    "train": (train_base_ei, train_base_ea),
    "val":   (val_base_ei, val_base_ea),
    "test":  (test_base_ei, test_base_ea),
}


def _local_lookup(seed_ids, query_ids):
    sorted_ids, sort_perm = torch.sort(seed_ids)
    pos_in_sorted = torch.searchsorted(sorted_ids, query_ids)
    return sort_perm[pos_in_sorted]


@torch.no_grad()
def compute_node_embeddings_for(base_ei, base_ea, target_idx, num_neighbors, batch_size):
    # Verbatim logic from train_baseline_comparison.ipynb Section 6, using the
    # frozen `model` loaded above.
    target_nodes = torch.unique(torch.cat([
        data.edge_index[0, target_idx], data.edge_index[1, target_idx]
    ]))
    base_graph = Data(x=data.x, edge_index=base_ei, edge_attr=base_ea, num_nodes=data.x.size(0))
    loader = NeighborLoader(base_graph, num_neighbors=num_neighbors, input_nodes=target_nodes,
                             batch_size=batch_size, shuffle=False)

    all_h = torch.zeros(data.x.size(0), model.hidden_dim)
    for sub in loader:
        aug_ei, aug_ea, aug_is_rev = augment_subgraph(sub.edge_index, sub.edge_attr, sub.num_nodes)
        x_dev = sub.x.to(DEVICE)
        aug_ei, aug_ea, aug_is_rev = aug_ei.to(DEVICE), aug_ea.to(DEVICE), aug_is_rev.to(DEVICE)
        h = model.encode(x_dev, aug_ei, aug_ea, aug_is_rev)
        all_h[sub.n_id[:sub.batch_size]] = h[:sub.batch_size].cpu()
        del sub, aug_ei, aug_ea, aug_is_rev, h
    del base_graph
    return all_h


@torch.no_grad()
def score_edges(base_ei, base_ea, target_idx, num_neighbors, batch_size, threshold=None):
    h = compute_node_embeddings_for(base_ei, base_ea, target_idx, num_neighbors, batch_size).to(DEVICE)
    src = data.edge_index[0, target_idx].to(DEVICE)
    dst = data.edge_index[1, target_idx].to(DEVICE)
    ea = data.edge_attr[target_idx].to(DEVICE)
    y = data.y[target_idx].float()

    logits_chunks = []
    for start in range(0, target_idx.size(0), batch_size):
        end = start + batch_size
        logits_chunks.append(model.score(h[src[start:end]], h[dst[start:end]], ea[start:end]).cpu())
    logits = torch.cat(logits_chunks)
    probs = torch.sigmoid(logits)

    y_np, probs_np = y.numpy().astype(int), probs.numpy()
    precision, recall, pr_thresholds = precision_recall_curve(y_np, probs_np)
    pr_auc = sk_auc(recall, precision)

    if threshold is None:
        f1_curve = np.divide(2 * precision * recall, precision + recall,
                              out=np.zeros_like(precision), where=(precision + recall) > 0)
        best_idx = int(np.argmax(f1_curve[:-1])) if len(pr_thresholds) > 0 else 0
        threshold = float(pr_thresholds[best_idx]) if len(pr_thresholds) > 0 else 0.5
        f1 = float(f1_curve[best_idx]) if len(pr_thresholds) > 0 else 0.0
    else:
        preds_np = (probs_np >= threshold).astype(int)
        f1 = f1_score(y_np, preds_np, pos_label=1, zero_division=0)

    del h
    return {"f1": f1, "pr_auc": pr_auc, "probs": probs, "threshold": threshold}


val_metrics = score_edges(val_base_ei, val_base_ea, data.val_mask.nonzero(as_tuple=True)[0],
                           NUM_NEIGHBORS_INFER, INFER_BATCH_SIZE)
LOCKED_THRESHOLD = val_metrics["threshold"]
print(f"Recomputed val-locked threshold = {LOCKED_THRESHOLD:.4f}  "
      f"(val F1={val_metrics['f1']:.4f}, matches Section 10's seed {checkpoint['seed']} run if close to "
      f"baseline_comparison_summary.json's reported test performance)")

# %% [markdown]
# ### 4b. Seeding verification (run first, cheap)
#
# `get_target_context` (Section 10) reseeds `torch.manual_seed(target_edge_idx)` right
# before calling `NeighborLoader`, so the same transaction always resolves to the same
# sampled subgraph (Section 10's markdown explains why this matters -- without it, the
# same transaction produced two different model probabilities on this environment in an
# earlier run). Neighbor sampling here goes through a compiled extension
# (`torch.ops.torch_sparse.neighbor_sample`), and PyTorch's own docs note that custom
# operators aren't always guaranteed to respect `torch.manual_seed()`. This cell checks
# that assumption directly, in under a second, self-contained (no dependency on
# `predicted_positive` or Section 10's helpers, so it runs this early) -- if it fails,
# better to find out now than after a multi-hour training run.

# %%
def _seed_check_subgraph(target_edge_idx, seed):
    src = data.edge_index[0, target_edge_idx].item()
    dst = data.edge_index[1, target_edge_idx].item()
    seed_nodes = torch.tensor([src, dst], dtype=torch.long)
    base_graph = Data(x=data.x, edge_index=train_base_ei, edge_attr=train_base_ea, num_nodes=data.x.size(0))
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    loader = NeighborLoader(base_graph, num_neighbors=NUM_NEIGHBORS_INFER, input_nodes=seed_nodes,
                             batch_size=seed_nodes.size(0), shuffle=False)
    sub = next(iter(loader))
    return sub.edge_index.clone()


_check_idx = data.train_mask.nonzero(as_tuple=True)[0][0].item()
_ei_a = _seed_check_subgraph(_check_idx, _check_idx)
_ei_b = _seed_check_subgraph(_check_idx, _check_idx)
SEED_CHECK_PASSED = torch.equal(_ei_a, _ei_b)

print(f"Seeding verification (torch.manual_seed controls NeighborLoader sampling): "
      f"{'PASS' if SEED_CHECK_PASSED else 'FAIL'}")
if not SEED_CHECK_PASSED:
    print("*** WARNING: get_target_context()'s determinism fix (Section 10) does NOT "
          "work on this environment's torch_sparse backend -- explanations may still "
          "vary between calls for the same transaction. The training run below will "
          "still proceed (it doesn't depend on this), but do not trust per-transaction "
          "Fidelity/importance reproducibility in Sections 14/15 until this is fixed. "
          "Flag this exact message to Claude for a backend-specific alternative. ***")

# %% [markdown]
# ## 5. Predicted-positive edge sets per phase (Decision 4)
#
# The explainer is trained/tuned/evaluated only on transactions the **frozen model
# itself** flags as fraud (`prob >= LOCKED_THRESHOLD`), scored within each phase's
# own message-passing scope -- not on ground-truth positives. This scores every edge
# in each mask once, which is the same order of cost as Section 10's final test run
# in the baseline notebook (that touched all 31.9M edges once already), so it's
# within the budget this environment has already handled -- but for `train_mask`
# (22.3M edges) specifically, run this as a detached background job
# (`nohup python -u ... &`, same workaround as before) rather than in an
# interactive cell if the kernel-idle-culling issue shows up again.

# %%
def get_predicted_positive_edges(phase, threshold=LOCKED_THRESHOLD):
    base_ei, base_ea = BASE_EDGES_BY_PHASE[phase]
    mask = getattr(data, f"{phase}_mask")
    target_idx = mask.nonzero(as_tuple=True)[0]
    t0 = time.time()
    metrics = score_edges(base_ei, base_ea, target_idx, NUM_NEIGHBORS_INFER,
                           INFER_BATCH_SIZE, threshold=threshold)
    probs = metrics["probs"]
    pred_pos_local = (probs >= threshold).nonzero(as_tuple=True)[0]
    pred_pos_global = target_idx[pred_pos_local]
    n_true_pos = int(data.y[target_idx].sum().item())
    print(f"[{phase}] scored {target_idx.numel():,} edges in {time.time()-t0:.1f}s -- "
          f"{pred_pos_global.numel():,} predicted positive (of {n_true_pos:,} true positive), "
          f"F1={metrics['f1']:.4f}")
    return pred_pos_global


predicted_positive = {}
for phase in ["train", "val", "test"]:
    predicted_positive[phase] = get_predicted_positive_edges(phase)

# Save so this expensive scoring pass doesn't need to be repeated if the kernel
# restarts partway through the rest of the notebook.
torch.save(predicted_positive, DATA_DIR / "predicted_positive_edges.pt")
print("Saved predicted_positive_edges.pt")

# %% [markdown]
# ## 6. Edge-id resolution, forward/reverse merge, and self-loop exclusion (Decision 3)
#
# `sub.e_id` (from `NeighborLoader`) gives the **global index into `data.edge_index` /
# `data.edge_attr` / `transaction_ids`** for every edge PyG actually sampled --
# this is what makes it possible to disambiguate parallel edges (multiple real
# transactions between the same account pair), which a plain `(src, dst)` match
# cannot do. This assert is here specifically to fail loudly, early, if a different
# PyG version doesn't expose it, rather than silently producing wrong transaction_id
# attributions later.

# %%
def _sanity_check_e_id_available():
    probe_nodes = data.edge_index[0, :64]
    probe_graph = Data(x=data.x, edge_index=train_base_ei, edge_attr=train_base_ea,
                        num_nodes=data.x.size(0))
    probe_loader = NeighborLoader(probe_graph, num_neighbors=[5, 5], input_nodes=probe_nodes,
                                   batch_size=probe_nodes.size(0), shuffle=False)
    probe_sub = next(iter(probe_loader))
    assert hasattr(probe_sub, "e_id"), (
        "NeighborLoader did not return `e_id` on this PyG version -- Section 6's "
        "transaction-id tracing (and everything downstream: merge, ranking, the "
        "final evidence JSON) depends on it. Check the installed torch_geometric "
        "version / sampling backend (pyg-lib vs torch-sparse) before proceeding."
    )
    print(f"e_id available -- sanity check passed. Example values: {probe_sub.e_id[:5].tolist()}")


_sanity_check_e_id_available()

# %%
def merge_and_filter(edge_index, e_id, importance_raw, edge_attr):
    """Collapse forward/reverse duplicates of the same transaction (max of their
    scores) and drop synthetic self-loops, returning one row per real transaction.

    importance_raw: 1D tensor, same length as edge_index.size(1) / e_id -- the raw
                     (already sigmoid-applied, in [0,1]) explainer score per edge
                     in the augmented subgraph.
    Returns: dict[int edge_idx (global, into data.edge_index)] -> float importance
    """
    is_synthetic_self_loop = (e_id == -1)

    # A genuine self-transaction (account paying itself) is NOT synthetic -- it has
    # e_id >= 0 and its real edge_attr, distinguishable via the is_self_loop column
    # (only meaningful when e_id >= 0; a synthetic row's edge_attr is all zero and
    # e_id is already -1, so this second check is a belt-and-suspenders assertion).
    if is_synthetic_self_loop.any():
        synth_ea = edge_attr[is_synthetic_self_loop]
        assert torch.allclose(synth_ea, torch.zeros_like(synth_ea)), \
            "A row flagged synthetic (e_id=-1) has non-zero edge_attr -- augment_subgraph_traceable bug."

    keep = ~is_synthetic_self_loop
    kept_eid = e_id[keep].tolist()
    kept_scores = importance_raw[keep].tolist()

    merged = {}
    for eid, score in zip(kept_eid, kept_scores):
        if eid not in merged or score > merged[eid]:
            merged[eid] = score
    return merged

# %% [markdown]
# ### 6b. Diagnostic: quantifying the same bug in the classifier's `augment_subgraph`
#
# Section 6 fixed an `e_id` desync in `augment_subgraph_traceable` caused by
# `add_remaining_self_loops` reordering existing self-loops instead of leaving them
# in place. **The exact same reordering affects `is_reverse` in the original,
# already-used `augment_subgraph`** (Section 3, copied verbatim from
# `train_baseline_comparison.ipynb`) -- it doesn't corrupt the message-passing
# structure itself (`edge_index`/`edge_attr` are unaffected), only the `is_reverse`
# bit fed into `EdgeEncoder` for rows near where a genuine self-transaction sat in
# the array. This cell measures how often that actually happens on real data, so
# the decision to (or not to) revisit `best_model_pna.pt` is based on a number, not
# a guess.

# %%
def _is_reverse_buggy(ei, ea, num_nodes):
    # Exact replica of augment_subgraph's is_reverse handling (Section 3), for comparison.
    is_reverse = torch.zeros(ei.size(1), 1, dtype=torch.float)
    non_self_loop = ei[0] != ei[1]
    rev_ei = ei[:, non_self_loop].flip(0)
    rev_ea = ea[non_self_loop]
    rev_is_reverse = torch.ones(rev_ei.size(1), 1, dtype=torch.float)
    ei2 = torch.cat([ei, rev_ei], dim=1)
    ea2 = torch.cat([ea, rev_ea], dim=0)
    is_reverse2 = torch.cat([is_reverse, rev_is_reverse], dim=0)
    n_before = ei2.size(1)
    ei3, ea3 = add_remaining_self_loops(ei2, ea2, fill_value=0.0, num_nodes=num_nodes)
    n_added = ei3.size(1) - n_before
    if n_added > 0:
        is_reverse2 = torch.cat([is_reverse2, torch.zeros(n_added, 1, dtype=torch.float)], dim=0)
    return ei3, is_reverse2


def _is_reverse_correct(ei, ea, num_nodes):
    # Section 6's fixed logic, is_reverse only (no eid needed for this comparison).
    is_reverse = torch.zeros(ei.size(1), 1, dtype=torch.float)
    non_self_loop = ei[0] != ei[1]
    rev_ei = ei[:, non_self_loop].flip(0)
    rev_is_reverse = torch.ones(rev_ei.size(1), 1, dtype=torch.float)
    ei2 = torch.cat([ei, rev_ei], dim=1)
    is_reverse2 = torch.cat([is_reverse, rev_is_reverse], dim=0)

    mask = ei2[0] != ei2[1]
    self_loop_mask = ~mask
    existing_loop_nodes = ei2[0, self_loop_mask]
    existing_loop_is_rev = is_reverse2[self_loop_mask]
    loop_index = torch.arange(num_nodes, dtype=ei2.dtype, device=ei2.device)
    loop_ei = loop_index.unsqueeze(0).repeat(2, 1)
    loop_is_reverse = torch.zeros(num_nodes, 1, dtype=torch.float)
    loop_is_reverse[existing_loop_nodes] = existing_loop_is_rev
    final_ei = torch.cat([ei2[:, mask], loop_ei], dim=1)
    final_is_reverse = torch.cat([is_reverse2[mask], loop_is_reverse], dim=0)
    return final_ei, final_is_reverse


@torch.no_grad()
def diagnose_is_reverse_bug(base_ei, base_ea, sample_target_idx, num_neighbors, batch_size=512):
    target_nodes = torch.unique(torch.cat([
        data.edge_index[0, sample_target_idx], data.edge_index[1, sample_target_idx]
    ]))
    base_graph = Data(x=data.x, edge_index=base_ei, edge_attr=base_ea, num_nodes=data.x.size(0))
    loader = NeighborLoader(base_graph, num_neighbors=num_neighbors, input_nodes=target_nodes,
                             batch_size=batch_size, shuffle=False)

    n_subgraphs, n_affected_subgraphs = 0, 0
    n_rows_checked, n_rows_mismatched = 0, 0
    structure_mismatch_warnings = 0

    for sub in loader:
        n_subgraphs += 1
        has_genuine_loop = (sub.edge_attr[:, IS_SELF_LOOP_COL] == 1).any().item()
        if not has_genuine_loop:
            continue
        n_affected_subgraphs += 1

        ei_buggy, is_rev_buggy = _is_reverse_buggy(sub.edge_index, sub.edge_attr, sub.num_nodes)
        ei_correct, is_rev_correct = _is_reverse_correct(sub.edge_index, sub.edge_attr, sub.num_nodes)

        if not torch.equal(ei_buggy, ei_correct):
            # Would mean our manual replica of add_remaining_self_loops doesn't
            # actually match the library's behavior on this PyG version -- if this
            # ever fires, the diagnostic's conclusion below isn't trustworthy, stop
            # and re-check add_remaining_self_loops' source for this installation.
            structure_mismatch_warnings += 1
            continue

        n_rows_checked += is_rev_buggy.size(0)
        n_rows_mismatched += (is_rev_buggy != is_rev_correct).sum().item()

    return {
        "subgraphs_checked": n_subgraphs,
        "subgraphs_with_genuine_self_loop": n_affected_subgraphs,
        "rows_checked": n_rows_checked,
        "rows_with_wrong_is_reverse": n_rows_mismatched,
        "structure_mismatch_warnings": structure_mismatch_warnings,
    }


DIAG_SAMPLE_SIZE = 5000
g = torch.Generator().manual_seed(RANDOM_SEED + 42)
diag_sample_idx = data.train_mask.nonzero(as_tuple=True)[0]
diag_sample_idx = diag_sample_idx[torch.randperm(diag_sample_idx.size(0), generator=g)[:DIAG_SAMPLE_SIZE]]

diag_result = diagnose_is_reverse_bug(train_base_ei, train_base_ea, diag_sample_idx, NUM_NEIGHBORS_INFER)
print(json.dumps(diag_result, indent=2))

if diag_result["structure_mismatch_warnings"] > 0:
    print("\n*** Manual replica of add_remaining_self_loops did not match the library's actual "
          "behavior on this PyG version for some subgraphs -- treat the numbers above as "
          "unreliable and flag this before drawing any conclusion. ***")
elif diag_result["subgraphs_with_genuine_self_loop"] == 0:
    print("\nNo genuine self-transactions turned up in this sample -- the bug is real but may "
          "not have materially affected best_model_pna.pt. Consider re-running with a larger "
          "DIAG_SAMPLE_SIZE before concluding it's a non-issue.")
else:
    pct_rows = 100 * diag_result["rows_with_wrong_is_reverse"] / max(diag_result["rows_checked"], 1)
    print(f"\n{diag_result['subgraphs_with_genuine_self_loop']:,} / {diag_result['subgraphs_checked']:,} "
          f"sampled subgraphs contained a genuine self-transaction, affecting "
          f"{diag_result['rows_with_wrong_is_reverse']:,} edge rows ({pct_rows:.4f}% of all rows checked) "
          f"with a flipped is_reverse bit. Use this rate to judge whether best_model_pna.pt is worth "
          f"revisiting -- this notebook doesn't retrain the classifier either way.")

# %% [markdown]
# ## 7. The PGExplainer MLP and the masking mechanism
#
# **Input representation:** for a candidate edge `(u, v)` inside the target
# transaction's k-hop subgraph, the MLP sees
# `concat(h_u, h_v, h_target_src, h_target_dst)` -- not just `concat(h_u, h_v)`.
# Reasoning: this is an *edge classification* task with two endpoints (`src`, `dst`),
# not a single-node task, and the same neighboring transaction can matter differently
# depending on which specific transaction is being explained. Conditioning the mask
# on both the candidate edge's own embeddings and the target's embeddings is the
# direct two-endpoint extension of the original PGExplainer paper's node-level design
# (which conditions on the single node being explained).
#
# **Masking mechanism:** `set_masks(model.node_encoder, mask, aug_ei, apply_sigmoid=True)`
# from `torch_geometric.explain.algorithm.utils` -- this patches every `MessagePassing`
# submodule (both `PNAConv` layers here) at the `propagate()` level, multiplying each
# edge's contribution to aggregation by its mask value. It's architecture-agnostic
# (works identically for `PNAConv`, `GCNConv`, etc.) since it hooks the base class,
# not `PNAConv`'s specific `forward()` signature -- this is exactly why Decision 1
# didn't need to rule this specific utility out, only the higher-level `Explainer` API.
# `clear_masks()` must be called after every forward pass or the mask leaks into the
# next batch.

# %%
class PGExplainerMLP(nn.Module):
    def __init__(self, hidden_dim, explainer_hidden=EXPLAINER_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, explainer_hidden),
            nn.ReLU(),
            nn.Linear(explainer_hidden, 1),
        )

    def forward(self, h_u, h_v, h_target_src, h_target_dst):
        z = torch.cat([h_u, h_v, h_target_src, h_target_dst], dim=1)
        return self.net(z).squeeze(-1)  # raw logits, sigmoid applied by the caller


def concrete_relax(logits, temperature=TEMPERATURE, training=True):
    # Binary concrete relaxation (Gumbel-sigmoid) -- differentiable stand-in for a
    # hard 0/1 edge-keep decision. At eval time we just take sigmoid(logits) with no
    # noise (deterministic soft mask).
    if training:
        eps = 1e-8
        u = torch.rand_like(logits)
        gumbel_noise = torch.log(u + eps) - torch.log(1 - u + eps)
        return torch.sigmoid((logits + gumbel_noise) / temperature)
    return torch.sigmoid(logits)


explainer_mlp = PGExplainerMLP(hidden_dim=checkpoint["hyperparameters"]["hidden_dim"]).to(DEVICE)
explainer_optimizer = torch.optim.Adam(explainer_mlp.parameters(), lr=EXPLAINER_LR)
print(f"PGExplainer MLP: {sum(p.numel() for p in explainer_mlp.parameters()):,} trainable params.")

# %% [markdown]
# ## 8. Per-target forward pass and the self-supervised loss
#
# **Per target transaction, two forward passes through the frozen classifier:**
#
# 1. **Unmasked** (`torch.no_grad()`): gives `p_original` (the actual model prediction
#    we're explaining) and the node embeddings `h` used as the explainer MLP's *input*
#    features (representing "what the neighborhood looks like", held constant).
# 2. **Masked** (gradient enabled, but only the mask carries a gradient back to
#    `explainer_mlp` -- the classifier's own weights are frozen so no gradient
#    accumulates there): the explainer MLP scores every real candidate edge, values
#    are merged by transaction (Decision 3, applied *before* the mask is built --
#    both the forward and reverse copy of a transaction get the same merged mask
#    value), synthetic self-loops are always kept at mask=1 (they're a structural
#    requirement for PNA's aggregation, not evidence), and the resulting `p_masked`
#    is compared against `p_original` via binary cross-entropy (this is the standard
#    PGExplainer "explain the model's own prediction" self-supervised setup -- no
#    ground-truth label is used anywhere in this loss).
#
# **Regularization:** mean mask value (sparsity -- most edges should end up
# unimportant) + mask entropy (confidence -- masks should land near 0 or 1, not
# hover at 0.5).

# %%
def build_full_mask_tensor(e_id, merged_scores, device):
    # merged_scores: dict[int eid] -> float/tensor score, from merge_and_filter's
    # eid-keyed max. Self-loops (eid == -1) always get mask=1 (never suppressed).
    out = torch.ones(e_id.size(0), device=device)
    for i, eid in enumerate(e_id.tolist()):
        if eid != -1 and eid in merged_scores:
            out[i] = merged_scores[eid]
    return out


def explainer_forward_and_loss(target_edge_idx, phase, training=True):
    base_ei, base_ea = BASE_EDGES_BY_PHASE[phase]
    src = data.edge_index[0, target_edge_idx].item()
    dst = data.edge_index[1, target_edge_idx].item()
    target_ea = data.edge_attr[target_edge_idx:target_edge_idx + 1].to(DEVICE)

    seed_nodes = torch.tensor([src, dst], dtype=torch.long)
    base_graph = Data(x=data.x, edge_index=base_ei, edge_attr=base_ea, num_nodes=data.x.size(0))
    loader = NeighborLoader(base_graph, num_neighbors=NUM_NEIGHBORS_INFER, input_nodes=seed_nodes,
                             batch_size=seed_nodes.size(0), shuffle=False)
    sub = next(iter(loader))

    aug_ei, aug_ea, aug_is_rev, aug_eid = augment_subgraph_traceable(
        sub.edge_index, sub.edge_attr, sub.e_id, sub.num_nodes)
    x_dev = sub.x.to(DEVICE)
    aug_ei, aug_ea, aug_is_rev = aug_ei.to(DEVICE), aug_ea.to(DEVICE), aug_is_rev.to(DEVICE)
    aug_eid = aug_eid.to(DEVICE)

    seed_ids = sub.n_id[:sub.batch_size]
    local_src = _local_lookup(seed_ids, torch.tensor([src], device=seed_ids.device))
    local_dst = _local_lookup(seed_ids, torch.tensor([dst], device=seed_ids.device))

    # --- Pass 1: unmasked, no_grad -- context embeddings + the prediction we explain ---
    with torch.no_grad():
        h0 = model.encode(x_dev, aug_ei, aug_ea, aug_is_rev)
        h_src0 = h0[local_src.to(DEVICE)]
        h_dst0 = h0[local_dst.to(DEVICE)]
        p_original = torch.sigmoid(model.score(h_src0, h_dst0, target_ea))

    # --- Explainer MLP: score every real candidate edge, merge fwd/rev, fix self-loops ---
    h_u = h0[aug_ei[0]]
    h_v = h0[aug_ei[1]]
    h_tgt_src_bcast = h_src0.expand(aug_ei.size(1), -1)
    h_tgt_dst_bcast = h_dst0.expand(aug_ei.size(1), -1)
    raw_logits = explainer_mlp(h_u, h_v, h_tgt_src_bcast, h_tgt_dst_bcast)
    raw_scores = concrete_relax(raw_logits, training=training)  # in [0, 1]

    merged = merge_and_filter(aug_ei, aug_eid, raw_scores.detach(), aug_ea)
    # Re-derive a *differentiable* merged score per instance (max via detached
    # selection, but keep gradient flowing through whichever copy provided the max)
    # -- simplest correct way: rebuild per-eid max using the live (non-detached)
    # raw_scores tensor via scatter-max semantics implemented with a small loop
    # over the (few hundred) unique eids in this subgraph.
    eid_list = aug_eid.tolist()
    best_idx_per_eid = {}
    for i, eid in enumerate(eid_list):
        if eid == -1:
            continue
        if eid not in best_idx_per_eid or raw_scores[i].item() > raw_scores[best_idx_per_eid[eid]].item():
            best_idx_per_eid[eid] = i
    merged_live = {eid: raw_scores[i] for eid, i in best_idx_per_eid.items()}

    mask = torch.ones(aug_ei.size(1), device=DEVICE)
    for i, eid in enumerate(eid_list):
        if eid != -1 and eid in merged_live:
            mask[i] = merged_live[eid]

    # --- Pass 2: masked -- mask carries a gradient back to explainer_mlp only ---
    set_masks(model.node_encoder, mask, aug_ei, apply_sigmoid=False)  # already in [0,1]
    h_masked = model.encode(x_dev, aug_ei, aug_ea, aug_is_rev)
    clear_masks(model.node_encoder)
    h_src_m = h_masked[local_src.to(DEVICE)]
    h_dst_m = h_masked[local_dst.to(DEVICE)]
    logit_masked = model.score(h_src_m, h_dst_m, target_ea)

    loss_pred = F.binary_cross_entropy_with_logits(logit_masked, p_original.detach())

    real_edge_mask_values = torch.stack([mask[i] for i, eid in enumerate(eid_list) if eid != -1]) \
        if any(eid != -1 for eid in eid_list) else torch.zeros(1, device=DEVICE)
    loss_size = real_edge_mask_values.mean()
    eps = 1e-8
    loss_entropy = -(real_edge_mask_values * torch.log(real_edge_mask_values + eps) +
                      (1 - real_edge_mask_values) * torch.log(1 - real_edge_mask_values + eps)).mean()

    total_loss = loss_pred + MASK_SPARSITY_COEF * loss_size + MASK_ENTROPY_COEF * loss_entropy

    del sub, aug_ei, aug_ea, aug_is_rev, h0, h_masked
    return {
        "loss": total_loss, "loss_pred": loss_pred.item(), "loss_size": loss_size.item(),
        "p_original": p_original.item(), "p_masked": torch.sigmoid(logit_masked).item(),
        "merged_scores": merged,  # eid -> raw importance, for Sections 10/11/14
    }

# %% [markdown]
# ### 8b. Resume support
#
# Explainer training (Section 9) is the expensive step (hours, not minutes). If a
# previous run already produced `pgexplainer_state.pt` + `pgexplainer_config.json`,
# Sections 9, 10, 11, and 12 below **load and skip re-computing** instead of
# retraining/re-sweeping from scratch -- this makes it safe to just re-run this
# whole script end to end (e.g. after fixing a bug in a *later* section like 13-15)
# without burning another several hours. Delete both files first if you actually
# want a full, from-scratch re-run (e.g. after changing `EXPLAINER_EPOCHS`,
# `MASK_SPARSITY_COEF`, or anything else that changes what training would produce).

# %%
EXPLAINER_STATE_PATH = DATA_DIR / "pgexplainer_state.pt"
EXPLAINER_CONFIG_PATH = DATA_DIR / "pgexplainer_config.json"

# These are deliberately INDEPENDENT checks, not one combined flag. An earlier
# version checked both files together, which meant a crash during the val/test
# fidelity sweep (Sections 10/11 -- observed in practice: repeated per-target
# NeighborLoader/Data construction across hundreds of calls grew memory usage
# until the OS OOM-killer silently ended the process, no Python traceback) lost
# several hours of ALREADY-COMPLETED training, because the trained weights were
# only ever written at the very end (old Section 12), after the sweep. Now
# Section 9 checkpoints EXPLAINER_STATE_PATH immediately once training itself
# finishes, before the sweep runs at all -- so a crash during the sweep only
# costs the (much cheaper) sweep, never the training.
EXPLAINER_TRAINED = EXPLAINER_STATE_PATH.exists()
EXPLAINER_FULLY_EVALUATED = EXPLAINER_CONFIG_PATH.exists()

print(f"Explainer weights already trained and checkpointed: {EXPLAINER_TRAINED}")
print(f"Full val-sweep + test-fidelity already saved: {EXPLAINER_FULLY_EVALUATED}")
if not EXPLAINER_TRAINED:
    print("-> Section 9 will train from scratch.")
if EXPLAINER_TRAINED and not EXPLAINER_FULLY_EVALUATED:
    print("-> Section 9 will load the checkpoint and skip training; "
          "Sections 10/11 will still run the sweep (not completed last time).")

# %% [markdown]
# ## 9. Train the explainer MLP on `train_mask` predicted-positive transactions
#
# Gradient accumulation over `EXPLAINER_BATCH_SIZE` targets per optimizer step (each
# target needs its own small `NeighborLoader` call, so this batches the *loss*, not
# the subgraph construction -- simpler to get right than sharing subgraphs across a
# batch of targets, at some throughput cost. **If this is too slow in practice**, the
# natural speed-up mirrors `train_one_epoch`'s trick in the baseline notebook: union
# the seed nodes of several targets into one `NeighborLoader` call, computing the
# *unmasked* pass once for the whole batch -- the masked pass still has to run
# per-target since the mask depends on which transaction is being explained, but
# that's the cheaper of the two passes to repeat.

# %%
def train_explainer_epoch(target_edges, phase, epoch_seed):
    explainer_mlp.train()
    g = torch.Generator().manual_seed(epoch_seed)
    perm = target_edges[torch.randperm(target_edges.size(0), generator=g)]

    total_loss, n_seen = 0.0, 0
    for start in range(0, perm.size(0), EXPLAINER_BATCH_SIZE):
        batch_targets = perm[start:start + EXPLAINER_BATCH_SIZE]
        explainer_optimizer.zero_grad()
        batch_loss = 0.0
        for t in batch_targets:
            out = explainer_forward_and_loss(t.item(), phase, training=True)
            batch_loss = batch_loss + out["loss"]
        batch_loss = batch_loss / batch_targets.size(0)
        batch_loss.backward()
        explainer_optimizer.step()

        total_loss += batch_loss.item() * batch_targets.size(0)
        n_seen += batch_targets.size(0)

    return total_loss / max(n_seen, 1)


if EXPLAINER_TRAINED:
    _ckpt = torch.load(EXPLAINER_STATE_PATH, weights_only=False)
    explainer_mlp.load_state_dict(_ckpt["explainer_state_dict"])
    explainer_history = _ckpt.get("history", [])
    print(f"Loaded pgexplainer_state.pt -- skipping training "
          f"({len(explainer_history)} epochs already recorded in a previous run).")
else:
    # NOTE: training on every predicted-positive train edge each epoch may be more
    # than needed for a small MLP to converge -- if predicted_positive["train"] is
    # very large, consider capping it (e.g. a fixed random subsample per epoch,
    # same spirit as the classifier's NEG_RATIO subsampling) rather than reducing
    # EXPLAINER_EPOCHS.
    train_targets = predicted_positive["train"]
    print(f"Training explainer on {train_targets.numel():,} predicted-positive train transactions.")

    explainer_history = []
    for epoch in range(1, EXPLAINER_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_explainer_epoch(train_targets, "train", epoch_seed=RANDOM_SEED + epoch)
        explainer_history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"  [explainer] epoch {epoch:2d}/{EXPLAINER_EPOCHS}  train_loss={train_loss:.4f}  ({time.time()-t0:.1f}s)")

    # Checkpoint IMMEDIATELY -- Sections 10/11 below make hundreds of per-target
    # NeighborLoader/Data calls and have already OOM-killed once on this
    # environment with no Python-catchable error. Don't let that risk the many
    # hours of training that already succeeded.
    torch.save({"explainer_state_dict": explainer_mlp.state_dict(),
                "hidden_dim": checkpoint["hyperparameters"]["hidden_dim"],
                "explainer_hidden": EXPLAINER_HIDDEN,
                "history": explainer_history},
               EXPLAINER_STATE_PATH)
    EXPLAINER_TRAINED = True
    print(f"Checkpointed trained explainer to {EXPLAINER_STATE_PATH.name} "
          f"(before running the val/test fidelity sweeps below).")

# %% [markdown]
# ## 10. Fidelity+ / Fidelity- and choosing the final top-k (Decision 5)
#
# - **Fidelity+** (necessity): drop the top-k explained edges, keep everything else --
#   `p_original - p_without_topk`. High is good (removing the "evidence" should hurt
#   the model's confidence).
# - **Fidelity-** (sufficiency): keep *only* the top-k explained edges (+ self-loops,
#   structurally required) -- `p_original - p_only_topk`. Low (near 0) is good (the
#   top-k alone should be close to sufficient for the same prediction).
# - **Characterization score**: `2*Fid+*(1-Fid-) / (Fid+ + (1-Fid-))`, an F1-style
#   combination of the two, used to pick `k`.
# - **Random-k baseline**: same two metrics but with `k` *randomly chosen* real edges
#   instead of the top-k by explainer score, evaluated on the same sample -- if
#   PGExplainer isn't doing better than picking random neighbors, the whole exercise
#   isn't adding value over "just show all neighbors."
#
# Swept on a **sample of `val_mask` predicted-positive transactions** (not the val
# transactions used to train anything -- this is purely a metric computation, no
# gradient, so reusing predicted-positive val edges here doesn't violate any
# train/val separation the classifier itself relies on).

# %%
def get_target_context(target_edge_idx, phase):
    base_ei, base_ea = BASE_EDGES_BY_PHASE[phase]
    src = data.edge_index[0, target_edge_idx].item()
    dst = data.edge_index[1, target_edge_idx].item()
    target_ea = data.edge_attr[target_edge_idx:target_edge_idx + 1].to(DEVICE)

    seed_nodes = torch.tensor([src, dst], dtype=torch.long)
    base_graph = Data(x=data.x, edge_index=base_ei, edge_attr=base_ea, num_nodes=data.x.size(0))

    # FIX: NeighborLoader samples a RANDOM subset of neighbors whenever a node's
    # degree exceeds the fanout (NUM_NEIGHBORS_INFER), so without a fixed seed the
    # exact same transaction could resolve to a different subgraph -- and therefore
    # a different model probability, a different ranked evidence list, even a
    # different Fidelity score -- on every call (observed directly: one demo
    # transaction flipped from p_fraud=0.997+ in Section 5's scoring pass to
    # p_fraud=0.002 here). Seeding on target_edge_idx makes every call for the same
    # transaction deterministic. This intentionally does NOT touch
    # explainer_forward_and_loss's own subgraph sampling (Section 8) -- that one
    # benefits from staying stochastic across epochs during training; only
    # explain-time calls (this function, used by Sections 10/11/14/15) need to be
    # reproducible.
    torch.manual_seed(target_edge_idx)
    np.random.seed(target_edge_idx % (2**32 - 1))
    loader = NeighborLoader(base_graph, num_neighbors=NUM_NEIGHBORS_INFER, input_nodes=seed_nodes,
                             batch_size=seed_nodes.size(0), shuffle=False)
    sub = next(iter(loader))

    aug_ei, aug_ea, aug_is_rev, aug_eid = augment_subgraph_traceable(
        sub.edge_index, sub.edge_attr, sub.e_id, sub.num_nodes)
    x_dev = sub.x.to(DEVICE)
    aug_ei, aug_ea, aug_is_rev, aug_eid = (aug_ei.to(DEVICE), aug_ea.to(DEVICE),
                                            aug_is_rev.to(DEVICE), aug_eid.to(DEVICE))
    seed_ids = sub.n_id[:sub.batch_size]
    local_src = _local_lookup(seed_ids, torch.tensor([src], device=seed_ids.device)).to(DEVICE)
    local_dst = _local_lookup(seed_ids, torch.tensor([dst], device=seed_ids.device)).to(DEVICE)

    with torch.no_grad():
        h0 = model.encode(x_dev, aug_ei, aug_ea, aug_is_rev)
        h_src0, h_dst0 = h0[local_src], h0[local_dst]
        p_original = torch.sigmoid(model.score(h_src0, h_dst0, target_ea)).item()

    return {"src": src, "dst": dst, "target_ea": target_ea, "x_dev": x_dev,
            "aug_ei": aug_ei, "aug_ea": aug_ea, "aug_is_rev": aug_is_rev, "aug_eid": aug_eid,
            "local_src": local_src, "local_dst": local_dst, "h0": h0, "p_original": p_original,
            "target_edge_idx": target_edge_idx}


@torch.no_grad()
def get_explainer_scores(ctx):
    explainer_mlp.eval()
    aug_ei, h0 = ctx["aug_ei"], ctx["h0"]
    h_src0, h_dst0 = h0[ctx["local_src"]], h0[ctx["local_dst"]]
    h_u, h_v = h0[aug_ei[0]], h0[aug_ei[1]]
    h_tgt_src_bcast = h_src0.expand(aug_ei.size(1), -1)
    h_tgt_dst_bcast = h_dst0.expand(aug_ei.size(1), -1)
    scores = torch.sigmoid(explainer_mlp(h_u, h_v, h_tgt_src_bcast, h_tgt_dst_bcast))
    merged = merge_and_filter(aug_ei, ctx["aug_eid"], scores, ctx["aug_ea"])
    # FIX: the target transaction is itself a real edge between src/dst, so it's
    # frequently sampled into its own k-hop neighborhood (near-certain for phase=
    # "test", where the base graph is the complete graph including the target
    # edge). Without this, a transaction could show up as "evidence" for its own
    # fraud classification -- a circular citation with zero explanatory value.
    merged.pop(ctx["target_edge_idx"], None)
    return merged


def _build_binary_mask(ctx, target_eids, mode):
    eid_list = ctx["aug_eid"].tolist()
    mask = torch.zeros(len(eid_list), device=DEVICE)
    for i, eid in enumerate(eid_list):
        if eid == -1:
            mask[i] = 1.0  # self-loops: structural, never suppressed
        elif mode == "remove":
            mask[i] = 0.0 if eid in target_eids else 1.0
        elif mode == "keep_only":
            mask[i] = 1.0 if eid in target_eids else 0.0
    return mask


@torch.no_grad()
def _score_with_mask(ctx, mask):
    set_masks(model.node_encoder, mask, ctx["aug_ei"], apply_sigmoid=False)
    h = model.encode(ctx["x_dev"], ctx["aug_ei"], ctx["aug_ea"], ctx["aug_is_rev"])
    p = torch.sigmoid(model.score(h[ctx["local_src"]], h[ctx["local_dst"]], ctx["target_ea"])).item()
    clear_masks(model.node_encoder)
    return p


def compute_fidelity_pair(ctx, topk_eids):
    p_without = _score_with_mask(ctx, _build_binary_mask(ctx, topk_eids, "remove"))
    p_only = _score_with_mask(ctx, _build_binary_mask(ctx, topk_eids, "keep_only"))
    return ctx["p_original"] - p_without, ctx["p_original"] - p_only


def characterization_score(fid_plus, fid_minus, eps=1e-8):
    return 2 * fid_plus * (1 - fid_minus) / (fid_plus + (1 - fid_minus) + eps)

# %%
if EXPLAINER_FULLY_EVALUATED:
    with open(EXPLAINER_CONFIG_PATH) as f:
        _saved_config = json.load(f)
    k_sweep_rows = _saved_config["val_k_sweep"]
    k_sweep_df = pd.DataFrame(k_sweep_rows)
    FINAL_TOP_K = _saved_config["final_top_k"]
    print(f"Loaded cached val k-sweep from a previous run -- chosen top-k = {FINAL_TOP_K}.")
else:
    VAL_FIDELITY_SAMPLE_SIZE = 300  # raise if this environment's budget allows
    GC_EVERY_N = 25  # periodic cleanup -- each iteration builds a fresh Data +
                      # NeighborLoader; this loop OOM-killed the process (silently,
                      # no Python traceback) on this environment before this was added

    g = torch.Generator().manual_seed(RANDOM_SEED)
    val_sample_idx = predicted_positive["val"][
        torch.randperm(predicted_positive["val"].size(0), generator=g)[:VAL_FIDELITY_SAMPLE_SIZE]
    ]

    k_sweep_rows = []
    for k in TOP_K_CANDIDATES:
        fid_plus_list, fid_minus_list = [], []
        rand_fid_plus_list, rand_fid_minus_list = [], []
        rng = np.random.default_rng(RANDOM_SEED)

        for i, t in enumerate(val_sample_idx.tolist()):
            ctx = get_target_context(t, "val")
            merged = get_explainer_scores(ctx)
            real_eids = list(merged.keys())
            if len(real_eids) == 0:
                continue

            ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:k]
            topk_set = set(eid for eid, _ in ranked)
            fp, fm = compute_fidelity_pair(ctx, topk_set)
            fid_plus_list.append(fp)
            fid_minus_list.append(fm)

            k_rand = min(k, len(real_eids))
            rand_set = set(rng.choice(real_eids, size=k_rand, replace=False).tolist())
            rfp, rfm = compute_fidelity_pair(ctx, rand_set)
            rand_fid_plus_list.append(rfp)
            rand_fid_minus_list.append(rfm)

            del ctx
            if (i + 1) % GC_EVERY_N == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        row = {
            "k": k,
            "fid_plus": float(np.mean(fid_plus_list)), "fid_minus": float(np.mean(fid_minus_list)),
            "char_score": float(characterization_score(np.mean(fid_plus_list), np.mean(fid_minus_list))),
            "random_fid_plus": float(np.mean(rand_fid_plus_list)),
            "random_fid_minus": float(np.mean(rand_fid_minus_list)),
            "random_char_score": float(characterization_score(np.mean(rand_fid_plus_list), np.mean(rand_fid_minus_list))),
        }
        k_sweep_rows.append(row)
        print(f"  k={k:2d}  Fid+={row['fid_plus']:.4f}  Fid-={row['fid_minus']:.4f}  "
              f"char={row['char_score']:.4f}   |  random: Fid+={row['random_fid_plus']:.4f}  "
              f"Fid-={row['random_fid_minus']:.4f}  char={row['random_char_score']:.4f}")

    k_sweep_df = pd.DataFrame(k_sweep_rows)
    FINAL_TOP_K = int(k_sweep_df.loc[k_sweep_df["char_score"].idxmax(), "k"])
    print(f"\nChosen top-k = {FINAL_TOP_K} (best characterization score on val)")
k_sweep_df

# %% [markdown]
# ## 11. Final faithfulness check on `test_mask` (touched once)
#
# Same computation as Section 10, fixed at `FINAL_TOP_K`, run once on a sample of
# `test_mask` predicted-positive transactions -- confirms the chosen `k` generalizes,
# never used to choose anything.

# %%
if EXPLAINER_FULLY_EVALUATED:
    with open(EXPLAINER_CONFIG_PATH) as f:
        _saved_config = json.load(f)
    test_fidelity_summary = _saved_config["test_fidelity"]
    print("Loaded cached test fidelity summary from a previous run:")
    print(json.dumps(test_fidelity_summary, indent=2))
else:
    TEST_FIDELITY_SAMPLE_SIZE = 300
    GC_EVERY_N = 25  # see Section 10 -- same per-target NeighborLoader/Data memory
                      # growth risk applies here

    g = torch.Generator().manual_seed(RANDOM_SEED + 999)
    test_sample_idx = predicted_positive["test"][
        torch.randperm(predicted_positive["test"].size(0), generator=g)[:TEST_FIDELITY_SAMPLE_SIZE]
    ]

    test_fid_plus, test_fid_minus = [], []
    test_rand_fid_plus, test_rand_fid_minus = [], []
    rng = np.random.default_rng(RANDOM_SEED + 999)

    for i, t in enumerate(test_sample_idx.tolist()):
        ctx = get_target_context(t, "test")
        merged = get_explainer_scores(ctx)
        real_eids = list(merged.keys())
        if len(real_eids) == 0:
            continue
        ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:FINAL_TOP_K]
        topk_set = set(eid for eid, _ in ranked)
        fp, fm = compute_fidelity_pair(ctx, topk_set)
        test_fid_plus.append(fp)
        test_fid_minus.append(fm)

        k_rand = min(FINAL_TOP_K, len(real_eids))
        rand_set = set(rng.choice(real_eids, size=k_rand, replace=False).tolist())
        rfp, rfm = compute_fidelity_pair(ctx, rand_set)
        test_rand_fid_plus.append(rfp)
        test_rand_fid_minus.append(rfm)

        del ctx
        if (i + 1) % GC_EVERY_N == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    test_fidelity_summary = {
        "top_k": FINAL_TOP_K,
        "fidelity_plus_mean": float(np.mean(test_fid_plus)), "fidelity_plus_std": float(np.std(test_fid_plus)),
        "fidelity_minus_mean": float(np.mean(test_fid_minus)), "fidelity_minus_std": float(np.std(test_fid_minus)),
        "char_score": float(characterization_score(np.mean(test_fid_plus), np.mean(test_fid_minus))),
        "random_fidelity_plus_mean": float(np.mean(test_rand_fid_plus)),
        "random_fidelity_minus_mean": float(np.mean(test_rand_fid_minus)),
        "random_char_score": float(characterization_score(np.mean(test_rand_fid_plus), np.mean(test_rand_fid_minus))),
    }
    print(json.dumps(test_fidelity_summary, indent=2))

# %% [markdown]
# ## 12. Save the trained explainer + config

# %%
if EXPLAINER_FULLY_EVALUATED:
    print("pgexplainer_config.json already on disk from a previous run -- not "
          "overwriting. Delete it first to force a re-save.")
else:
    # explainer_state_dict was already checkpointed right after training finished
    # (Section 9) -- only the sweep/test results need saving here.
    pgexplainer_config = {
        "final_top_k": FINAL_TOP_K,
        "locked_threshold": LOCKED_THRESHOLD,
        "training": {"epochs": EXPLAINER_EPOCHS, "lr": EXPLAINER_LR, "batch_size": EXPLAINER_BATCH_SIZE,
                     "mask_sparsity_coef": MASK_SPARSITY_COEF, "mask_entropy_coef": MASK_ENTROPY_COEF,
                     "history": explainer_history},
        "val_k_sweep": k_sweep_rows,
        "test_fidelity": test_fidelity_summary,
    }
    with open(EXPLAINER_CONFIG_PATH, "w") as f:
        json.dump(pgexplainer_config, f, indent=2)

    print("Saved pgexplainer_config.json")

# %% [markdown]
# ## 13. Feature-level attribution for the target transaction (Integrated Gradients)
#
# Only the direct path is needed: `target_edge_attr -> EdgeEncoder -> concat(h_src,
# h_dst, target_edge_emb) -> head -> prob`. `h_src`/`h_dst` are held fixed (already
# computed, no message-passing re-run here) -- this is why this is cheap. Continuous
# + binary columns (`edge_attr[:, :edge_cat_start]`) are interpolated from a
# zero baseline; categorical columns are passed through unchanged and excluded from
# the gradient path entirely (Decision confirmed: no IG score for categoricals,
# they're reported as plain facts).

# %%
def integrated_gradients_target(ctx, steps=IG_STEPS):
    h_src0 = ctx["h0"][ctx["local_src"]].detach()
    h_dst0 = ctx["h0"][ctx["local_dst"]].detach()
    raw_ea = ctx["target_ea"].detach()
    cont_part = raw_ea[:, :edge_cat_start]
    cat_part = raw_ea[:, edge_cat_start:]
    baseline_cont = torch.zeros_like(cont_part)

    grads_accum = torch.zeros_like(cont_part)
    for step in range(1, steps + 1):
        alpha = step / steps
        interp_cont = (baseline_cont + alpha * (cont_part - baseline_cont)).clone().requires_grad_(True)
        interp_ea = torch.cat([interp_cont, cat_part], dim=1)
        edge_emb = model.edge_encoder(interp_ea)  # is_reverse=None -> defaults to 0, correct for a target edge
        z = torch.cat([h_src0, h_dst0, edge_emb], dim=1)
        prob = torch.sigmoid(model.head(z).squeeze(-1))
        grad = torch.autograd.grad(prob, interp_cont)[0]
        grads_accum = grads_accum + grad

    avg_grad = grads_accum / steps
    attribution = (cont_part - baseline_cont) * avg_grad
    return attribution.squeeze(0).detach().cpu()  # length = edge_cat_start


def decode_transaction_features(edge_attr_row):
    """edge_attr_row: 1D array/tensor, length == len(EDGE_FEATURE_COLS), in the
    exact scaled/encoded form stored in data.edge_attr. Returns a plain dict of
    human-readable values -- inverse-transforms the scaler for continuous columns,
    decodes categorical codes back to their original strings, passes binary/count
    columns through unchanged.
    """
    row = edge_attr_row.cpu().numpy() if torch.is_tensor(edge_attr_row) else np.asarray(edge_attr_row)
    decoded = {}

    scaled_cols = [c for c in EDGE_FEATURE_COLS[:edge_cat_start] if c in SCALED_TO_RAW]
    if scaled_cols:
        scaled_vals = np.array([[row[EDGE_FEATURE_COLS.index(c)] for c in scaled_cols]])
        # NOTE: edge_scaler.inverse_transform undoes STANDARDIZATION only, taking values
        # back to the log-space the scaler was fit on (log_amount_paid etc.). If the LLM
        # prompt needs raw currency units, confirm against preprocessing_and_feature_
        # engineering.ipynb's exact transform (np.log1p vs np.log) and apply the matching
        # inverse (np.expm1 vs np.exp) here -- left as log-space for now, flagged rather
        # than silently guessed.
        raw_log_vals = edge_scaler.inverse_transform(scaled_vals)[0]
        for c, v in zip(scaled_cols, raw_log_vals):
            decoded[SCALED_TO_RAW[c]] = float(v)

    for c in EDGE_FEATURE_COLS[:edge_cat_start]:
        if c not in SCALED_TO_RAW:
            decoded[c] = float(row[EDGE_FEATURE_COLS.index(c)])

    for c in CAT_COLS:
        col_name = f"{c}_enc"
        idx = EDGE_FEATURE_COLS.index(col_name)
        code_val = int(row[idx])
        decoded[c] = cat_decoders.get(c, {}).get(code_val, f"<UNK code {code_val}>")

    return decoded

# %% [markdown]
# ## 14. `explain_transaction(transaction_id)` -- the reusable entry point
#
# This is the function that gets carried, unchanged, into the FastAPI backend
# (Decision 7). It resolves the transaction's phase automatically (so message
# passing always uses the correct phase-scoped base graph -- a train transaction
# never gets explained using `val`/`test` edges), builds the ranked, merged,
# self-loop-filtered evidence list (Decision 3), attaches the Integrated Gradients
# breakdown to the target transaction only (Decision: Scenario B, target-only), and
# reports a per-call Fidelity+/- sanity number so a caller can flag unusually
# low-confidence explanations.

# %%
def explain_transaction(transaction_id, top_k=None):
    if transaction_id not in txn_id_to_edge_idx:
        raise ValueError(f"Unknown transaction_id: {transaction_id}")
    edge_idx = txn_id_to_edge_idx[transaction_id]
    phase = txn_id_to_split.get(transaction_id)
    if phase not in BASE_EDGES_BY_PHASE:
        raise ValueError(f"transaction_id {transaction_id} has unexpected split value: {phase!r}")
    k = top_k if top_k is not None else FINAL_TOP_K

    ctx = get_target_context(edge_idx, phase)
    merged_scores = get_explainer_scores(ctx)  # target's own edge already excluded
    ranked = sorted(merged_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    contributing_transactions = []
    for eid, score in ranked:
        contributing_transactions.append({
            "transaction_id": edge_idx_to_txn_id.get(eid),
            "importance": round(float(score), 4),
            "features": decode_transaction_features(data.edge_attr[eid]),
        })

    ig_attr = integrated_gradients_target(ctx)
    ig_cols = EDGE_FEATURE_COLS[:edge_cat_start]
    feature_attribution = {SCALED_TO_RAW.get(c, c): round(float(v), 4) for c, v in zip(ig_cols, ig_attr.tolist())}

    topk_set = set(eid for eid, _ in ranked)
    fid_plus, fid_minus = compute_fidelity_pair(ctx, topk_set)

    # FIX: get_target_context is now deterministic for a GIVEN call, but its
    # per-target NeighborLoader sampling is a genuinely different sampling context
    # from Section 5's batched scoring pass that originally flagged this
    # transaction as predicted-positive -- the two can land on different random
    # neighbor draws for high-degree nodes, and therefore different probabilities.
    # Rather than pretend perfect consistency across two different sampling
    # contexts, surface it honestly: flag explicitly when this explanation's own
    # freshly-computed probability no longer clears the threshold, instead of
    # silently handing the LLM stage a full "fraud evidence" package for a
    # transaction that -- under this specific neighbor sample -- isn't even
    # currently classified as fraud.
    below_threshold_note = None
    if ctx["p_original"] < LOCKED_THRESHOLD:
        below_threshold_note = (
            f"This transaction's freshly-computed probability ({ctx['p_original']:.4f}) is "
            f"below the decision threshold ({LOCKED_THRESHOLD:.4f}) under this explanation's "
            f"own neighbor sample, even though it was flagged predicted-positive by a separate "
            f"scoring pass. Treat this explanation with reduced confidence -- the original "
            f"flagging and this explanation may have sampled different neighbors for a "
            f"high-degree account."
        )

    result = {
        "target_transaction": {
            "transaction_id": transaction_id,
            "features": decode_transaction_features(data.edge_attr[edge_idx]),
            "feature_attribution": feature_attribution,
        },
        "contributing_transactions": contributing_transactions,
        "model_prediction": {
            "probability": round(ctx["p_original"], 4),
            "predicted_label": int(ctx["p_original"] >= LOCKED_THRESHOLD),
        },
        "fidelity_check": {"fidelity_plus": round(fid_plus, 4), "fidelity_minus": round(fid_minus, 4)},
    }
    if below_threshold_note is not None:
        result["confidence_warning"] = below_threshold_note
    return result

# %% [markdown]
# ## 15. Demo on `test_mask` predicted-positive transactions
#
# Runs `explain_transaction()` end to end on a handful of test-set transactions the
# model flagged, prints the output, and -- purely as an offline diagnostic, never as
# a training signal or evidence source -- checks whether the ranked contributing
# transactions land in accounts `pattern_diagnostics.csv` already tags as part of a
# known laundering pattern (fan-in/fan-out/cycle). This is a sanity check on
# *plausibility*, not a correctness proof: PGExplainer is explaining the *model's*
# prediction, and the model itself is an imperfect classifier (test F1 ~= 0.574).

# %%
DEMO_N = 12
DEMO_POOL_SIZE = 40  # oversample -- some candidates get skipped if their freshly
                      # computed probability (Section 14's own deterministic
                      # NeighborLoader sample) no longer clears LOCKED_THRESHOLD

g = torch.Generator().manual_seed(RANDOM_SEED + 7)
demo_pool_idx = predicted_positive["test"][
    torch.randperm(predicted_positive["test"].size(0), generator=g)[:DEMO_POOL_SIZE]
]
demo_pool_txn_ids = [edge_idx_to_txn_id[i.item()] for i in demo_pool_idx]

demo_results = {}
n_skipped = 0
for txn_id in demo_pool_txn_ids:
    if len(demo_results) >= DEMO_N:
        break
    result = explain_transaction(txn_id)
    if "confidence_warning" in result:
        n_skipped += 1
        continue  # this candidate no longer clears the threshold under its own
                  # deterministic sample -- skip it for the demo showcase rather
                  # than display a confusing "predicted positive" example with a
                  # near-zero probability
    demo_results[txn_id] = result
    print(f"\n=== {txn_id}  (p_fraud={result['model_prediction']['probability']:.3f}) ===")
    print("Target features:", {k: round(v, 3) if isinstance(v, float) else v
                                for k, v in result["target_transaction"]["features"].items()})
    top_feat = sorted(result["target_transaction"]["feature_attribution"].items(),
                       key=lambda kv: abs(kv[1]), reverse=True)[:3]
    print("Top feature attributions:", top_feat)
    print(f"Fidelity+={result['fidelity_check']['fidelity_plus']:.3f}  "
          f"Fidelity-={result['fidelity_check']['fidelity_minus']:.3f}")
    for c in result["contributing_transactions"]:
        print(f"  -> {c['transaction_id']}  importance={c['importance']:.3f}  "
              f"amount_paid~{c['features'].get('log_amount_paid', 'n/a'):.2f} (log-space)")

if n_skipped > 0:
    print(f"\n({n_skipped} candidate(s) skipped -- probability dropped below threshold "
          f"under their own deterministic neighbor sample; see Section 14's confidence_warning.)")
if len(demo_results) < DEMO_N:
    print(f"\nWARNING: only found {len(demo_results)}/{DEMO_N} valid demo transactions in a "
          f"pool of {DEMO_POOL_SIZE} -- raise DEMO_POOL_SIZE if this happens often.")

# %%
# --- Optional: pattern-diagnostic plausibility check (offline, diagnostic only) ---
# graph_data.pt already carries data.node_pattern_label / data.node_pattern_type as
# ride-along metadata (attached in aml_graph_construction.ipynb Section 7b, sourced
# from pattern_diagnostics.csv's "account"/"pattern_label"/"pattern_type" columns --
# note lowercase "account", NOT "Account"). Reusing it directly here instead of
# re-loading the CSV and re-deriving the Bank_Account -> account match ourselves
# avoids duplicating that collision-prone lookup a second time.
if hasattr(data, "node_pattern_label") and hasattr(data, "node_pattern_type"):
    node_pattern_label = data.node_pattern_label
    node_pattern_type = data.node_pattern_type

    for txn_id, result in demo_results.items():
        edge_idx = txn_id_to_edge_idx[txn_id]
        src_idx = data.edge_index[0, edge_idx].item()
        dst_idx = data.edge_index[1, edge_idx].item()
        src_flagged = bool(node_pattern_label[src_idx] > 0)
        dst_flagged = bool(node_pattern_label[dst_idx] > 0)
        target_types = [t for t, f in [(node_pattern_type[src_idx], src_flagged),
                                        (node_pattern_type[dst_idx], dst_flagged)] if f]

        n_neighbor_flagged = 0
        for c in result["contributing_transactions"]:
            c_edge_idx = txn_id_to_edge_idx.get(c["transaction_id"])
            if c_edge_idx is None:
                continue
            c_src = data.edge_index[0, c_edge_idx].item()
            c_dst = data.edge_index[1, c_edge_idx].item()
            if node_pattern_label[c_src] > 0 or node_pattern_label[c_dst] > 0:
                n_neighbor_flagged += 1

        print(f"{txn_id}: target endpoints in known pattern = {src_flagged or dst_flagged} "
              f"{target_types}  |  {n_neighbor_flagged}/{len(result['contributing_transactions'])} "
              f"contributing transactions touch a known-pattern account")
else:
    print("data.node_pattern_label / data.node_pattern_type not found on graph_data.pt -- skipping "
          "the optional plausibility check (not required for the pipeline, only a diagnostic; "
          "means aml_graph_construction.ipynb's Section 7b was likely run with "
          "PARSE_PATTERNS_FILE=False upstream).")

# %% [markdown]
# ## 16. Next step
#
# `explain_transaction(transaction_id)` now returns the exact evidence-locked JSON
# shape the LLM notebook needs: a target transaction with its own decoded features
# + feature-level attribution, and a ranked, deduplicated list of contributing
# transactions with their own decoded features -- nothing else. The next notebook
# builds the prompt template that constrains the LLM to cite *only* fields present
# in this structure (no free-form narrative about the graph, no invented account
# details), plus the faithfulness check on the LLM's *output* (does its explanation
# actually mention the fields we gave it, and not fields we didn't).
