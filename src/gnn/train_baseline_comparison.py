#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script version of train_baseline_comparison.ipynb
# Code cells are reproduced verbatim and in the original order; markdown cells are kept as comments.
# Cell markers (# %%) let VS Code / Spyder / Jupytext run the file cell by cell.

# %% [markdown]
# # Baseline Model Comparison — GCN vs GraphSAGE vs PNA vs GINEConv
#
# **Pipeline stage:** `preprocessing.ipynb` → `aml_graph_construction.ipynb` → **this notebook**
# → `pgexplainer.ipynb` (attribution) → LLM evidence-locked prompting
#
# **Goal:** train four PGExplainer-compatible GNN edge-classifier baselines on the same
# data, same protocol, and same compute budget; pick the strongest one by minority-class
# F1; then run a focused Optuna search to fine-tune only the winner.
#
# **Decisions this notebook implements:**
#
# 1. **Models:** GCN, GraphSAGE, PNA, GINEConv only — all local message-passing,
#    all eligible for production. FraudGT dropped entirely (global attention,
#    not PGExplainer-compatible, not a PyG built-in).
# 2. **Evaluation distribution:** val/test scored on the real class imbalance
#    (~0.11% positive) — no subsampling at eval time. `NEG_RATIO=10` subsampling
#    is training-only.
# 3. **Primary metric:** minority-class F1 @ threshold 0.5, comparable to the IBM AML
#    benchmark paper (Altman et al. 2023) on this same dataset. PR-AUC is secondary.
# 4. **Test set:** touched exactly once, after the winner and its tuned
#    hyperparameters are both locked in.
# 5. **Seeds:** 1 for screening, 5 for the winner's final report.
# 6. **Hyperparameter search:** Optuna, winner only.
# 7. **Message-passing scope per phase:** train phase sees `train_mask` edges only,
#    val phase sees `train_mask | val_mask`, test phase sees everything.
# 8. **Self-loops:** `add_remaining_self_loops`, uniform across all four models.
# 9. **Directionality:** every message-passing edge gets a reverse copy (excluding
#    genuine self-loops) tagged `is_reverse`.
#
# ---
#
# **Revision note (this version fixes a server-crash issue from the first draft):**
# the original implementation built three full, reverse-doubled, self-looped copies of
# the edge structure (train/val/test) *up front* and held all three in memory
# simultaneously — for a 31.9M-edge graph that's 14+ GB of `edge_attr` alone, on top of
# everything else, which is enough to exceed a typical JupyterHub container's RAM limit
# regardless of GPU size. **Decisions 7/8/9 above are unchanged** — what changed is
# *when* the reverse-edge and self-loop augmentation happens: instead of transforming
# the whole graph once, it's now applied to each small subgraph *after* neighbor
# sampling, on only the few thousand edges that specific mini-batch actually touches.
# Same message-passing semantics, a small fraction of the memory footprint.
#
# **Also tuned down for the reported hardware** (`NVIDIA A16, 15.7 GB`): batch sizes,
# neighbor fanout, hidden dimension, and PNA's tower count are all set more
# conservatively than the first draft. The A16 is a VDI/inference-oriented card with
# comparatively low FP32 throughput next to a training-class GPU (A100/H100) — expect
# training to be noticeably slower even once it's no longer crashing, not just memory-
# constrained. Watch the per-epoch timing the screening round prints and scale
# `SCREENING_EPOCHS` / `N_OPTUNA_TRIALS` / `FINAL_EPOCHS` down further if needed before
# committing to the full run.

# %% [markdown]
# ## 1. Imports & config

# %%
# --- Imports ---
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, SAGEConv, PNAConv, GINEConv
from torch_geometric.utils import add_remaining_self_loops, degree

from sklearn.metrics import f1_score, precision_recall_curve, auc as sk_auc

import optuna

DATA_DIR = Path("/home/jovyan/AML_Project/Data/Bassam Data")
assert DATA_DIR.exists(), f"{DATA_DIR} not found -- run preprocessing.ipynb and aml_graph_construction.ipynb first."

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")


def free_gpu(*objs):
    '''Explicitly drop references and release CUDA cache -- called between models/
    trials/seeds, where a whole model+optimizer's worth of GPU tensors becomes
    unreachable at once and is worth reclaiming immediately rather than waiting for
    the next allocation to trigger it.'''
    for o in objs:
        del o
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# %%
# --- Config ---
# Second speed pass: GPU is barely touched (1.1GB / 15.7GB at batch=4096, per the
# diagnostic runs), so pushing batch size further should keep cutting the number of
# NeighborLoader reconstructions per epoch (still the dominant cost) -- expect a
# smaller marginal win than the first jump (256->4096 was ~9x), since some of the
# per-epoch time is real per-node compute that scales with batch size too, not pure
# fixed overhead.
#
# NOT changed: num_workers on NeighborLoader. Each mini-batch constructs a NEW
# NeighborLoader with batch_size == len(seed_nodes) (i.e. everything in ONE internal
# batch, consumed via next(iter(loader))) -- there's no second batch for a worker
# pool to prefetch while the GPU is busy with the first, so num_workers would only
# add process-spawn overhead here, not parallelism. It would help under a
# fundamentally different loop structure (one persistent loader per epoch, e.g.
# LinkNeighborLoader, iterated many times) -- a real option if batch-size increases
# stop paying off, but a bigger restructure than is warranted right now.
#
# NOT used: the second A16 (idle, 3 MiB used). A single Python process only talks to
# one GPU at a time -- using both means running independent processes. The practical
# way to do that here without restructuring this notebook: open a second Jupyter
# kernel, set DEVICE = torch.device("cuda:1") in it, and split the 4 screening models
# 2-and-2 across the two kernels (they're fully independent runs). Not wired into
# this notebook automatically, since it changes how you run cells, not what's in them.

NEG_RATIO = 10

SCREENING_SEED = 42
SCREENING_EPOCHS = 5

FINAL_N_SEEDS = 5
FINAL_EPOCHS = 20

N_OPTUNA_TRIALS = 30
OPTUNA_EPOCHS_PER_TRIAL = 8

NUM_NEIGHBORS = [10, 5]
TRAIN_BATCH_SIZE = 8192        # كان 4096
INFER_BATCH_SIZE = 16384       # كان 8192

HIDDEN_DIM = 64
N_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-5
PNA_TOWERS = 2

MODEL_NAMES = ["gcn", "sage", "pna", "gine"]

torch.manual_seed(SCREENING_SEED)
np.random.seed(SCREENING_SEED)

print(f"Updated config: TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE}, "
      f"INFER_BATCH_SIZE={INFER_BATCH_SIZE}, HIDDEN_DIM={HIDDEN_DIM}")

# %% [markdown]
# ## 2. Load `graph_data.pt`

# %%
data = torch.load(DATA_DIR / "graph_data.pt", weights_only=False)
print(data)

edge_cat_start = data.edge_cat_start
print(f"\nedge_cat_start = {edge_cat_start}")

with open(DATA_DIR / "metadata.json") as f:
    metadata = json.load(f)

CAT_COLS = ["Payment Format", "Payment Currency", "Receiving Currency"]
vocab_sizes = [metadata["vocab_sizes"][c] for c in CAT_COLS]
print("Categorical vocab sizes (incl. UNK=0):", dict(zip(CAT_COLS, vocab_sizes)))

n_nodes = data.x.size(0)
n_edges_total = data.edge_index.size(1)
in_dim = data.x.size(1)
print(f"\nn_nodes={n_nodes:,}  n_edges_total={n_edges_total:,}  node_feature_dim={in_dim}")

for split in ["train", "val", "test"]:
    mask = getattr(data, f"{split}_mask")
    pos_rate = data.y[mask].float().mean().item()
    print(f"  {split}: {mask.sum().item():,} edges, positive rate = {pos_rate:.4%}")

# %% [markdown]
# ## 3. Phase-scoped base graphs + per-subgraph augmentation (decisions 7, 8, 9)
#
# **What changed from the first draft, and why:** decisions 7/8/9 themselves are
# unchanged -- train-phase message passing still only sees `train_mask` edges, val
# sees `train_mask | val_mask`, test sees everything; every message-passing edge still
# gets a reverse copy (except genuine self-loops); every node still gets exactly one
# self-loop. What moved is **when** that transform happens.
#
# **Before:** the full phase-scoped edge set was reverse-doubled and self-looped once,
# up front, and kept in memory as `train_mp_ei/ea`, `val_mp_ei/ea`, `test_mp_ei/ea`
# simultaneously -- for a 31.9M-edge graph that's 14+ GB of `edge_attr` alone, which is
# almost certainly what was crashing the JupyterHub container regardless of GPU size.
#
# **Now:** each phase keeps only its lightweight *base* (masked, not doubled) edge set —
# `train_base_ei/ea` is ~22M edges once, not ~44M edges three times over. Neighbor
# sampling runs on the base graph, and the reverse+self-loop augmentation is applied
# **after** sampling, to the small local subgraph `NeighborLoader` returns (typically a
# few thousand edges) — right before that subgraph is fed to the model. Message-passing
# semantics are the same; the only second-order difference is that `NeighborLoader`'s
# traversal now samples fanout from the base (undoubled) graph rather than the
# pre-augmented one, which shifts its hop-selection probabilities slightly but doesn't
# change what information ends up reachable within the same number of hops.

# %%
def augment_subgraph(edge_index, edge_attr, num_nodes):
    # Apply decisions 8/9 (self-loops, directionality) to an already-sampled LOCAL
    # subgraph -- keeps memory bounded regardless of total graph size.
    #
    # Returns (edge_index, edge_attr, is_reverse) as THREE separate values.
    # edge_attr always keeps the ORIGINAL column layout (edge_cat_start continuous +
    # 3 categorical, exactly matching data.edge_attr / graph_data.pt) -- is_reverse
    # is never appended as an extra column inside it. (Fix: an earlier version did
    # append is_reverse into edge_attr, which worked for message-passing edges but
    # broke EdgeClassifier.score() for the target edge, whose edge_attr comes
    # straight from data.edge_attr and never gets that column appended -- see
    # Section 4/5 markdown for the full explanation.)
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
        # Synthetic self-loops are neither forward nor reverse -- flag 0. Their
        # edge_attr row is all-zero (fill_value=0.0 above), landing on
        # category-index 0 (UNK) for every categorical column, same fallback
        # bucket preprocessing.ipynb uses for unseen categories.
        is_reverse = torch.cat([is_reverse, torch.zeros(n_added, 1, dtype=torch.float)], dim=0)

    return ei, ea, is_reverse


# Lightweight phase bases -- masked only, NOT reverse-doubled or self-looped.
# Roughly 1.9 / 2.3 / 2.7 GB of edge_attr respectively.
train_base_ei = data.edge_index[:, data.train_mask]
train_base_ea = data.edge_attr[data.train_mask]

val_base_mask = data.train_mask | data.val_mask
val_base_ei = data.edge_index[:, val_base_mask]
val_base_ea = data.edge_attr[val_base_mask]

test_base_ei = data.edge_index
test_base_ea = data.edge_attr

print(f"train base edges: {train_base_ei.size(1):,}")
print(f"val   base edges: {val_base_ei.size(1):,}")
print(f"test  base edges: {test_base_ei.size(1):,}  (= all edges, only used in Section 10)")

# %%
def compute_pna_deg_histogram(base_edge_index, num_nodes):
    '''In-degree histogram matching what PNA will actually see post-augmentation
    (reverse edges + self-loops), computed from TRAIN base only (train-only fitting,
    same principle as every other fitted artifact upstream). Uses only the int64
    edge_index -- not edge_attr -- and discards the doubled index tensor immediately
    after, so this stays a one-off ~700 MB transient, not something held for the rest
    of the notebook.'''
    non_self_loop = base_edge_index[0] != base_edge_index[1]
    rev_ei = base_edge_index[:, non_self_loop].flip(0)
    full_ei = torch.cat([base_edge_index, rev_ei], dim=1)
    full_ei, _ = add_remaining_self_loops(full_ei, num_nodes=num_nodes)
    d = degree(full_ei[1], num_nodes=num_nodes, dtype=torch.long)
    del full_ei, rev_ei, non_self_loop
    gc.collect()
    return torch.bincount(d)

pna_deg = compute_pna_deg_histogram(train_base_ei, n_nodes)
print(f"PNA degree histogram: {pna_deg.numel()} bins, "
      f"max observed post-augmentation train in-degree = {pna_deg.numel() - 1}")

# %% [markdown]
# ## 4. Edge encoder, node-encoder backbones, and the classification head
#
# GCN/GraphSAGE don't consume `edge_attr` inside message passing (only at the
# classification head); PNA/GINEConv do, via `edge_dim`. `PNA_TOWERS` reduced to 2 to
# fit the reported GPU.
#
# **Fix applied here (found when this notebook was first run):** `edge_attr` now
# always keeps its *original* column layout (`edge_cat_start` continuous + 3
# categorical -- exactly `data.edge_attr`'s shape, never with anything appended).
# `is_reverse` is passed to `EdgeEncoder.forward()` as a *separate* argument,
# defaulting to zero when omitted. The earlier version appended `is_reverse` as an
# extra column inside `edge_attr` itself, which worked for message-passing edges
# (always routed through `augment_subgraph`) but broke for the edge being classified
# at the head -- that one comes straight from `data.edge_attr` and never gets an
# `is_reverse` column appended, so the encoder's column-slicing silently ate one of
# the 3 real categorical columns instead of a synthetic one, and crashed once a
# mini-batch actually reached the scoring step (`IndexError: index 2 is out of bounds
# for dimension 1 with size 2`).

# %%
class EdgeEncoder(nn.Module):
    # edge_attr is ALWAYS the raw layout: [continuous/binary (edge_cat_start cols)]
    # + [categorical codes (3 cols)] -- exactly as in data.edge_attr / graph_data.pt,
    # whether this is a message-passing edge or a target/classification edge.
    # is_reverse is passed separately (never baked into edge_attr's columns) and
    # defaults to all-zero when omitted -- correct for target edges, which are
    # always the original transaction direction, never a synthetic reverse copy.
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

# %%
class NodeEncoderGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim)])
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_embedding=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class NodeEncoderSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)])
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_embedding=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


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


class NodeEncoderGINE(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        mlp0 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList([GINEConv(mlp0, edge_dim=hidden_dim)])
        for _ in range(n_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_embedding):
        x = self.input_proj(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_embedding)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


NODE_ENCODER_CLASSES = {"gcn": NodeEncoderGCN, "sage": NodeEncoderSAGE,
                         "pna": NodeEncoderPNA, "gine": NodeEncoderGINE}
MODELS_USING_EDGE_ATTR_IN_MP = {"pna", "gine"}

# %% [markdown]
# ## 5. The `EdgeClassifier` wrapper
#
# `encode()` now takes and forwards an optional `is_reverse` tensor to the edge
# encoder (used for message-passing edges, via `augment_subgraph`'s third return
# value). `score()` never passes one -- the target edge is always the original
# transaction, so `EdgeEncoder` defaults it to zero internally.

# %%
class EdgeClassifier(nn.Module):
    def __init__(self, model_name, in_dim, edge_cat_start, vocab_sizes,
                 hidden_dim=32, n_layers=2, dropout=0.2, deg=None, pna_towers=2):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.uses_edge_attr_in_mp = model_name in MODELS_USING_EDGE_ATTR_IN_MP

        self.edge_encoder = EdgeEncoder(edge_cat_start, vocab_sizes, emb_dim=8, out_dim=hidden_dim)

        encoder_cls = NODE_ENCODER_CLASSES[model_name]
        if model_name == "pna":
            assert deg is not None, "PNA requires a train-only degree histogram"
            self.node_encoder = encoder_cls(in_dim, hidden_dim, deg, n_layers, dropout,
                                             edge_dim=hidden_dim, towers=pna_towers)
        else:
            self.node_encoder = encoder_cls(in_dim, hidden_dim, n_layers, dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index, edge_attr, is_reverse=None):
        edge_emb = self.edge_encoder(edge_attr, is_reverse)
        if self.uses_edge_attr_in_mp:
            return self.node_encoder(x, edge_index, edge_emb)
        return self.node_encoder(x, edge_index)

    def score(self, h_src, h_dst, target_edge_attr):
        # is_reverse omitted -> defaults to zero, correct: the classification
        # target is always the original transaction direction.
        target_edge_emb = self.edge_encoder(target_edge_attr)
        z = torch.cat([h_src, h_dst, target_edge_emb], dim=1)
        return self.head(z).squeeze(-1)

# %% [markdown]
# ## 6. Training and evaluation utilities
#
# Sample a subgraph from the phase's *base* (undoubled) graph -> call
# `augment_subgraph` on that small subgraph (now returns `edge_index, edge_attr,
# is_reverse` as three separate values, per the Section 4/5 fix) -> forward, passing
# `is_reverse` through to `model.encode()`. The target edge's own `edge_attr`, used in
# `model.score()`, is passed with no `is_reverse` argument at all (defaults to zero).
#
# `_local_lookup` replaces a per-batch `torch.full((n_nodes,), -1)` allocation --
# scales with batch size, not total node count.

# %%
def _local_lookup(seed_ids, query_ids):
    # Map global node ids in query_ids to their row position in seed_ids (the order
    # NeighborLoader returns embeddings in). Scales with len(seed_ids), not the
    # total node count.
    sorted_ids, sort_perm = torch.sort(seed_ids)
    pos_in_sorted = torch.searchsorted(sorted_ids, query_ids)
    return sort_perm[pos_in_sorted]


def sample_training_targets(data, neg_ratio, seed):
    # Per-epoch negative subsampling (NEG_RATIO=10) -- global indices into
    # data.edge_index/edge_attr/y, train_mask only.
    g = torch.Generator().manual_seed(seed)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    y_train = data.y[train_idx]

    pos_idx = train_idx[y_train == 1]
    neg_idx_all = train_idx[y_train == 0]

    n_neg = min(len(neg_idx_all), len(pos_idx) * neg_ratio)
    perm = torch.randperm(len(neg_idx_all), generator=g)[:n_neg]
    neg_idx = neg_idx_all[perm]

    idx = torch.cat([pos_idx, neg_idx])
    shuffle = torch.randperm(len(idx), generator=g)
    return idx[shuffle]

# %%
def train_one_epoch(model, optimizer, data, base_ei, base_ea, num_neighbors,
                     batch_size, neg_ratio, seed, device):
    model.train()
    epoch_idx = sample_training_targets(data, neg_ratio, seed)

    target_src_all = data.edge_index[0, epoch_idx]
    target_dst_all = data.edge_index[1, epoch_idx]
    target_ea_all = data.edge_attr[epoch_idx]
    target_y_all = data.y[epoch_idx].float()

    base_graph = Data(x=data.x, edge_index=base_ei, edge_attr=base_ea, num_nodes=data.x.size(0))

    perm = torch.randperm(epoch_idx.size(0))
    total_loss, n_seen = 0.0, 0

    for start in range(0, epoch_idx.size(0), batch_size):
        batch_pos = perm[start:start + batch_size]
        b_src = target_src_all[batch_pos]
        b_dst = target_dst_all[batch_pos]
        b_ea = target_ea_all[batch_pos].to(device)   # raw layout -- no is_reverse column
        b_y = target_y_all[batch_pos].to(device)

        seed_nodes = torch.unique(torch.cat([b_src, b_dst]))

        loader = NeighborLoader(base_graph, num_neighbors=num_neighbors, input_nodes=seed_nodes,
                                 batch_size=seed_nodes.size(0), shuffle=False)
        sub = next(iter(loader))

        aug_ei, aug_ea, aug_is_rev = augment_subgraph(sub.edge_index, sub.edge_attr, sub.num_nodes)
        x_dev = sub.x.to(device)
        aug_ei, aug_ea, aug_is_rev = aug_ei.to(device), aug_ea.to(device), aug_is_rev.to(device)

        h = model.encode(x_dev, aug_ei, aug_ea, aug_is_rev)
        h_seed = h[:sub.batch_size]

        seed_ids = sub.n_id[:sub.batch_size]
        h_src = h_seed[_local_lookup(seed_ids, b_src)]
        h_dst = h_seed[_local_lookup(seed_ids, b_dst)]

        # is_reverse omitted here -> defaults to 0, correct: target is always the
        # original transaction direction.
        logits = model.score(h_src, h_dst, b_ea)
        loss = F.binary_cross_entropy_with_logits(logits, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * b_y.size(0)
        n_seen += b_y.size(0)

        del sub, aug_ei, aug_ea, aug_is_rev, h, h_seed, h_src, h_dst, logits, loss

    del base_graph
    return total_loss / max(n_seen, 1)

# %%
def compute_node_embeddings_for(model, data, base_ei, base_ea, target_idx, num_neighbors,
                                 batch_size, device):
    # Layer-wise inference restricted to nodes touching target_idx's edges. Samples
    # + augments per mini-batch, no gradients, caches into one persistent
    # (n_nodes x hidden_dim) buffer.
    target_nodes = torch.unique(torch.cat([
        data.edge_index[0, target_idx], data.edge_index[1, target_idx]
    ]))
    base_graph = Data(x=data.x, edge_index=base_ei, edge_attr=base_ea, num_nodes=data.x.size(0))
    loader = NeighborLoader(base_graph, num_neighbors=num_neighbors, input_nodes=target_nodes,
                             batch_size=batch_size, shuffle=False)

    all_h = torch.zeros(data.x.size(0), model.hidden_dim)
    model.eval()
    with torch.no_grad():
        for sub in loader:
            aug_ei, aug_ea, aug_is_rev = augment_subgraph(sub.edge_index, sub.edge_attr, sub.num_nodes)
            x_dev = sub.x.to(device)
            aug_ei, aug_ea, aug_is_rev = aug_ei.to(device), aug_ea.to(device), aug_is_rev.to(device)
            h = model.encode(x_dev, aug_ei, aug_ea, aug_is_rev)
            all_h[sub.n_id[:sub.batch_size]] = h[:sub.batch_size].cpu()
            del sub, aug_ei, aug_ea, aug_is_rev, h
    del base_graph
    return all_h


def evaluate(model, data, base_ei, base_ea, target_idx, num_neighbors, infer_batch_size,
             device, threshold=None):
    # Scores target_idx (val_mask or test_mask, FULL -- no subsampling).
    #
    # threshold=None (default) -> auto-selects the threshold that maximizes F1 on
    # THIS SAME target set's precision-recall curve. Correct for val (that's what
    # val is for) and for screening/Optuna, which only ever look at val.
    #
    # threshold=<a float> -> uses it directly instead of auto-selecting. This is
    # what Section 10's final run passes for the TEST evaluation -- the threshold
    # is locked in from a val-set evaluation first, so test is scored but never
    # used to make any decision, keeping "test touched once" intact even though
    # threshold=0.5 was never actually a good operating point given training-time
    # subsampling (NEG_RATIO=10) shifting the model's probability calibration
    # relative to the real ~0.11% positive rate at eval time.
    h = compute_node_embeddings_for(model, data, base_ei, base_ea, target_idx,
                                     num_neighbors, infer_batch_size, device).to(device)

    src = data.edge_index[0, target_idx].to(device)
    dst = data.edge_index[1, target_idx].to(device)
    ea = data.edge_attr[target_idx].to(device)
    y = data.y[target_idx].float()

    model.eval()
    logits_chunks = []
    with torch.no_grad():
        for start in range(0, target_idx.size(0), infer_batch_size):
            end = start + infer_batch_size
            logits_chunks.append(model.score(h[src[start:end]], h[dst[start:end]], ea[start:end]).cpu())
    logits = torch.cat(logits_chunks)
    probs = torch.sigmoid(logits)

    y_np, probs_np = y.numpy().astype(int), probs.numpy()

    precision, recall, pr_thresholds = precision_recall_curve(y_np, probs_np)
    pr_auc = sk_auc(recall, precision)

    if threshold is None:
        f1_curve = np.divide(2 * precision * recall, precision + recall,
                              out=np.zeros_like(precision), where=(precision + recall) > 0)
        if len(pr_thresholds) > 0:
            best_idx = int(np.argmax(f1_curve[:-1]))
            threshold = float(pr_thresholds[best_idx])
            f1 = float(f1_curve[best_idx])
        else:
            threshold = 0.5
            f1 = 0.0
    else:
        preds_np = (probs_np >= threshold).astype(int)
        f1 = f1_score(y_np, preds_np, pos_label=1, zero_division=0)

    del h
    return {"f1": f1, "pr_auc": pr_auc, "probs": probs, "threshold": threshold}

# %% [markdown]
# ## 7. Screening round — GCN vs GraphSAGE vs PNA vs GINEConv
#
# Single seed, identical hyperparameters across all four. Evaluated on `val_mask`,
# full imbalance. `test_base_ei/ea` is not referenced anywhere in this section.
#
# Trained models are not kept after their metrics are extracted -- only
# `screening_results` (numbers) survives each loop iteration; the model itself is
# freed via `free_gpu()` before moving to the next architecture.

# %%
def build_model(model_name, in_dim, edge_cat_start, vocab_sizes, hidden_dim, n_layers,
                 dropout, device, deg=None, pna_towers=PNA_TOWERS):
    model = EdgeClassifier(model_name, in_dim, edge_cat_start, vocab_sizes,
                            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
                            deg=deg, pna_towers=pna_towers)
    return model.to(device)


def run_training(model_name, data, train_base_ei, train_base_ea, val_base_ei, val_base_ea,
                  n_epochs, seed, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                  dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY,
                  num_neighbors=NUM_NEIGHBORS, train_batch_size=TRAIN_BATCH_SIZE,
                  infer_batch_size=INFER_BATCH_SIZE, pna_towers=PNA_TOWERS, verbose=True):
    # One full train+val run for one model_name/seed/hyperparameter combo. Shared by
    # the screening round, the Optuna objective, and the final 5-seed run.
    torch.manual_seed(seed)
    np.random.seed(seed)

    deg = pna_deg if model_name == "pna" else None
    model = build_model(model_name, in_dim, edge_cat_start, vocab_sizes, hidden_dim,
                         n_layers, dropout, DEVICE, deg=deg, pna_towers=pna_towers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, data, train_base_ei, train_base_ea,
                                      num_neighbors, train_batch_size, NEG_RATIO,
                                      seed=seed + epoch, device=DEVICE)
        val_metrics = evaluate(model, data, val_base_ei, val_base_ea,
                                data.val_mask.nonzero(as_tuple=True)[0],
                                num_neighbors, infer_batch_size, DEVICE)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_f1": val_metrics["f1"], "val_pr_auc": val_metrics["pr_auc"]})
        if verbose:
            print(f"  [{model_name}] epoch {epoch:2d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  val_F1={val_metrics['f1']:.4f}  "
                  f"val_PR-AUC={val_metrics['pr_auc']:.4f}  ({time.time()-t0:.1f}s)")

    return model, history

# %%
screening_results = {}

for model_name in MODEL_NAMES:
    print(f"\n=== Screening: {model_name.upper()} ===")
    model, history = run_training(
        model_name, data, train_base_ei, train_base_ea, val_base_ei, val_base_ea,
        n_epochs=SCREENING_EPOCHS, seed=SCREENING_SEED,
    )
    screening_results[model_name] = history[-1]
    screening_results[model_name]["history"] = history

    free_gpu(model)

print("\nScreening round complete.")

# %% [markdown]
# ## 8. Compare the 4 models, pick the winner
#
# Winner = highest **val minority-class F1**.
#
# **Calibration reference — Altman et al. 2023, minority-class F1 on HI-Medium, 5-seed
# mean ± std:**
#
# | Model | F1 (%) |
# |---|---|
# | MLP (no graph) | 7.95 ± 1.11 |
# | GAT | 15.66 ± 20.22 (unstable) |
# | GIN | 38.48 ± 2.54 |
# | GIN+EU (≈ GINEConv here) | 54.90 ± 1.62 |
# | PNA | 68.12 ± 2.63 |
#
# Not a pass/fail bar. But numbers far outside this range in either direction are a
# signal to check the implementation before trusting the ranking below.

# %%
screening_df = pd.DataFrame([
    {"model": m, "val_f1": r["val_f1"], "val_pr_auc": r["val_pr_auc"]}
    for m, r in screening_results.items()
]).sort_values("val_f1", ascending=False).reset_index(drop=True)

print(screening_df.to_string(index=False))

WINNER = screening_df.iloc[0]["model"]
print(f"\nWinner (highest val F1): {WINNER.upper()}")
print("Runner-up gap: "
      f"{screening_df.iloc[0]['val_f1'] - screening_df.iloc[1]['val_f1']:.4f} F1 points "
      f"over {screening_df.iloc[1]['model'].upper()}")

# %% [markdown]
# **A close margin is worth a manual look before committing.** If the gap between
# 1st and 2nd place is small relative to how much F1 moves epoch-to-epoch in the
# training curves, rerun the top 2 with 2-3 different seeds before proceeding to
# Section 9.

# %% [markdown]
# ## 9. Hyperparameter search — winner only
#
# Runs only on `WINNER`. Each trial is short (`OPTUNA_EPOCHS_PER_TRIAL`), evaluated on
# `val_mask` F1 -- `test_base_*` still untouched. Each trial's model is freed via
# `free_gpu()` immediately after scoring.

# %%
def optuna_objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 2, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    fanout_hop1 = trial.suggest_categorical("fanout_hop1", [5, 10, 15])
    fanout_hop2 = trial.suggest_categorical("fanout_hop2", [5, 10])
    num_neighbors = [fanout_hop1, fanout_hop2] + [fanout_hop2] * max(n_layers - 2, 0)

    model = None
    try:
        model, history = run_training(
            WINNER, data, train_base_ei, train_base_ea, val_base_ei, val_base_ea,
            n_epochs=OPTUNA_EPOCHS_PER_TRIAL, seed=SCREENING_SEED,
            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
            lr=lr, weight_decay=weight_decay, num_neighbors=num_neighbors,
            verbose=False,
        )
        best_val_f1 = max(h["val_f1"] for h in history)
        trial.set_user_attr("history", history)
        return best_val_f1
    finally:
        # Guaranteed cleanup even on OOM mid-training -- otherwise a failed
        # trial's leftover GPU allocations can cause the NEXT trial to fail too.
        if model is not None:
            free_gpu(model)
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()


STUDY_DB = str(DATA_DIR / f"{WINNER}_optuna_study.db")
study = optuna.create_study(
    direction="maximize",
    study_name=f"{WINNER}_tuning",
    storage=f"sqlite:///{STUDY_DB}",
    load_if_exists=True,
)
n_remaining = max(N_OPTUNA_TRIALS - len(study.trials), 0)
print(f"{len(study.trials)} trials already completed, {n_remaining} remaining.")
study.optimize(
    optuna_objective, n_trials=n_remaining, show_progress_bar=False,
    catch=(torch.cuda.OutOfMemoryError,),   # trial واحدة تفشل بـ OOM مش لازم توقف الدراسة كلها
)

print(f"\nBest trial: val F1 = {study.best_value:.4f}")
print("Best hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# %%
best_params = study.best_params
BEST_HP = {
    "hidden_dim": best_params["hidden_dim"],
    "n_layers": best_params["n_layers"],
    "dropout": best_params["dropout"],
    "lr": best_params["lr"],
    "weight_decay": best_params["weight_decay"],
    "num_neighbors": (
        [best_params["fanout_hop1"], best_params["fanout_hop2"]]
        + [best_params["fanout_hop2"]] * max(best_params["n_layers"] - 2, 0)
    ),
}
print("Final hyperparameters for the winning model:")
print(json.dumps(BEST_HP, indent=2))

with open(DATA_DIR / f"best_hyperparams_{WINNER}.json", "w") as f:
    json.dump({"model": WINNER, **BEST_HP}, f, indent=2)

# %% [markdown]
# ## 10. Final report — 5 seeds, tuned hyperparameters, test touched once
#
# The only section that references `test_base_ei/ea` / `test_mask`. Everything above
# used `val_mask` exclusively.
#
# Each seed's weights are moved to CPU and kept only until Section 11 decides which
# one to save -- GPU memory is freed after every seed via `free_gpu()`.

# %%
final_results = []
final_models = {}

for seed_offset in range(FINAL_N_SEEDS):
    seed = 1000 + seed_offset
    print(f"\n=== Final run: {WINNER.upper()}, seed {seed} ===")
    model, history = run_training(
        WINNER, data, train_base_ei, train_base_ea, val_base_ei, val_base_ea,
        n_epochs=FINAL_EPOCHS, seed=seed, **BEST_HP,
    )

    # Lock the decision threshold from VAL (auto-selected -- see evaluate()'s
    # threshold=None behavior), then score TEST with that fixed threshold. Test is
    # scored but never used to choose anything -- "touched once" stays true even
    # though we're not using the naive 0.5 default anymore.
    val_metrics = evaluate(model, data, val_base_ei, val_base_ea,
                            data.val_mask.nonzero(as_tuple=True)[0],
                            BEST_HP["num_neighbors"], INFER_BATCH_SIZE, DEVICE)
    locked_threshold = val_metrics["threshold"]

    test_metrics = evaluate(model, data, test_base_ei, test_base_ea,
                             data.test_mask.nonzero(as_tuple=True)[0],
                             BEST_HP["num_neighbors"], INFER_BATCH_SIZE, DEVICE,
                             threshold=locked_threshold)
    print(f"  seed {seed}: val-locked threshold={locked_threshold:.4f}  "
          f"TEST F1={test_metrics['f1']:.4f}  TEST PR-AUC={test_metrics['pr_auc']:.4f}")

    final_results.append({"seed": seed, "threshold": locked_threshold,
                           "test_f1": test_metrics["f1"], "test_pr_auc": test_metrics["pr_auc"]})

    final_models[seed] = {k: v.cpu() for k, v in model.state_dict().items()}
    free_gpu(model)

# %%
final_df = pd.DataFrame(final_results)
f1_mean, f1_std = final_df["test_f1"].mean(), final_df["test_f1"].std()
pr_auc_mean, pr_auc_std = final_df["test_pr_auc"].mean(), final_df["test_pr_auc"].std()

print(final_df.to_string(index=False))
print(f"\n{WINNER.upper()} final test performance ({FINAL_N_SEEDS} seeds):")
print(f"  Minority-class F1 : {f1_mean:.4f} ± {f1_std:.4f}")
print(f"  PR-AUC (secondary): {pr_auc_mean:.4f} ± {pr_auc_std:.4f}")

print("\nReference (Altman et al. 2023, HI-Medium, 5-seed F1):")
print("  MLP=7.95±1.11  GAT=15.66±20.22  GIN=38.48±2.54  GIN+EU=54.90±1.62  PNA=68.12±2.63")

# %% [markdown]
# ## 11. Save results

# %%
best_seed_idx = (final_df["test_f1"] - f1_mean).abs().idxmin()
best_seed = int(final_df.iloc[best_seed_idx]["seed"])
best_state_dict = final_models[best_seed]

torch.save({
    "model_state_dict": best_state_dict,
    "model_name": WINNER,
    "hyperparameters": BEST_HP,
    "seed": best_seed,
    "pna_deg_histogram": pna_deg if WINNER == "pna" else None,
}, DATA_DIR / f"best_model_{WINNER}.pt")

summary = {
    "screening_results": {m: {"val_f1": r["val_f1"], "val_pr_auc": r["val_pr_auc"]}
                           for m, r in screening_results.items()},
    "winner": WINNER,
    "best_hyperparameters": BEST_HP,
    "final_test_f1_mean": float(f1_mean),
    "final_test_f1_std": float(f1_std),
    "final_test_pr_auc_mean": float(pr_auc_mean),
    "final_test_pr_auc_std": float(pr_auc_std),
    "final_seeds": final_df["seed"].tolist(),
    "reference_benchmark_hi_medium_f1": {
        "MLP": "7.95 ± 1.11", "GAT": "15.66 ± 20.22", "GIN": "38.48 ± 2.54",
        "GIN+EU": "54.90 ± 1.62", "PNA": "68.12 ± 2.63",
        "source": "Altman et al. 2023, arXiv:2306.16424",
    },
}

with open(DATA_DIR / "baseline_comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved:")
print(f"  {DATA_DIR / f'best_model_{WINNER}.pt'}")
print(f"  {DATA_DIR / 'baseline_comparison_summary.json'}")
print(f"\nWinning architecture: {WINNER.upper()}")
print(f"Final test F1: {f1_mean:.4f} ± {f1_std:.4f}")

# %% [markdown]
# ## Next step
#
# Load `best_model_{WINNER}.pt` and `baseline_comparison_summary.json` in
# `pgexplainer.ipynb`. That notebook needs the same `augment_subgraph` function and
# the same phase-scoped base edge sets (`train_base_ei/ea`, etc.) from Section 3 here,
# including the separate `is_reverse` handling from Section 4/5 -- PGExplainer's
# attributions have to be computed over the identical message-passing structure the
# model was actually trained and evaluated on.
