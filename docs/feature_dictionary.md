# Feature Dictionary

This document lists every feature used to train the GNN model, what it means, and why it was included. Features are split into two groups: **edge features** (describe a transaction) and **node features** (describe an account).

---

## Edge Features (21 total) — describe each transaction

| # | Feature | Type | What it means | Why it was selected |
|---|---------|------|----------------|----------------------|
| 1 | `log_amount_paid` | Continuous | Log-scaled amount sent | Amounts span several orders of magnitude; log scaling prevents the model from being biased toward large transactions |
| 2 | `log_amount_received` | Continuous | Log-scaled amount received | Same reason as above |
| 3 | `log_amount_ratio` | Continuous | Ratio of received/paid amount | A ratio close to 1.0 is a known structuring signal (moving the exact same amount to avoid detection) |
| 4 | `log_hours_since_last_tx` | Continuous | Time since this account's last transaction | Captures transaction velocity — a causal, account-history-only feature |
| 5 | `log_tx_count_cumulative` | Continuous | Running count of this account's transactions so far | Captures account activity level over time |
| 6 | `same_currency` | Binary | 1 if sender/receiver use the same currency | Currency conversion is a common laundering technique to complicate tracing |
| 7 | `is_cross_bank` | Binary | 1 if the transaction crosses banks | Cross-bank transfers are harder to trace than within-bank ones |
| 8 | `is_self_loop` | Binary | 1 if sender = receiver | Self-transactions can indicate structuring; kept and learned rather than removed |
| 9 | `hour_sin` | Continuous | Cyclical encoding of hour of day | Avoids the discontinuity of raw hour values (23 → 0) |
| 10 | `hour_cos` | Continuous | Cyclical encoding of hour of day | Same as above |
| 11 | `dow_sin` | Continuous | Cyclical encoding of day of week | Some laundering patterns favor specific days |
| 12 | `dow_cos` | Continuous | Cyclical encoding of day of week | Same as above |
| 13 | `is_weekend` | Binary | 1 if the transaction occurred on a weekend | Simple flag capturing weekday/weekend behavior differences |
| 14 | `rapid_tx_1h` | Binary | 1 if another transaction occurred within 1 hour | Structuring signal — rapid repeated transactions |
| 15 | `rapid_tx_6h` | Binary | 1 if another transaction occurred within 6 hours | Same, wider window |
| 16 | `rapid_tx_24h` | Binary | 1 if another transaction occurred within 24 hours | Same, widest window |
| 17 | `ratio_near_one` | Binary | 1 if amount ratio is close to 1.0 | Explicit flag version of `log_amount_ratio`, easier for the model (and later PGExplainer) to key on |
| 18 | `is_night` | Binary | 1 if the transaction happened at night | Unusual transaction timing can be a risk indicator |
| 19 | `Payment Format_enc` | Categorical code | Encoded payment method (ACH, wire, etc.) | Feeds into an `nn.Embedding` layer rather than being treated as a number |
| 20 | `Payment Currency_enc` | Categorical code | Encoded currency sent | Same as above |
| 21 | `Receiving Currency_enc` | Categorical code | Encoded currency received | Same as above |

**Note:** Columns 19–21 mark where categorical features start (`edge_cat_start = 18`). They are routed to `nn.Embedding` in the model, not treated as continuous values.

---

## Node Features (11 total) — describe each account

| # | Feature | Type | What it means | Why it was selected |
|---|---------|------|----------------|----------------------|
| 1 | `log_out_degree` | Continuous | Number of outgoing transactions (log-scaled) | Basic account activity signal |
| 2 | `log_in_degree` | Continuous | Number of incoming transactions (log-scaled) | Same |
| 3 | `log_total_degree` | Continuous | Total transactions (log-scaled) | Same |
| 4 | `log_out_amount_sum` | Continuous | Total amount sent (log-scaled) | Captures account-level financial volume |
| 5 | `log_in_amount_sum` | Continuous | Total amount received (log-scaled) | Same |
| 6 | `log_out_amount_mean` | Continuous | Average amount sent | Typical transaction size for the account |
| 7 | `log_in_amount_mean` | Continuous | Average amount received | Same |
| 8 | `log_out_unique_counterparties` | Continuous | Number of distinct accounts this one sends to | Accounts touching many different counterparties can indicate fan-out/gather patterns |
| 9 | `log_in_unique_counterparties` | Continuous | Number of distinct accounts sending to this one | Same, fan-in direction |
| 10 | `signed_log_net_flow` | Continuous | Net money flow (in − out), signed and log-scaled | Captures whether an account accumulates or drains money |
| 11 | `is_new_node` | Binary | 1 if the account had no activity in the training window | Flags accounts unseen during training — the realistic "new account" scenario the model must handle in production |

---

## Not used as model features (tracked separately)

| Item | Why it's excluded from training |
|---|---|
| `transaction_id` | A generated lookup key, not a signal — excluded to prevent the model from learning on a meaningless sequential ID |
| `pattern_label` / `pattern_type` (from `HI-Medium_Patterns.txt`) | Ground-truth synthetic labels not available at real inference time; used only as post-hoc diagnostics to sanity-check the explainer's output, never as training signal |

---

## Design principles behind every feature

1. **Leakage-safe:** every scaler/encoder is fit on the training split only, or the feature is purely causal (depends only on past account history)
2. **Interpretable:** binary flags and simple ratios were preferred over opaque transforms, since PGExplainer and the LLM stage need to reference clear, explainable signals
3. **No target leakage:** any feature that would encode information only available after the fact (e.g. dataset-wide laundering rate per category) was excluded
