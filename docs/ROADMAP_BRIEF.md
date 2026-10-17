# Project roadmap (brief)

This is the whole project on one page. It covers the LLM red-team pipeline
and the ML coursework analysis that `main_analysis.py` runs on campaign
transcripts. For every file, model and dataset in detail, see
[ROADMAP_DETAILED.md](ROADMAP_DETAILED.md).

```
──────────────────────── DATA SOURCES ────────────────────────

     JailbreakBench / JBB-Behaviors  (Hugging Face, MIT)
     data/import_jailbreakbench.py
     100 harmful + 100 benign goals, with SHA-256 manifest
                     │
                     ▼
     held-out campaign seeds (never used for PPO or judge training)

     data/build_demo_assets.py → safe placeholder judge rows
     data/seed_prompts.txt     → 8 high-level local safety probes

──────────────────────────── TRAIN JUDGES ────────────────────────────

        safe placeholder labeled examples (144 train / 36 val each)
                /                                    \
               ↓                                      ↓
     Reward DistilBERT  (seed 42)           Eval DistilBERT  (seed 84)
     fast PPO reward signal                  independent final scorer
               │                                      │
               ▼                                      │
──────────────────────── ATTACKER TRAINING ─────────────────────────

Qwen2.5-0.5B-Instruct
          ↓
   freeze base weights
          ↓
       add LoRA (r=16, α=32)  ← trainable policy parameters
          +
     add Value Head            ← trainable critic
          ↓
       seed prompt
          ↓
 generate evaluation prompt
          ↓
   Target LLM (local SmolLM2-135M, allowlist-gated)
          ↓
    target response
          ↓
   Reward DistilBERT → 3 logits → softmax
          ↓
P(refusal), P(partial), P(full)
          ↓
  reward shaping  R = 1.0·P(full) + 0.3·P(partial) − 0.1·P(refusal)
   + degeneracy checks (−1.0)
   − 0.4 · MiniLM max cosine similarity (diversity)
          ↓
  per-token KL vs reference Qwen (adapters disabled), coef 0.05
          ↓
          GAE (γ=1.0, λ=0.95)
          ↓
 normalized advantages
          ↓
 clipped PPO (ε=0.2) + value loss (0.5)
          ↓
 update LoRA + Value Head
          ↓
   TRAINED ATTACKER

──────────────────────────── CAMPAIGN ────────────────────────────

trained attacker  +  seeds (seed_prompts.txt OR JailbreakBench goals)
       ↓
multi-turn best-of-N agent (6 turns × 3 candidate techniques)
       ↓
authorized target (allowlist + circuit breaker)
       ↓
live Reward Judge chooses useful branch
       ↓
complete conversation
       ↓
independent Eval Judge (DistilBERT ± optional LLM judge, flag if Δ ≥ 0.35)
       ↓
redaction (email, phone, card, IP, SSN-like, VIN-like patterns)
       ↓
JSONL transcript + JSON report (per-JBB-category rates when seeded from JBB)

──────────────────────── POST-CAMPAIGN ANALYSIS ────────────────────────

selected successful prompts
          ↓
       MiniLM (all-MiniLM-L6-v2)
          ↓
 normalized semantic embeddings
          ↓
    cosine distance matrix
          ↓
        DBSCAN (eps 0.25, min 2)
          ↓
 recurring strategy families + noise

──────────────── ML COURSEWORK ANALYSIS  (main_analysis.py) ────────────────

 synthetic campaign (data/generate_synthetic_campaign.py)
 978 turns / 220 conversations, same row schema as a real transcript,
 known hidden state per turn
          ↓
 src/analysis/utils.py  → load rows, judge-score columns, split by conversation
          ↓
 ┌ Review 1 ── eda.py, severity_regression.py, state_classification.py
 │             EDA · linear regression · logistic regression · SVM
 ├ Review 2 ── unsupervised.py   (NumPy, from scratch)
 │             PCA → K-means · Gaussian mixture (EM) · hierarchical clustering
 │             scored with silhouette, ARI and matched accuracy vs the true state
 └ Review 3 ── strategy_clustering.py, compliance_hmm.py, comparison.py
               DBSCAN · HMM · comparative analysis
          ↓
 outputs/analysis/metrics.json + plots
```

## What each stage is for

| Stage | One-line purpose |
|---|---|
| JailbreakBench | Public, citable benchmark goals. They are used only as held-out campaign seeds, and the report breaks results down by JBB category. |
| Train judges | Two DistilBERT classifiers trained separately. The Reward Judge gives PPO its training signal. The Eval Judge grades campaign outcomes and never sends a gradient back to the attacker. |
| Attacker training | Qwen2.5-0.5B with LoRA adapters and a value head, trained by PPO. The KL penalty to the frozen base stops output from drifting into gibberish. The MiniLM penalty stops the attacker from collapsing onto one strategy. |
| Campaign | Runs multi-turn conversations against an allowlisted target. Picks the best of N techniques at each turn. Scores the whole conversation, redacts it, and writes a JSONL transcript plus a report. |
| Post-campaign | Embeds successful prompts with MiniLM and finds recurring strategy families with DBSCAN. |
| ML coursework | Course algorithms run on transcript-shaped data with a known hidden state, so every model can be checked against ground truth. Review 2's unsupervised models are written from their equations. |
