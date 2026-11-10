# FTEC5660 IP2 — Reproducing A-MEM (Agentic Memory)

**Author:** Wang Shuo
**Model:** `deepseek-chat` via DeepSeek API (OpenAI-compatible)
**Date:** November 2026

## What this repo does

We re-implement the core mechanism of **A-MEM: Agentic Memory for LLM Agents**
(Zhang et al., 2025) from the paper's description, and benchmark it against a
naive "keep last N turns" baseline on a synthetic multi-hop QA task.

The agent:
1. Extracts `(subject, relation, object)` triplets from every user turn via an LLM call.
2. Embeds each triplet locally with TF-IDF (no HuggingFace download — see Debug Diary).
3. After `consolidate_after` new triplets, asks the LLM to merge / dedup entries.
4. At answer time, retrieves the top-k triplets by cosine similarity and injects them into the QA prompt.

## Reproduction target

- **Claim:** an agentic memory layer improves long-context QA over naive sliding-window history.
- **Metric:** exact-match accuracy on 51 synthetic episodes (17 easy / 17 medium / 17 hard).
- **Our number:** A-MEM = **100%** (51/51), naive = **21.6%** (11/51).

## Modification

We varied the consolidation trigger `consolidate_after` ∈ {2, 5, 999 (=never)}:

| Config | Accuracy | Avg latency / episode |
|---|---|---|
| ca=2 (frequent consolidation) | 100% | 0.72 s |
| ca=5 (paper-like default)     | 100% | 0.71 s |
| ca=999 (no consolidation)      | 100% | 0.75 s |

**Finding:** on this clean benchmark, consolidation does not move accuracy —
the triplet extractor already cleans input well enough. The cost (extra LLM
calls) is real but small. The paper's claimed benefit is likely to appear on
noisy / high-volume memory streams, not on a clean synthetic set.

## Install & run

```bash
pip install -r requirements.txt
# set DEEPSEEK_API_KEY in .env
python benchmark.py                 # regenerates data/benchmark.jsonl
python run.py --consolidate-after 5 --out results/baseline_ca5.json
python run.py --consolidate-after 999 --out results/mod_noconsol.json
```

## Files

- `amem.py` — A-MEM agent + naive baseline
- `benchmark.py` — synthetic episode generator (3 difficulty levels)
- `run.py` — evaluation harness
- `results/` — JSON outputs from each run
- `report.md` — the 10-page report

No API keys are committed. `.env` is gitignored.