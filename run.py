"""Run the benchmark: naive baseline vs A-MEM, with configurable consolidate_after."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

from amem import AMemAgent, NaiveBaseline
from benchmark import gen


def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def score(prediction: str, gold: str) -> bool:
    p = prediction.lower().strip().rstrip(".")
    g = gold.lower().strip()
    # Direct substring match; avoids \b issues with tokens like "C++".
    return g in p


def run(consolidate_after: int, out_path: str, model: str = "deepseek-chat",
        limit: int | None = None):
    load_env()
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )
    eps = gen()
    if limit:
        eps = eps[:limit]

    results = []
    for i, ep in enumerate(eps):
        print(f"[{i+1}/{len(eps)}] {ep.eid} ({ep.level}) ...", flush=True)
        # fresh agent per episode (memory is per-episode)
        amem = AMemAgent(client, model=model, consolidate_after=consolidate_after)
        naive = NaiveBaseline(client, model=model)
        for t in ep.turns:
            amem.write_memory(t)
            naive.write_memory(t)
        t0 = time.time()
        a_pred = amem.answer(ep.question)
        a_time = time.time() - t0
        t0 = time.time()
        n_pred = naive.answer(ep.question)
        n_time = time.time() - t0
        results.append({
            "eid": ep.eid, "level": ep.level,
            "question": ep.question, "gold": ep.answer,
            "amem_pred": a_pred, "amem_correct": score(a_pred, ep.answer),
            "amem_time_s": round(a_time, 2),
            "naive_pred": n_pred, "naive_correct": score(n_pred, ep.answer),
            "naive_time_s": round(n_time, 2),
            "nodes_stored": len(amem.nodes),
        })

    # aggregate
    def agg(key):
        total = sum(1 for r in results if r[key])
        by_level = {}
        for r in results:
            by_level.setdefault(r["level"], [0, 0])
            by_level[r["level"]][1] += 1
            if r[key]:
                by_level[r["level"]][0] += 1
        return total, {k: v[0] / v[1] for k, v in by_level.items()}

    a_total, a_by = agg("amem_correct")
    n_total, n_by = agg("naive_correct")
    summary = {
        "model": model,
        "consolidate_after": consolidate_after,
        "n_episodes": len(results),
        "amem_accuracy": a_total / len(results),
        "amem_by_level": a_by,
        "naive_accuracy": n_total / len(results),
        "naive_by_level": n_by,
        "amem_avg_time": sum(r["amem_time_s"] for r in results) / len(results),
        "naive_avg_time": sum(r["naive_time_s"] for r in results) / len(results),
    }
    out = {"summary": summary, "results": results}
    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidate-after", type=int, default=5)
    ap.add_argument("--out", default="results/run_ca5.json")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run(args.consolidate_after, args.out, args.model, args.limit)
