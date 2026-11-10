"""Minimal A-MEM (Agentic Memory) re-implementation for FTEC5660 IP2.

Reference: "A-MEM: Agentic Memory for LLM Agents" (Zhang et al., 2025).
Core idea:
  1. On every user turn, ask the LLM to extract (subject, relation, object)
     triplets and store them as memory nodes with embeddings.
  2. After `consolidate_after` new nodes, ask the LLM to merge / resolve
     duplicate or conflicting triplets (agentic consolidation).
  3. At answer time, retrieve top-k triplets by cosine similarity to the
     question and inject them into the prompt.

We use DeepSeek as both the extractor/consolidator and the final QA model,
and `sentence-transformers` (all-MiniLM-L6-v2) for local embeddings so we
do not depend on OpenAI embeddings.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

from openai import OpenAI


@dataclass
class MemoryNode:
    id: int
    subject: str
    relation: str
    obj: str
    salience: float = 1.0
    embedding: list[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.subject} | {self.relation} | {self.obj}"


class AMemAgent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "deepseek-chat",
        consolidate_after: int = 5,
        top_k: int = 5,
    ):
        self.client = client
        self.model = model
        self.consolidate_after = consolidate_after
        self.top_k = top_k
        self.nodes: list[MemoryNode] = []
        self._next_id = 0
        self._pending = 0  # new nodes since last consolidation
        self._vec = None
        self._vec_corpus_size = -1
        # lazy embedding model
        self._emb = None

    # ---------- embeddings (local TF-IDF, no network) ----------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a persistent TF-IDF vectorizer. Refits when new
        nodes arrive so the vocabulary stays stable between write and retrieve."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        # Corpus = all known node texts (so the space includes stored facts)
        corpus = [n.text for n in self.nodes] + texts
        if len(corpus) < 2:
            corpus = corpus + ["subject relation object", "another fact here"]
        if self._vec is None or len(corpus) != self._vec_corpus_size:
            self._vec = TfidfVectorizer(lowercase=True, stop_words="english").fit(corpus)
            self._vec_corpus_size = len(corpus)
        arr = self._vec.transform(texts).toarray()
        norms = (arr ** 2).sum(axis=1, keepdims=True) ** 0.5 + 1e-9
        return (arr / norms).tolist()

    # ---------- LLM helper ----------
    def _chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""

    # ---------- memory writing ----------
    def write_memory(self, user_turn: str) -> int:
        """Extract triplets from a user turn and append them."""
        sys = (
            "You extract factual triples from user messages. "
            "Return ONE triple per line as: SUBJECT | RELATION | OBJECT. "
            "Only extract durable facts (people, places, dates, preferences, "
            "relationships). Do not extract greetings or questions. "
            "If there are no durable facts, return exactly: NONE."
        )
        out = self._chat(sys, user_turn).strip()
        added = 0
        for line in out.splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE" or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3 or not all(parts):
                continue
            subj, rel, obj = parts
            emb = self._embed([f"{subj} | {rel} | {obj}"])[0]
            self.nodes.append(MemoryNode(
                id=self._next_id, subject=subj, relation=rel, obj=obj,
                salience=1.0, embedding=emb,
            ))
            self._next_id += 1
            added += 1
        self._pending += added
        if self._pending >= self.consolidate_after:
            self.consolidate()
        return added

    # ---------- agentic consolidation ----------
    def consolidate(self) -> None:
        """Ask LLM to merge / dedup / correct recent triplets."""
        if not self.nodes:
            return
        recent = self.nodes[-max(2 * self.consolidate_after, 10):]
        listing = "\n".join(f"{n.id}: {n.text}" for n in recent)
        sys = (
            "You are consolidating a knowledge base. Given triples, identify "
            "duplicates, contradictions, and over-specific entries. "
            "Return a JSON list of edits. Each edit is: "
            "{'action':'drop','id':N} or {'action':'merge','keep':N,'into':M,'note':'...'}. "
            "Return ONLY JSON, no prose."
        )
        user = f"Triples:\n{listing}\n\nReturn JSON list of edits."
        out = self._chat(sys, user).strip()
        m = re.search(r"\[.*\]", out, re.DOTALL)
        if not m:
            self._pending = 0
            return
        import json
        try:
            edits = json.loads(m.group(0))
        except Exception:
            self._pending = 0
            return
        by_id = {n.id: n for n in self.nodes}
        for e in edits:
            try:
                if e.get("action") == "drop" and e.get("id") in by_id:
                    self.nodes = [n for n in self.nodes if n.id != e["id"]]
                elif e.get("action") == "merge" and e.get("keep") in by_id and e.get("into") in by_id:
                    keep, into = by_id[e["keep"]], by_id[e["into"]]
                    keep.salience += into.salience * 0.5
                    self.nodes = [n for n in self.nodes if n.id != e["keep"]]
            except Exception:
                continue
        self._pending = 0

    # ---------- retrieval ----------
    def retrieve(self, query: str) -> list[MemoryNode]:
        if not self.nodes:
            return []
        q_emb = self._embed([query])[0]
        scored = []
        for n in self.nodes:
            cos = sum(a * b for a, b in zip(q_emb, n.embedding))
            scored.append((cos * n.salience, n))
        scored.sort(key=lambda x: -x[0])
        top = [n for _, n in scored[: self.top_k]]
        for n in top:
            n.salience += 0.1  # reinforcement
        return top

    # ---------- QA ----------
    def answer(self, question: str) -> str:
        ctx = self.retrieve(question)
        facts = "\n".join(f"- {n.text}" for n in ctx) or "- (no facts retrieved)"
        sys = (
            "Answer the question using ONLY the provided facts. "
            "If the facts are insufficient, say 'I do not have enough information.' "
            "Be concise."
        )
        user = f"Known facts:\n{facts}\n\nQuestion: {question}"
        return self._chat(sys, user).strip()


class NaiveBaseline:
    """Naive agent: keeps the last `history_len` turns verbatim."""

    def __init__(self, client: OpenAI, model: str = "deepseek-chat", history_len: int = 6):
        self.client = client
        self.model = model
        self.history_len = history_len
        self.history: list[str] = []

    def write_memory(self, user_turn: str) -> int:
        self.history.append(user_turn)
        if len(self.history) > self.history_len * 2:
            self.history = self.history[-self.history_len * 2 :]
        return 0

    def answer(self, question: str) -> str:
        ctx = "\n".join(f"- {h}" for h in self.history[-self.history_len :]) or "- (empty)"
        sys = (
            "Answer the question using ONLY the conversation history. "
            "If the history does not contain enough information, say "
            "'I do not have enough information.' Be concise."
        )
        user = f"Conversation history:\n{ctx}\n\nQuestion: {question}"
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user", "content": user}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception:
                time.sleep(2 ** attempt)
        return ""
