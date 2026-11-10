"""Synthetic multi-hop memory benchmark for A-MEM.

Each episode is a list of `turns` (user statements the agent must remember)
and a `question` + `answer` pair. Episodes are generated from templates so
we have a clean ground truth.

Three difficulty levels:
  easy   : the answer is in a single turn (1-hop)
  medium : the answer requires combining 2 turns (2-hop)
  hard   : the answer requires combining 3 turns (3-hop)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class Episode:
    eid: str
    turns: list[str]
    question: str
    answer: str
    level: str


FIRST = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Peter",
         "Queen", "Rose", "Sam", "Tina"]
CITY = ["Berlin", "Tokyo", "Toronto", "Sydney", "Zurich", "Austin", "Lyon",
        "Osaka", "Boston", "Madrid", "Seoul", "Dublin"]
COMPANY = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Wonka",
           "Hooli", "Pied Piper", "Vandelay"]
PRODUCT = ["Orion", "Nova", "Helios", "Aurora", "Zenith", "Pulse", "Echo",
           "Flux", "Quasar", "Vertex"]
PET = ["Mochi", "Biscuit", "Shadow", "Luna", "Rocky", "Coco", "Buddy", "Milo"]
COLOR = ["navy", "maroon", "teal", "ochre", "ivory", "coral", "slate", "olive"]
LANG = ["Python", "Rust", "Go", "Julia", "TypeScript", "Java", "C++", "Ruby"]


def gen(seed: int = 42, n_each: int = 17) -> list[Episode]:
    rng = random.Random(seed)
    eps: list[Episode] = []
    eid = 0
    FILLER = [
        "The weather in Paris was rainy yesterday.",
        "I had oatmeal for breakfast.",
        "My neighbor plays the piano every evening.",
        "There is a new coffee shop on 5th street.",
        "I watered my plants this morning.",
        "The bus was late by ten minutes today.",
        "I finished a book about mountain climbing.",
        "A package arrived on Tuesday.",
        "The office printer ran out of paper.",
        "I went for a run along the river.",
        "My phone battery died mid-afternoon.",
        "A storm knocked out power on Sunday.",
    ]
    for level in ("easy", "medium", "hard"):
        for _ in range(n_each):
            name = rng.choice(FIRST)
            city = rng.choice(CITY)
            company = rng.choice(COMPANY)
            product = rng.choice(PRODUCT)
            pet = rng.choice(PET)
            color = rng.choice(COLOR)
            lang = rng.choice(LANG)
            # Filler turns (8) push key facts out of a 6-turn naive window.
            fillers = rng.sample(FILLER, 8)
            if level == "easy":
                key_turns = [
                    f"Hi, I'm {name}. My favorite programming language is {lang}.",
                ]
                q = f"What is {name}'s favorite programming language?"
                a = lang
            elif level == "medium":
                key_turns = [
                    f"Hi, I'm {name}. I work at {company}.",
                    f"{company} is headquartered in {city}.",
                ]
                q = f"In which city is {name}'s company headquartered?"
                a = city
            else:  # hard, 3-hop
                key_turns = [
                    f"Hi, I'm {name}. I work at {company}.",
                    f"{company}'s flagship product is called {product}.",
                    f"{product} is written primarily in {lang}.",
                ]
                q = f"What programming language is used to build {name}'s company's flagship product?"
                a = lang
            # Interleave key turns early, then fillers; question comes last.
            turns = key_turns[:1] + fillers[:4] + key_turns[1:] + fillers[4:]
            eps.append(Episode(
                eid=f"e{eid:03d}", turns=turns, question=q, answer=a, level=level,
            ))
            eid += 1
    return eps


def write_jsonl(path: str, eps: list[Episode]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ep in eps:
            f.write(json.dumps({
                "eid": ep.eid, "level": ep.level, "turns": ep.turns,
                "question": ep.question, "answer": ep.answer,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    eps = gen()
    write_jsonl("data/benchmark.jsonl", eps)
    levels = {}
    for e in eps:
        levels[e.level] = levels.get(e.level, 0) + 1
    print(f"wrote {len(eps)} episodes: {levels}")
