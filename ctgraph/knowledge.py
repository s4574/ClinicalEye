"""DeepTumorVQA's own knowledge base (the benchmark's `lookup_medical_knowledge` tool), retrieved by keywords.

The 27 entries are parsed as a literal from the cloned benchmark repo (data/ext/dtvqa_code/src/deeptumorvqa/
eval/tools.py, KNOWLEDGE_BASE) with ast.literal_eval - the repo's code is never imported or executed.
Retrieval mimics the tool: entries ranked by how many of their keywords occur in the query.
Giving these definitions is fair: the benchmark's agent settings (oracle / predicted / vision) all expose this tool.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

from . import paths as P

TOOLS_PY = P.DTVQA_SRC / "deeptumorvqa/eval/tools.py"


@lru_cache(1)
def knowledge_base() -> list[dict]:
    src = TOOLS_PY.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name) and tgt.id == "KNOWLEDGE_BASE":
                return ast.literal_eval(node.value)
    raise RuntimeError("KNOWLEDGE_BASE not found")


def lookup(query: str, k: int = 3) -> list[str]:
    q = query.lower()
    words = set(re.findall(r"[a-z0-9/]+", q))
    scored = []
    for e in knowledge_base():
        s = sum(1 for kw in e["keywords"] if kw.lower() in words or (" " in kw and kw.lower() in q))
        if s:
            scored.append((s, e["text"]))
    scored.sort(key=lambda t: -t[0])
    return [t for _, t in scored[:k]]


def knowledge_text(question: str, k: int = 3) -> str:
    hits = lookup(question, k)
    if not hits:
        return ""
    return ("BENCHMARK KNOWLEDGE (definitions used by this benchmark, from its knowledge-lookup tool; follow them):\n"
            + "\n".join(f"- {h}" for h in hits))
