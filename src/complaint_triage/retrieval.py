"""Retrieval over the policy corpus.

Documents are split into paragraph-level chunks, each tagged with its source
file and the product(s) it applies to. Retrieval is BM25 (lexical) by default:
it is dependency-light, fast on CPU, and strong when the query and corpus share
vocabulary, which is the case here (both use product and regulation terms).
A `product_filter` keeps only chunks for the predicted product plus the general
standards, which is where most of the precision comes from.

Swapping in embeddings later means implementing `Retriever.search` with a
different scorer; nothing upstream changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from . import config as C

CORPUS_DIR = C.DATA_DIR / "corpus"
GENERAL = "all"


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str        # file name
    title: str
    products: tuple[str, ...]
    text: str


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not m:
        return {}, raw
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    return meta, m.group(2)


def load_chunks(corpus_dir: Path = CORPUS_DIR, min_chars: int = 40) -> list[Chunk]:
    """One chunk per paragraph, headers dropped, tiny fragments merged away."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        title = meta.get("title", path.stem)
        products = tuple(p.strip() for p in meta.get("products", GENERAL).split("|"))
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip() and not p.strip().startswith("#")]
        for i, para in enumerate(paras):
            if len(para) >= min_chars:
                chunks.append(Chunk(f"{path.stem}#{i}", path.name, title, products, para))
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_chunks()
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    def search(self, query: str, k: int = 4, product_filter: str | None = None) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        cands = []
        for c, s in zip(self.chunks, scores):
            if product_filter and not (product_filter in c.products or GENERAL in c.products):
                continue
            cands.append((c, float(s)))
        cands.sort(key=lambda t: t[1], reverse=True)
        return cands[:k]
