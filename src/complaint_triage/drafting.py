"""Draft a first response to a complaint, grounded in the policy corpus.

Pipeline: redact PII -> (classifier gives product) -> retrieve chunks for that
product -> prompt the LLM with the chunks -> check citations -> return the
draft with its evidence, token usage and cost.

Configuration comes from `.env` (never committed):
    ANTHROPIC_API_KEY=...
    DRAFT_MODEL=claude-haiku-4-5          # optional override
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv

from . import config as C
from .guardrails import CitationReport, check_citations, redact_pii
from .retrieval import Chunk, Retriever

load_dotenv(C.ROOT / ".env")

DRAFT_MODEL = os.getenv("DRAFT_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5")

# USD per million tokens (input, output). Update if pricing changes.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
}


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (in_tok * pin + out_tok * pout) / 1e6


SYSTEM_PROMPT = """You draft first responses to consumer complaints on behalf of Northbridge Financial's complaints team.

Rules:
- Use ONLY the numbered policy excerpts provided. Every timeline, dollar figure, or regulatory step you state must come from an excerpt, and you must cite it inline as [n] immediately after the sentence that uses it.
- Cite the excerpt that literally contains the number or step you are stating. Before writing a timeline, find it in an excerpt and cite THAT excerpt's number; if you cannot find the exact number in any excerpt, do not state a timeline at all.
- Do not describe actions as already taken (refunded, corrected, closed). Describe what the team will do.
- If the excerpts do not cover something the consumer asked, say the team will follow up on that point rather than inventing an answer.
- Do not admit fault or liability. Do not use exclamation marks. Do not ask for a full SSN, full account number, or password.
- Structure: (1) acknowledge the specific issue, (2) what we will do and by when, (3) what we need from the consumer, (4) case reference and contact line. Under 250 words.
- Address the consumer as "Dear Customer". Sign as "Northbridge Financial Complaints Team".
"""


@dataclass
class DraftResult:
    draft: str
    product: str
    chunks: list[dict]
    citations: dict
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    pii_redacted: dict[str, int] = field(default_factory=dict)
    used_retrieval: bool = True


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.title}) {c.text}" for i, c in enumerate(chunks, 1))


def build_user_prompt(narrative: str, product: str, chunks: list[Chunk], case_ref: str) -> str:
    excerpts = _format_chunks(chunks) if chunks else "(no policy excerpts available)"
    return (f"Predicted product category: {product}\nCase reference: {case_ref}\n\n"
            f"POLICY EXCERPTS:\n{excerpts}\n\n"
            f"CONSUMER COMPLAINT:\n{narrative}\n\n"
            "Write the first response now.")


class Drafter:
    def __init__(self, client=None, retriever: Retriever | None = None, model: str = DRAFT_MODEL, k: int = 4):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
        self.client = client
        self.retriever = retriever or Retriever()
        self.model = model
        self.k = k

    def draft(self, narrative: str, product: str, case_ref: str = "NB-000000",
              use_retrieval: bool = True) -> DraftResult:
        clean, pii = redact_pii(narrative)
        chunks = [c for c, _ in self.retriever.search(clean, k=self.k, product_filter=product)] if use_retrieval else []
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model, max_tokens=600, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(clean, product, chunks, case_ref)}],
        )
        latency = time.time() - t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        rep: CitationReport = check_citations(text, len(chunks), [c.text for c in chunks])
        return DraftResult(
            draft=text, product=product,
            chunks=[{"id": c.id, "title": c.title, "text": c.text} for c in chunks],
            citations=asdict(rep) | {"ok": rep.ok},
            model=self.model, input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
            cost_usd=cost_usd(self.model, resp.usage.input_tokens, resp.usage.output_tokens),
            latency_s=latency, pii_redacted=pii, used_retrieval=use_retrieval,
        )
