"""Guardrails around the LLM call.

Input side: redact PII before the narrative leaves the process. CFPB already
masks most PII as XXXX, but production complaints will not be pre-masked.

Output side: every factual claim in a draft should trace to a retrieved chunk.
The draft cites chunks as [1], [2], ...; `check_citations` verifies that the
draft cites at least one chunk, cites only chunks that exist, and does not
mention timelines ("N days") that do not appear in the cited chunks. That last
check is a cheap, deterministic hallucination detector for the claims that
matter most in a regulatory response.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "account": re.compile(r"\b(?:acct|account)\s*(?:#|no\.?|number)?\s*:?\s*\d{6,}\b", re.IGNORECASE),
    "long_digits": re.compile(r"\b\d{8,}\b"),
}


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pat in PII_PATTERNS.items():
        text, n = pat.subn(f"[{name.upper()}]", text)
        if n:
            counts[name] = n
    return text, counts


CITE_RE = re.compile(r"\[(\d+)\]")
DAYS_RE = re.compile(r"\b(\d+)\s*(?:business\s+)?days?\b", re.IGNORECASE)


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    unsupported_timelines: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.cited) and not self.invalid and not self.unsupported_timelines


def check_citations(draft: str, n_chunks: int, chunk_texts: list[str]) -> CitationReport:
    rep = CitationReport()
    rep.cited = sorted({int(m) for m in CITE_RE.findall(draft)})
    rep.invalid = [i for i in rep.cited if not 1 <= i <= n_chunks]
    cited_text = " ".join(chunk_texts[i - 1] for i in rep.cited if 1 <= i <= n_chunks).lower()
    for m in DAYS_RE.finditer(draft):
        phrase = m.group(0).lower()
        num = m.group(1)
        # accept if the same number of days appears anywhere in the cited chunks
        if not re.search(rf"\b{num}\s*(?:business\s+)?days?\b", cited_text):
            rep.unsupported_timelines.append(phrase)
    return rep
