"""Phase 3 tests. A fake client stands in for the API so tests are free and offline."""
from __future__ import annotations

from types import SimpleNamespace

from complaint_triage.drafting import Drafter, build_user_prompt, cost_usd
from complaint_triage.guardrails import check_citations, redact_pii
from complaint_triage.retrieval import Retriever, load_chunks


class FakeClient:
    """Returns a canned draft that cites [1] and uses a timeline from chunk 1."""

    def __init__(self, reply: str):
        self.reply = reply
        self.messages = self

    def create(self, **kw):
        self.last_kwargs = kw
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            usage=SimpleNamespace(input_tokens=900, output_tokens=180),
        )


def test_corpus_loads_with_products():
    chunks = load_chunks()
    assert len(chunks) > 30
    assert any("Debt collection" in c.products for c in chunks)
    assert any("all" in c.products for c in chunks)


def test_retriever_respects_product_filter():
    r = Retriever()
    hits = r.search("provisional credit unauthorized transaction", k=4,
                    product_filter="Checking or savings account")
    assert hits
    for chunk, _ in hits:
        assert "Checking or savings account" in chunk.products or "all" in chunk.products
    assert any("10 business days" in c.text for c, _ in hits)


def test_redact_pii():
    text = "call me at 410-555-1234 or jane@example.com, SSN 123-45-6789, acct 123456789"
    clean, counts = redact_pii(text)
    assert "[PHONE]" in clean and "[EMAIL]" in clean and "[SSN]" in clean
    assert "555" not in clean and "example.com" not in clean
    assert counts["phone"] == 1


def test_citation_check_flags_unsupported_timeline():
    chunks = ["We investigate within 10 business days.", "Provisional credit within 45 days."]
    ok = check_citations("We will investigate within 10 business days [1].", 2, chunks)
    assert ok.ok
    bad = check_citations("We will respond within 3 days [1].", 2, chunks)
    assert not bad.ok and bad.unsupported_timelines == ["3 days"]
    inv = check_citations("See [5].", 2, chunks)
    assert inv.invalid == [5]
    none = check_citations("We will look into it.", 2, chunks)
    assert not none.ok


def test_drafter_end_to_end_with_fake_client():
    reply = "Dear Customer, we have opened a dispute and will investigate within 10 business days [1]."
    client = FakeClient(reply)
    d = Drafter(client=client, k=3)
    res = d.draft("Someone took $500 from my checking account, call 410-555-1234",
                  "Checking or savings account", case_ref="NB-000001")
    assert res.draft == reply
    assert len(res.chunks) == 3
    assert res.pii_redacted.get("phone") == 1
    assert "[PHONE]" in client.last_kwargs["messages"][0]["content"]
    assert res.cost_usd == cost_usd(d.model, 900, 180) > 0
    # citation check runs against the actual retrieved chunks
    assert isinstance(res.citations["ok"], bool)


def test_no_retrieval_prompt_has_no_excerpts():
    prompt = build_user_prompt("text", "Mortgage", [], "NB-1")
    assert "no policy excerpts" in prompt
