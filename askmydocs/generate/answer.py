from __future__ import annotations

from dataclasses import dataclass
import os
import re
from openai import OpenAI

from askmydocs.generate.citations import build_citations, format_citation_block


# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class Answer:
    text: str
    citations_block: str
    used_chunks: list[str]
    abstained: bool
    evidence_score: float


# ----------------------------
# OpenAI client
# ----------------------------
_client = OpenAI()  # reads OPENAI_API_KEY from env


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None and v.strip() != "" else default


def _openai_generate(prompt: str) -> str:
    """
    Uses OpenAI Responses API. Model defaults to gpt-4.1-nano.
    """
    model = _env("OPENAI_MODEL", "gpt-4.1-nano")

    resp = _client.responses.create(
        model=model,
        input=prompt,
    )
    return (resp.output_text or "").strip()


# ----------------------------
# Prompting
# ----------------------------
def _build_prompt(question: str, contexts: list[str], cite_keys: list[str]) -> str:
    joined = "\n\n".join(f"{key}\n{ctx}" for key, ctx in zip(cite_keys, contexts))

    return f"""You are a factual assistant. Answer ONLY using the provided sources.
If the sources do not contain enough information, say: "I don’t have enough evidence in the provided documents to answer that."

Rules:
- Every major claim MUST include at least one citation like [1], [2], etc.
- Do NOT invent citations.
- Do NOT use outside knowledge.
- Keep the answer concise (<= 120 words).

Question:
{question}

Sources:
{joined}

Write the answer now.
"""


_CITATION_RE = re.compile(r"\[\d+\]")


# ----------------------------
# Main entry point
# ----------------------------
def answer_with_citations(
    question: str,
    reranked_hits: list,
    min_evidence_score: float | None = None,
    max_sources: int = 2,
) -> Answer:
    """
    reranked_hits: list of RerankHit with .score, .text, .meta, .chunk_id
    """
    if min_evidence_score is None:
        min_evidence_score = float(_env("MIN_EVIDENCE_SCORE", "0.0"))

    if not reranked_hits:
        return Answer(
            text="I don’t have enough evidence in the provided documents to answer that.",
            citations_block="",
            used_chunks=[],
            abstained=True,
            evidence_score=0.0,
        )

    top_score = float(reranked_hits[0].score)

    # Abstain if evidence is weak
    if top_score < min_evidence_score:
        citations, _ = build_citations(reranked_hits, max_sources=max_sources)
        return Answer(
            text="I don’t have enough evidence in the provided documents to answer that.",
            citations_block=format_citation_block(citations),
            used_chunks=[c.chunk_id for c in citations],
            abstained=True,
            evidence_score=top_score,
        )

    citations, chunk_to_key = build_citations(reranked_hits, max_sources=max_sources)

    # Keep contexts short for speed/cost
    contexts = [h.text[:1200] for h in reranked_hits[:max_sources]]
    cite_keys = [chunk_to_key[h.chunk_id] for h in reranked_hits[:max_sources]]

    prompt = _build_prompt(question, contexts, cite_keys)

    model_answer = _openai_generate(prompt)

    # Enforce citations (at least one) – fast check
    has_any_cite = any(k in model_answer for k in cite_keys) or bool(_CITATION_RE.search(model_answer))

    if not has_any_cite:
        # One retry with stronger citation instruction
        retry_prompt = prompt + "\n\nIMPORTANT: Rewrite the answer and include citations like [1], [2] for each key claim."
        model_answer_2 = _openai_generate(retry_prompt)

        has_any_cite_2 = any(k in model_answer_2 for k in cite_keys) or bool(_CITATION_RE.search(model_answer_2))
        if not has_any_cite_2:
            return Answer(
                text="I don’t have enough evidence in the provided documents to answer that.",
                citations_block=format_citation_block(citations),
                used_chunks=[c.chunk_id for c in citations],
                abstained=True,
                evidence_score=top_score,
            )
        model_answer = model_answer_2

    return Answer(
        text=model_answer,
        citations_block=format_citation_block(citations),
        used_chunks=[c.chunk_id for c in citations],
        abstained=False,
        evidence_score=top_score,
    )