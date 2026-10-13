"""Quality baseline for retrieval, evidence and refusal behaviour.

The golden set lives in ``tests/data/golden_qa.json`` and is intentionally
outside the runtime database: it is a fixed measuring stick, not user data.

Everything here runs fully offline (no LLM key required) so the numbers are
deterministic and can be used as a regression guard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Memory, MemoryStatus
from app.services.outbound import record_outbound
from app.services.prediction import answer_question, predict_tendency
from app.services.retrieval import hybrid_search, local_embed

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_PATH = REPO_ROOT / "tests" / "data" / "golden_qa.json"

# A candidate embedding must beat the baseline by this much before we switch.
EMBEDDING_SWITCH_MIN_DELTA = 0.05

Embedder = Callable[[str], list[float]]


class _OfflineLLM:
    """Force the deterministic offline path so metrics never depend on network."""

    configured = False

    def chat_json(self, *_args, **_kwargs):  # pragma: no cover - never reached
        raise RuntimeError("The offline evaluation must not call the model")

    def chat_stream(self, *_args, **_kwargs):  # pragma: no cover - never reached
        raise RuntimeError("The offline evaluation must not call the model")
        yield  # pragma: no cover


_OFFLINE_LLM = _OfflineLLM()


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expect_any: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    expect_refusal: bool = False
    mode: str = "ask"


@dataclass(frozen=True)
class GoldenSet:
    profile: tuple[dict, ...] = ()
    cases: tuple[GoldenCase, ...] = ()
    description: str = ""


@dataclass
class BaselineMetrics:
    case_count: int = 0
    recall_at_k: float = 0.0
    evidence_accuracy: float = 0.0
    wrong_subject_rate: float = 0.0
    refusal_accuracy: float = 0.0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "case_count": self.case_count,
            "recall_at_k": round(self.recall_at_k, 4),
            "evidence_accuracy": round(self.evidence_accuracy, 4),
            "wrong_subject_rate": round(self.wrong_subject_rate, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "failures": list(self.failures),
        }


def load_golden_set(path: Path | None = None) -> GoldenSet:
    source = Path(path) if path else DEFAULT_GOLDEN_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    cases = []
    for raw in data.get("cases") or []:
        if not isinstance(raw, dict) or not raw.get("question"):
            continue
        cases.append(
            GoldenCase(
                id=str(raw.get("id") or raw["question"])[:60],
                question=str(raw["question"]),
                expect_any=tuple(raw.get("expect_any") or ()),
                forbid=tuple(raw.get("forbid") or ()),
                expect_refusal=bool(raw.get("expect_refusal")),
                mode=str(raw.get("mode") or "ask"),
            )
        )
    return GoldenSet(
        profile=tuple(data.get("profile") or ()),
        cases=tuple(cases),
        description=str(data.get("description") or ""),
    )


def seed_profile(db: Session, golden: GoldenSet) -> int:
    """Insert the golden profile as confirmed memories. Returns the count."""
    created = 0
    for item in golden.profile:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        memory = Memory(
            content=content,
            memory_type=str(item.get("memory_type") or "fact"),
            status=MemoryStatus.CONFIRMED.value,
            confidence=float(item.get("confidence") or 1.0),
            subject=item.get("subject"),
            is_inference=bool(item.get("is_inference")),
            conflict_group=item.get("conflict_group"),
            valid_from=_parse_dt(item.get("valid_from")),
            valid_to=_parse_dt(item.get("valid_to")),
            embedding=local_embed(content),
        )
        db.add(memory)
        created += 1
    db.commit()
    return created


def evaluate(
    db: Session,
    golden: GoldenSet | None = None,
    *,
    k: int = 5,
) -> BaselineMetrics:
    golden = golden or load_golden_set()
    retrieval = evaluate_retrieval(db, golden, k=k)
    answers = evaluate_answers(db, golden)
    refusals = evaluate_refusals(db, golden)

    recall_cases = retrieval["cases"]
    forbid_cases = retrieval["forbid_cases"]
    answer_cases = answers["cases"]
    refusal_cases = refusals["cases"]

    failures: list[str] = []
    failures.extend(retrieval["misses"])
    failures.extend(answers["misses"])
    failures.extend(refusals["misses"])

    return BaselineMetrics(
        case_count=len(golden.cases),
        recall_at_k=_ratio(retrieval["hits"], recall_cases),
        evidence_accuracy=_ratio(answers["hits"], answer_cases),
        wrong_subject_rate=_ratio(retrieval["forbidden_hits"], forbid_cases),
        refusal_accuracy=_ratio(refusals["hits"], refusal_cases),
        failures=failures,
    )


def evaluate_retrieval(
    db: Session, golden: GoldenSet, *, k: int = 5
) -> dict[str, object]:
    hits = 0
    cases = 0
    forbidden_hits = 0
    forbid_cases = 0
    misses: list[str] = []

    for case in golden.cases:
        results = hybrid_search(db, case.question, include_messages=True, limit=k)
        blob = " ".join(hit.content for hit in results)

        if case.expect_any:
            cases += 1
            if any(term in blob for term in case.expect_any):
                hits += 1
            else:
                misses.append(f"{case.id}: retrieval missed {'/'.join(case.expect_any)}")
        if case.forbid:
            forbid_cases += 1
            if any(term in blob for term in case.forbid):
                forbidden_hits += 1
                misses.append(f"{case.id}: retrieval leaked to another subject ({'/'.join(case.forbid)})")

    return {
        "hits": hits,
        "cases": cases,
        "forbidden_hits": forbidden_hits,
        "forbid_cases": forbid_cases,
        "misses": misses,
    }


def evaluate_answers(db: Session, golden: GoldenSet) -> dict[str, object]:
    hits = 0
    cases = 0
    misses: list[str] = []

    for case in golden.cases:
        if not case.expect_any or case.expect_refusal:
            continue
        cases += 1
        result = answer_question(db, case.question, llm=_OFFLINE_LLM)
        evidence_blob = " ".join(
            str(item.get("content") or "") for item in (result.get("evidence") or [])
        )
        blob = f"{result.get('answer') or ''} {evidence_blob}"
        if any(term in blob for term in case.expect_any):
            hits += 1
        else:
            misses.append(f"{case.id}: answer did not cite {'/'.join(case.expect_any)}")

    return {"hits": hits, "cases": cases, "misses": misses}


def evaluate_refusals(db: Session, golden: GoldenSet) -> dict[str, object]:
    hits = 0
    cases = 0
    misses: list[str] = []

    for case in golden.cases:
        cases += 1
        if case.mode == "tendency":
            can_answer = predict_tendency(db, case.question, llm=_OFFLINE_LLM).can_answer
        else:
            can_answer = bool(
                answer_question(db, case.question, llm=_OFFLINE_LLM).get("can_answer")
            )
        if can_answer == (not case.expect_refusal):
            hits += 1
        else:
            expectation = "should have refused" if case.expect_refusal else "should have answered"
            misses.append(f"{case.id}: {expectation} but the result was the opposite")

    return {"hits": hits, "cases": cases, "misses": misses}


def semantic_recall(
    db: Session,
    golden: GoldenSet,
    embed: Embedder,
    *,
    k: int = 5,
) -> float:
    """Recall@k of a single embedding variant over confirmed memories."""
    memories = (
        db.query(Memory)
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .all()
    )
    vectors = [(m.content, embed(m.content)) for m in memories]

    hits = 0
    cases = 0
    for case in golden.cases:
        if not case.expect_any:
            continue
        cases += 1
        query = embed(case.question)
        scored = sorted(
            ((_cosine(query, vec), content) for content, vec in vectors),
            reverse=True,
        )
        blob = " ".join(content for _, content in scored[:k])
        if any(term in blob for term in case.expect_any):
            hits += 1
    return _ratio(hits, cases)


def should_switch_embedding(
    baseline_score: float, candidate_score: float, *, min_delta: float = EMBEDDING_SWITCH_MIN_DELTA
) -> bool:
    return candidate_score - baseline_score >= min_delta


class EmbeddingUnavailable(RuntimeError):
    """Raised when an experimental embedding endpoint cannot be used."""


@dataclass
class EndpointEmbedder:
    """OpenAI-compatible ``/embeddings`` client, used only for comparison runs.

    Nothing in the request path calls this: the shipped default stays the local
    hash embedding until the golden set proves a candidate is clearly better.
    """

    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 30.0
    redacted: bool = True
    _cache: dict[str, list[float]] = field(default_factory=dict, repr=False)

    @property
    def host(self) -> str:
        return urlparse(self.base_url).netloc or self.base_url

    def __call__(self, text: str) -> list[float]:
        key = text.strip()
        if not key:
            return []
        if key not in self._cache:
            self._cache[key] = self._fetch(key)
        return self._cache[key]

    def _fetch(self, text: str) -> list[float]:
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        ok = False
        error: str | None = None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    url, headers=headers, json={"model": self.model, "input": text}
                )
            if resp.status_code >= 400:
                error = f"HTTP {resp.status_code}"
                raise EmbeddingUnavailable(f"{self.host} returned HTTP {resp.status_code}")
            data = resp.json()
            vector = (data.get("data") or [{}])[0].get("embedding") or []
            ok = True
            return [float(x) for x in vector]
        except EmbeddingUnavailable:
            raise
        except (httpx.HTTPError, ValueError, AttributeError, IndexError, TypeError) as exc:
            error = type(exc).__name__
            raise EmbeddingUnavailable(
                f"Cannot use the embedding endpoint at {self.host}: {error}"
            ) from exc
        finally:
            # Embedding text can contain private content: only metadata is kept.
            record_outbound(
                provider="embedding",
                host=self.host,
                model=self.model,
                purpose="embedding",
                char_count=len(text),
                redacted=self.redacted,
                ok=ok,
                error=error,
            )


def embedding_variants() -> dict[str, Embedder]:
    """Local hash baseline, plus the configured endpoint if there is one."""
    variants: dict[str, Embedder] = {"local-hash": local_embed}
    settings = get_settings()
    base_url = (settings.embedding_base_url or "").strip()
    model = (settings.embedding_model or "").strip()
    if not base_url or not model:
        return variants
    host = urlparse(base_url).netloc or base_url
    name = (
        "local-endpoint"
        if host.startswith(("127.0.0.1", "localhost", "::1"))
        else "remote-endpoint"
    )
    variants[name] = EndpointEmbedder(
        base_url=base_url, model=model, api_key=settings.embedding_api_key
    )
    return variants


def compare_embeddings(
    db: Session,
    golden: GoldenSet,
    *,
    variants: dict[str, Embedder] | None = None,
    k: int = 5,
) -> dict[str, object]:
    """Compare embedding variants offline. Never switches the default by itself."""
    variants = variants if variants is not None else {"local-hash": local_embed}
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name, fn in variants.items():
        try:
            scores[name] = semantic_recall(db, golden, fn, k=k)
        except EmbeddingUnavailable as exc:
            errors[name] = str(exc)
    baseline_name = "local-hash" if "local-hash" in scores else next(iter(scores), "")
    baseline = scores.get(baseline_name, 0.0)
    result: dict[str, object] = {
        "scores": {name: round(score, 4) for name, score in scores.items()},
        "baseline": baseline_name,
        "baseline_score": round(baseline, 4),
        "min_delta": EMBEDDING_SWITCH_MIN_DELTA,
        "recommended_switch": {
            name: round(score, 4)
            for name, score in scores.items()
            if name != baseline_name and should_switch_embedding(baseline, score)
        },
    }
    if errors:
        result["errors"] = errors
    return result


def format_embedding_report(result: dict[str, object]) -> str:
    scores = result.get("scores") or {}
    lines = [
        f"Baseline: {result.get('baseline')} (Recall@k {result.get('baseline_score')})",
        "Candidates: " + "; ".join(f"{name} {score}" for name, score in scores.items()),
    ]
    switch = result.get("recommended_switch") or {}
    if switch:
        lines.append(
            f"Recommended switch (gain ≥ {result.get('min_delta')}): "
            + ", ".join(f"{name} ({score})" for name, score in switch.items())
        )
    else:
        lines.append("Recommend keeping the baseline: no candidate reached the meaningful-gain threshold")
    for name, error in (result.get("errors") or {}).items():
        lines.append(f"{name} unavailable: {error}")
    return "\n".join(lines)


def format_report(metrics: BaselineMetrics) -> str:
    lines = [
        f"Cases: {metrics.case_count}",
        f"Recall@k: {metrics.recall_at_k:.2%}",
        f"Evidence accuracy: {metrics.evidence_accuracy:.2%}",
        f"Wrong-subject rate: {metrics.wrong_subject_rate:.2%}",
        f"Refusal accuracy: {metrics.refusal_accuracy:.2%}",
    ]
    if metrics.failures:
        lines.append("Failures: " + "; ".join(metrics.failures))
    return "\n".join(lines)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _ratio(hits: int, total: int) -> float:
    return hits / total if total else 1.0


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
