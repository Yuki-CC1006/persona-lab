"""Run the offline quality baseline against a real database.

Usage:
    uv run python scripts/quality_baseline.py            # in-memory, golden profile
    uv run python scripts/quality_baseline.py --attach    # use the real local database (read-only)

The golden profile is only written into a throwaway in-memory database, so
running this never touches your actual data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.evaluation import (  # noqa: E402
    compare_embeddings,
    embedding_variants,
    evaluate,
    format_embedding_report,
    format_report,
    load_golden_set,
    seed_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Persona Lab retrieval and refusal quality baseline")
    parser.add_argument(
        "--attach",
        action="store_true",
        help="Use the real local database (read-only; the golden profile is never written)",
    )
    args = parser.parse_args()

    golden = load_golden_set()

    if args.attach:
        from app.db.session import get_engine, get_session_factory

        get_engine()
        db = get_session_factory()()
    else:
        from app.db.session import get_session_factory, reset_engine_for_tests

        reset_engine_for_tests("sqlite:///:memory:")
        db = get_session_factory()()
        seed_profile(db, golden)

    try:
        metrics = evaluate(db, golden)
        print(format_report(metrics))
        print()
        variants = embedding_variants()
        if len(variants) == 1:
            print("Embedding comparison: local hashing only (EMBEDDING_BASE_URL / EMBEDDING_MODEL not configured)")
        else:
            print("Embedding comparison:")
            print(
                format_embedding_report(
                    compare_embeddings(db, golden, variants=variants)
                )
            )
    finally:
        db.close()

    return 0 if not metrics.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
