# Persona Lab

A local-first **persona lab**: turn conversation evidence from one relationship into long-term memories that are traceable, reviewable, and deletable, and deliver a reproducible quality baseline for retrieval and refusal.

It does not predict anyone, and it does not make judgments for you. Tendency inference is an **experimental** feature that arrives with evidence: it gives a confidence level, supporting evidence, and counterexamples, and it clearly refuses when the evidence is insufficient.

Positioning boundaries (what it is not, up front):

- **Not** an "AI girlfriend / predict your girlfriend" tool. It does not presume the nature of a relationship — friends, family, colleagues, and partners are all treated the same.
- **Not** a general-purpose assistant. General knowledge, programming, and unrelated tasks are refused by the local scope gate (see below).
- **Not** a cloud service. By default, your data lives only in your own SQLite database.

> Hermes Agent is **not** a runtime dependency. The built-in `USER.md` has a very small capacity and is not suitable as a chat fact store. This project keeps the full data in a local SQLite database; you can optionally export a refined, confirmed profile via `/api/hermes/profile`.

## Feature boundaries

- Raw messages cannot be overwritten by the model; duplicate imports are idempotent by message fingerprint.
- Candidate memories extracted by AI from imported content must be confirmed by a human before they enter the profile and tendency analysis.
- During conversations, the profile is updated directly only when there is an explicit instruction such as "remember" or "correct"; ordinary statements are never written silently.
- Tendency inference is **evidence-backed inference**, not behavior prediction: it attaches a confidence level, supporting evidence, and counterexamples; when the evidence is insufficient it clearly says "cannot determine".
- By default it binds only to `127.0.0.1`; you can preview the redacted result before sending anything to the cloud.
- Confirmed memories can be exported as Markdown (`/profile/export.md`), which is human-readable and diffable; candidates and raw chat never make it into the export.

## How it differs from similar tools

Tools in this space broadly fall into two camps: one distills chat logs into a **"digital clone" that imitates how they talk** (e.g. WeLoom, wechat-skill-distill), and the other is a **general memory layer for agents** (e.g. Mem0, Vault, Recall). Persona Lab deliberately sits in the middle:

| | Digital clone | Agent memory layer | Persona Lab |
|---|---|---|---|
| Default narrative | Revival / companionship / lifelike replies | For agents | An **auditable evidence base** for one relationship |
| Memory writing | Fully automatic distillation | Agent writes automatically | AI only proposes candidates; they enter the store only after **human confirmation** |
| When there is no evidence | Tends to make things up convincingly | Returns retrieval results directly | Local scope gate + **explicit refusal** |
| Data boundary | Depends on the implementation | Mostly local | Defaults to `127.0.0.1` + redaction preview + encrypted backups |
| External interface | Mainly chat | Write-MCP | **Read-only** MCP |

What it aims for is "help me record the facts of this relationship clearly, look them up, and explain the basis" — not "play that person for me".

## Quick start

```bash
cd persona-lab
cp .env.example .env
# Edit .env and fill in your DeepSeek LLM_API_KEY. Import/review/search work without it.

uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Open in your browser: http://127.0.0.1:8765

You can also use:

```bash
uv run persona-lab
```

## Importing WeChat data

WeChat officially **cannot** export chat text. First obtain JSON/JSONL through your own method, with both parties' consent, and then import it.

Supported formats:

1. **WeFlow session JSON**: `{ "session": {...}, "messages": [...] }`, uses `isSend` to distinguish the two sides automatically, and skips non-text messages (recommended)
2. WeChat JSON array: `[{ "sender", "content", "timestamp", "type" }, ...]`
3. Object wrapper: `{ "messages": [ ... ] }` (also recognizes `data` / `chats` / `records`)
4. JSONL: one object per line
5. Generic chat text and Markdown (paste or upload `.txt`/`.md`), each line written as "nickname: content" or with a time prefix

See the examples in [`samples/weflow_sample.json`](samples/weflow_sample.json), [`samples/wechat_sample.json`](samples/wechat_sample.json), and [`samples/wechat_sample.jsonl`](samples/wechat_sample.jsonl). The "Try an example" button on the import page loads this fictional data directly, so you can run the whole flow with zero data.

Flow: choose a source → preview the field mapping and redaction → fill in both nicknames → confirm the import → (optional) AI extraction → confirm/reject in "Candidate review".

To delete a particular import, open the bottom of that import's detail page and click "Delete this import". It will first list the impact (messages, chunks, candidates, and confirmed memories that would lose their evidence), and after you confirm it cleans up only the data produced by that import.

## Configuration

| Variable | Meaning |
|------|------|
| `LLM_API_KEY` | DeepSeek API key ([get one here](https://platform.deepseek.com/api_keys)); empty means offline mode |
| `LLM_BASE_URL` | Defaults to `https://api.deepseek.com/v1` |
| `LLM_MODEL` | Defaults to `deepseek-v4-flash`; you can also switch between Flash/Pro and thinking mode on the settings page |
| `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` | Optional OpenAI-compatible embedding endpoint, used only for comparison during quality evaluation |
| `REDACTION_ENABLED` | Redaction switch applied before sending |
| `BIND_HOST` / `BIND_PORT` | Listen address (please keep it at 127.0.0.1) |
| `DATABASE_URL` | SQLite path |
| `BACKUP_DIR` / `BACKUP_KEEP` | Encrypted backup directory and number of copies to keep (default 5) |

The model endpoint can be switched between DeepSeek / Ollama / LM Studio / a custom endpoint on the settings page, with a connectivity test; with no API key, import, review, search, and Q&A are all available offline.

Real keys belong only in `.env`; never commit them to Git.

## Backup and restore

Settings page → "Open backup and restore":

- Encrypted export: scrypt-derived key + HMAC-SHA256 stream encryption + encrypt-then-MAC; the passphrase is never stored on disk and plaintext is never written to a file.
- Retention policy: keep only the latest `BACKUP_KEEP` copies (default 5).
- Restore: first upload the backup, decrypt it, and preview "backup row counts vs. current row counts"; only after you confirm does it overwrite everything.

## Retrieval quality baseline

```bash
uv run python scripts/quality_baseline.py          # offline evaluation of the golden set (does not touch real data)
uv run python scripts/quality_baseline.py --attach # read-only evaluation against the real database
```

It outputs Recall@k, evidence accuracy, wrong-subject rate, and refusal accuracy, and compares local hashing with the configured embedding endpoint. A candidate is only recommended for switching if it beats the baseline by more than 5%.

### Hard scope gate (`app/services/scope.py`)

The "assistant that refuses" does not leave the decision to the model. Every question first goes through a local, deterministic judgment; out-of-scope questions (general knowledge, writing code, etc.) are refused directly, with an explanation that this store only records what happened between you and the other person, and the model is not called.

Order of judgment: greetings and short follow-ups pass → if a subject name / profile vocabulary / knowledge-base hit appears, it passes → clear out-of-scope signals are refused → anything unrecognized is refused with a suggested follow-up. As a result, "help me write a message to her" is still in scope. `tests/test_scope_gate.py` asserts that the configured-model and offline paths classify the golden set identically.

## Read-only MCP

```bash
uv run python scripts/mcp_server.py   # stdio, only two read-only tools
```

- `persona_search`: search confirmed memories (candidates are not included in results)
- `persona_profile`: return the refined confirmed profile

There are no write tools at all, so an external agent cannot modify your data.

## Tests

```bash
uv run pytest -q
```

## Deleting data

- Single import: "Delete this import" on the import detail page.
- All data: "Clear all data" on the settings page, or just delete `data/persona_lab.db`.
- Before restoring, export an encrypted backup first; the most recent copies are kept in `data/backups`.

## Architecture summary

```
JSON/JSONL → import normalization → raw text in SQLite
                ↓
           local redaction & segmentation → (optional) cloud structured extraction → candidate memories
                ↓
           human review → confirmed memories (FTS5 + local embedding)
                ↓
           Q&A / situational tendency (experimental; refuses when evidence is insufficient)
```

## Roadmap (build it after it's validated)

The "Usage rhythm" section on the settings page shows the current status of these gates:

- Cross-device sync: requires 4 consecutive weeks of activity (we won't build it until that's met)
- Collaboration and subscriptions: requires evidence from paid interviews (at least 5 users willing to pay)
- A dedicated vector index: evaluate only once real scale reaches 200+ confirmed memories and latency problems appear
- Read-only MCP: already provided (`scripts/mcp_server.py`), still no write access of any kind
- WeFlow session JSON import: already supported (`app/importers/weflow.py`); incremental sync is still pending

## Privacy notice

Please make sure both people in the conversation have explicitly agreed to the import and AI analysis. Do not expose your raw chat store to the public internet. Tendency hints are not mind-reading, and must not be used to manipulate others.
