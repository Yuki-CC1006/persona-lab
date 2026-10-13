"""Single source of truth for the product's name and on-disk identifiers.

The rename from the previous internal name touched the package, the CLI, the
database filename, the backup format, the MCP tools and every page title. They
live here so a future rename is one edit instead of a scavenger hunt.

``LEGACY_*`` describe what the project was called before; they exist only so the
one-time database migration can find an existing file.
"""

from __future__ import annotations

PRODUCT_NAME = "Persona Lab"
PRODUCT_SLUG = "persona-lab"

DB_FILENAME = "persona_lab.db"

# Previous internal codename, kept only so the one-time migration can find an
# existing database file. Never use this for new artifacts (paths, formats,
# tool names) — it is a legacy filename, not a brand.
LEGACY_DB_FILENAME = "rxy_wiki.db"
