"""Shared backend constants (avoid duplicating large literals across modules)."""

from __future__ import annotations

# Distinct format slugs aligned with Scryfall legality keys; used when DB is missing or empty.
FALLBACK_LEGALITY_FORMATS: tuple[str, ...] = (
    "alchemy",
    "brawl",
    "commander",
    "duel",
    "explorer",
    "future",
    "gladiator",
    "historic",
    "historicbrawl",
    "legacy",
    "modern",
    "oathbreaker",
    "oldschool",
    "pauper",
    "paupercommander",
    "penny",
    "pioneer",
    "premodern",
    "predh",
    "standard",
    "standardbrawl",
    "timeless",
    "vintage",
)
