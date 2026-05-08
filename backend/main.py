from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


def _load_backend_env() -> None:
    # Priority: explicit backend/.env, then repo-root .env, then cwd discovery.
    backend_env = Path(__file__).resolve().parent / ".env"
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    if backend_env.is_file():
        load_dotenv(dotenv_path=backend_env, override=False)
    if repo_env.is_file():
        load_dotenv(dotenv_path=repo_env, override=False)
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(dotenv_path=discovered, override=False)


_load_backend_env()

from backend.agent.search_agent import run_search  # noqa: E402
from backend.utils.constants import FALLBACK_LEGALITY_FORMATS  # noqa: E402
from backend.utils.db_paths import get_database_path  # noqa: E402
from backend.utils.profiling import emit_event, start_request_event  # noqa: E402


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Refusing to start backend without OpenRouter key."
        )
    yield


DATABASE_PATH = get_database_path()

app = FastAPI(title="MTG Agentic Search API", lifespan=_lifespan)


def _formats_from_database() -> list[str]:
    if not DATABASE_PATH.is_file():
        return list(FALLBACK_LEGALITY_FORMATS)
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            rows = conn.execute(
                "SELECT DISTINCT format FROM card_legalities ORDER BY format COLLATE NOCASE"
            ).fetchall()
    except sqlite3.Error:
        return list(FALLBACK_LEGALITY_FORMATS)
    if not rows:
        return list(FALLBACK_LEGALITY_FORMATS)
    return [r[0] for r in rows]


class SearchRequest(BaseModel):
    query: str = Field(default="")
    formats: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    include_reasoning: bool = Field(default=False)


class CardItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    scryfall_url: str
    oracle_text: str = ""
    mana_cost: str = ""
    power_toughness: str = ""
    reasoning: str | None = None


class SearchResponse(BaseModel):
    cards: list[CardItem]


@app.post("/search", response_model=SearchResponse)
async def search_cards(req: SearchRequest) -> SearchResponse:
    start_request_event()
    raw_cards = await run_search(
        query=req.query,
        formats=req.formats,
        limit=req.limit,
        include_reasoning=req.include_reasoning,
    )
    emit_event("agent responded to front end")
    return SearchResponse(cards=[CardItem.model_validate(c) for c in raw_cards])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/formats")
async def list_formats() -> dict[str, list[str]]:
    """Distinct format slugs from ``card_legalities`` (matches Scryfall JSON keys)."""
    return {"formats": _formats_from_database()}
