from __future__ import annotations

from typing import Any
from datetime import datetime

from agents import Agent, Runner
from pydantic import BaseModel, ConfigDict
from backend.agent.common_prompts import MTG_GLOSSARY
from backend.agent.tools.card_search_tool import search_mtg_cards
from backend.utils.agents_sdk import openrouter_async_agents_model

_SEARCH_AGENT_MODEL = "gpt-5.4-mini"

def build_system_prompt(current_date: str, include_reasoning: bool = False) -> str:
    _SYSTEM_PROMPT = """
    You are an "Magic: The Gathering" (later referred to as MTG) card search assistant.

    ## Role and Responsibilities
    Prioritize game-mechanics fields when understanding user's search intent.

    ## Tool Usage
    Always call the search_mtg_cards tool before returning output (you may call it more than once; see "Result count" below).
    If the user query is not in English, first translate/rephrase it to English for tool input.

    ## Relevance filtering (mandatory)
    Hybrid search can return false positives: the card text may share keywords with the request but express the opposite or an unrelated mechanic.
    Before you emit your final answer, drop any card that does not actually satisfy the user's intent when you read name, type_line, and oracle_text.
    Examples (not exhaustive):
    - User wants a spell that *deals* damage → exclude cards that only *prevent* damage, *replace* damage, or *gain life* when damage would be dealt, unless the user asked for those.
    - User wants removal/destruction → exclude cards that mainly grant indestructible or prevention if that contradicts the request.
    - User wants creatures with a combat keyword → exclude cards that only mention that word in flavor or unrelated rules text.
    When unsure, prefer excluding the card rather than padding the list with weak matches.

    ## Result count
    The user specifies a target number of cards `limit` (passed into the tool as `limit`).
    1) After relevance filtering, if you have **fewer than `limit`** cards, call `search_mtg_cards` again with an improved `query`
       (clearer mechanics, synonyms, split or combined phrases, or different keywords that still match intent). You may also toggle `use_prefilter` if the first pass was too narrow or too noisy.
       Repeat only as needed—typically at most two follow-up calls—until you reach `limit` strong matches or further searches are unlikely to help. You may request extra rows from the tool (e.g. somewhat above `limit`) to allow for filtering, but your **final** JSON must not exceed `limit` cards.
    2) If you have **more than `limit`** cards after filtering, rank them by how well they match the user's query (mechanics, type, colors). **Drop the weakest** matches until exactly `limit` remain.
    3) Your returned JSON array length must be **at most** `limit`. If you cannot find enough truly matching cards after reasonable tool use, return fewer rather than filling with mismatches.

    """

    _REASONING_INSTRUCTIONS = """
    ## Reasoning (required for this request)
    In addition to every field returned by the tool for each card, include a string field `reasoning` for every card.
    Each `reasoning` must be 1-3 sentences explaining why that card was selected
    (mechanics, type line, colors, how it fits the user's query and chosen formats, and whether FTS keywords vs semantic search likely surfaced it).
    """

    _ADDITIONAL_INSTRUCTIONS = f"""
    ## Additional informations
    - Today's date is {current_date}.
    """

    if include_reasoning:
        return _SYSTEM_PROMPT + MTG_GLOSSARY + _REASONING_INSTRUCTIONS + _ADDITIONAL_INSTRUCTIONS
    else:
        return _SYSTEM_PROMPT + MTG_GLOSSARY + _ADDITIONAL_INSTRUCTIONS


def build_user_prompt(
    query: str,
    formats: list[str],
    limit: int,
    include_reasoning: bool = False,
) -> str:
    return (
        f"Search for cards using query={query!r}, formats={formats!r}, limit={limit}. "
        "Drop irrelevant or opposite-meaning hits, refine with extra tool calls if you are short of strong matches, "
        f"and cap the final list at {limit} cards (or fewer if matches run out). "
    )


class AgentCardOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    scryfall_url: str = ""
    oracle_text: str = ""
    mana_cost: str = ""
    power: str = ""
    toughness: str = ""
    power_toughness: str = ""
    reasoning: str = ""

class AgentSearchOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cards: list[AgentCardOutput]

async def run_search(
    query: str,
    formats: list[str],
    limit: int,
    include_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """Execute the agent once and return card dicts shaped for the HTTP API."""
    prompt = build_user_prompt(query, formats, limit, include_reasoning)
    instructions = build_system_prompt(datetime.now().strftime("%Y-%m-%d"), include_reasoning)
    agent = Agent(
        name="MTG Card Search Agent",
        model=openrouter_async_agents_model(_SEARCH_AGENT_MODEL),
        instructions=instructions,
        tools=[search_mtg_cards],
        output_type=AgentSearchOutput,
    )
    result = await Runner.run(agent, prompt)
    cards: list[dict[str, str]] = []
    for item in result.final_output.cards:
        power = item.power.strip()
        toughness = item.toughness.strip()
        pt = item.power_toughness.strip()
        if not pt and power and toughness:
            pt = f"{power}/{toughness}"
        cards.append(
            {
                "name": item.name,
                "scryfall_url": item.scryfall_url,
                "oracle_text": item.oracle_text,
                "mana_cost": item.mana_cost,
                "power_toughness": pt,
                "reasoning": item.reasoning,
            }
        )
    return cards[:limit]
