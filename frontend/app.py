from __future__ import annotations

import html
import os
import re
from typing import Any

import gradio as gr
import httpx
from dotenv import load_dotenv

# When the API is unavailable, keep this aligned with backend.utils.constants.FALLBACK_LEGALITY_FORMATS.
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

SEARCH_ICON = "\U0001f50d"

_CUSTOM_CSS = """
.search-row {
    align-items: stretch !important;
    flex-direction: row !important;
}
.search-row .search-icon-btn {
    min-width: 3.2rem !important;
    max-width: 3.2rem !important;
    flex-shrink: 0 !important;
    align-self: stretch !important;
    padding: 0 !important;
    font-size: 1.35rem !important;
    line-height: 1 !important;
    border-radius: 4px !important;
}
.search-row .textbox,
.search-row textarea,
.search-row input[type="text"] {
    min-height: 3rem !important;
}
.mtg-results-wrap { overflow-x: auto; width: 100%; }
.mtg-results {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.95rem;
}
.mtg-results th, .mtg-results td {
    border: 1px solid var(--border-color-primary, #ddd);
    padding: 0.5rem 0.6rem;
    vertical-align: top;
}
.mtg-results th { text-align: left; background: var(--table-even-bg, rgba(0,0,0,0.04)); }
.mtg-results tr:nth-child(even) { background: var(--table-even-bg, rgba(0,0,0,0.02)); }
.mtg-name { min-width: 8rem; }
/* Mana cost: grow with symbol count; horizontal scroll on .mtg-results-wrap if needed. */
.mtg-results th.mtg-mana-col,
.mtg-results td.mtg-mana {
    white-space: nowrap;
    vertical-align: middle;
    width: 1%;
    min-width: max-content;
    /* Extra horizontal room so SVG symbols do not sit flush against borders. */
    padding-left: 0.65rem;
    padding-right: 1rem;
}
.mtg-otext { max-width: 28rem; white-space: pre-wrap; }
.mtg-empty { color: var(--body-text-color-subdued, #666); }
"""

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = os.getenv("API_URL", f"{API_BASE_URL}/search")
FORMATS_URL = f"{API_BASE_URL}/formats"
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "127.0.0.1")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "7860"))

# Scryfall-hosted SVG mana and cost symbols (e.g. W.svg, WB.svg, 2W.svg).
SCRYFALL_CARD_SYMBOLS_BASE = "https://svgs.scryfall.io/card-symbols"
_SCRYFALL_SYMBOL_CODE = re.compile(r"^[0-9A-Z]+$")


def _mana_inner_to_scryfall_code(inner: str) -> str:
    """Map `{inner}` text to Scryfall card-symbol filename stem (no `.svg`)."""
    t = inner.strip()
    if not t:
        return ""
    if "/" in t:
        parts = [p.strip() for p in t.split("/") if p.strip()]
        return "".join(p.upper() for p in parts)
    return t.upper()


def _scryfall_mana_symbol_url(inner: str) -> str | None:
    code = _mana_inner_to_scryfall_code(inner)
    if not code or not _SCRYFALL_SYMBOL_CODE.fullmatch(code):
        return None
    return f"{SCRYFALL_CARD_SYMBOLS_BASE}/{code}.svg"


def make_card_name_cell(url: str = "", name: str = "", reasoning: str = "") -> str:
    """Link or span with optional native tooltip (title=)."""
    safe_name = html.escape(name, quote=True)
    r = reasoning.strip()
    title_attr = f' title="{html.escape(r, quote=True)}"' if r else ""

    if url.strip():
        safe_url = html.escape(url.strip(), quote=True)
        return (
            f"<a href='{safe_url}' target='_blank' rel='noopener'{title_attr}>"
            f"{safe_name}</a>"
        )
    if r:
        return f"<span{title_attr}>{safe_name}</span>"
    return safe_name


def render_mana_cost(mana_cost: str = "") -> str:
    text = mana_cost.strip()
    if not text:
        return ""
    tokens = re.findall(r"\{([^}]+)\}", text)
    if not tokens:
        return html.escape(text)
    chunks: list[str] = []
    for token in tokens:
        t_disp = token.strip()
        t_upper = t_disp.upper()
        url = _scryfall_mana_symbol_url(token)
        if url:
            safe_url = html.escape(url, quote=True)
            safe_alt = html.escape(t_disp, quote=True)
            chunks.append(
                "<img "
                f"src='{safe_url}' "
                f"alt='{{{safe_alt}}}' "
                f"title='{{{safe_alt}}}' "
                "style='height:1.1em;width:auto;"
                "vertical-align:middle;"
                "display:inline-block;"
                "background:transparent;"
                "margin-right:2px;"
                "'/>"
            )
        elif t_upper.isdigit() or t_upper == "X":
            chunks.append(f"<span style='margin-right:4px;'>{html.escape(t_upper)}</span>")
        else:
            chunks.append(
                f"<span style='margin-right:4px;'>{html.escape('{' + t_disp + '}')}</span>"
            )
    return (
        "<span style='display:inline-flex;align-items:center;gap:2px;white-space:nowrap;'>"
        + "".join(chunks)
        + "</span>"
    )


def load_format_choices():
    try:
        response = httpx.get(FORMATS_URL, timeout=15.0)
        response.raise_for_status()
        choices = response.json().get("formats") or []
    except Exception:
        choices = []
    if not choices:
        choices = list(FALLBACK_LEGALITY_FORMATS)
    default = ["standard"] if "standard" in choices else ([choices[0]] if choices else [])
    return gr.update(choices=choices, value=default), choices


def select_all_formats(choices: list[str]):
    return gr.update(value=list(choices))


def deselect_all_formats():
    return gr.update(value=[])


def _rows_to_html(rows: list[list[Any]]) -> str:
    """Build results table; name/mana cells contain trusted HTML from our builders."""
    if not rows:
        return '<p class="mtg-empty">No cards returned.</p>'
    parts: list[str] = [
        '<div class="mtg-results-wrap"><table class="mtg-results"><thead><tr>',
        '<th class="mtg-name">Card Name</th>'
        '<th class="mtg-mana-col">Mana cost</th>'
        "<th>P/T</th>"
        '<th class="mtg-otext">Card text</th>',
        "</tr></thead><tbody>",
    ]
    for row in rows:
        name_h, mana_h, pt, otext = row[0], row[1], row[2], row[3]
        parts.append("<tr>")
        parts.append(f'<td class="mtg-name">{name_h}</td>')
        parts.append(f'<td class="mtg-mana">{mana_h}</td>')
        parts.append(f"<td>{html.escape(str(pt))}</td>")
        parts.append(f'<td class="mtg-otext">{html.escape(str(otext))}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def search_cards(
    user_message: str,
    include_reasoning: bool,
    formats: list[str],
    limit: int,
):
    payload = {
        "query": user_message,
        "formats": formats,
        "limit": limit,
        "include_reasoning": include_reasoning,
    }
    response = httpx.post(API_URL, json=payload, timeout=120.0)
    response.raise_for_status()
    cards = response.json().get("cards", [])

    rows = []
    for card in cards:
        reasoning = (card.get("reasoning") or "") if include_reasoning else ""
        rows.append(
            [
                make_card_name_cell(
                    card.get("scryfall_url", "") or "",
                    card.get("name", "") or "",
                    reasoning,
                ),
                render_mana_cost(card.get("mana_cost") or ""),
                card.get("power_toughness", "") or "",
                card.get("oracle_text", "") or "",
            ]
        )
    return _rows_to_html(rows)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MTG Agentic Search", css=_CUSTOM_CSS) as demo:
        gr.Markdown("## MTG Agentic Card Search")
        format_choices_state = gr.State(list(FALLBACK_LEGALITY_FORMATS))

        with gr.Row():
            include_reasoning = gr.Checkbox(
                label="Include reasoning",
                value=False,
            )

        with gr.Row(elem_classes=["search-row"]):
            search_btn = gr.Button(
                SEARCH_ICON,
                elem_classes=["search-icon-btn"],
                scale=0,
            )
            query = gr.Textbox(
                show_label=False,
                placeholder="Find aggressive red creatures",
                lines=1,
                max_lines=8,
                scale=1,
            )

        with gr.Row():
            with gr.Column(scale=0, min_width=120):
                select_all_btn = gr.Button("Select all", size="sm")
                deselect_all_btn = gr.Button("Deselect all", size="sm")
            formats = gr.CheckboxGroup(
                label="Formats",
                choices=list(FALLBACK_LEGALITY_FORMATS),
                value=["standard"],
                scale=1,
            )

        limit = gr.Slider(label="Result limit", minimum=1, maximum=50, step=1, value=10)
        output = gr.HTML(
            value='<p class="mtg-empty">Run a search to see results. Press Enter to search, Shift+Enter for a new line, or click the search button.</p>'
        )

        _search_inputs = [query, include_reasoning, formats, limit]

        search_btn.click(fn=search_cards, inputs=_search_inputs, outputs=output)
        query.submit(fn=search_cards, inputs=_search_inputs, outputs=output)

        demo.load(fn=load_format_choices, outputs=[formats, format_choices_state])
        select_all_btn.click(
            fn=select_all_formats,
            inputs=[format_choices_state],
            outputs=formats,
        )
        deselect_all_btn.click(fn=deselect_all_formats, outputs=formats)

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(server_name=FRONTEND_HOST, server_port=FRONTEND_PORT)


if __name__ == "__main__":
    main()
