from __future__ import annotations

from typing import Any


def build_card_embedding_text(card: dict[str, Any]) -> str:
    parts: list[str] = []

    name = str(card.get("name") or "").strip()
    if name:
        parts.append(f"Name: {name}")

    type_line = str(card.get("type_line") or "").strip()
    if type_line:
        parts.append(f"Type: {type_line}")

    mana_cost = str(card.get("mana_cost") or "").strip()
    if mana_cost:
        parts.append(f"Mana: {mana_cost}")

    oracle_text = str(card.get("oracle_text") or "").strip()
    if oracle_text:
        parts.append(f"Oracle: {oracle_text}")

    keywords = [str(v).strip() for v in (card.get("keywords") or []) if str(v).strip()]
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    colors = [str(v).strip() for v in (card.get("colors") or []) if str(v).strip()]
    if colors:
        parts.append(f"Colors: {', '.join(colors)}")

    color_identity = [
        str(v).strip() for v in (card.get("color_identity") or []) if str(v).strip()
    ]
    if color_identity:
        parts.append(f"Color identity: {', '.join(color_identity)}")

    power = card.get("power")
    toughness = card.get("toughness")
    if power is not None and str(power).strip():
        parts.append(f"Power: {power}")
    if toughness is not None and str(toughness).strip():
        parts.append(f"Toughness: {toughness}")
        try:
            t = int(str(toughness).strip())
            size_keywords: list[str] = []
            if t == 1:
                size_keywords.append("squishy")
            elif t == 2:
                size_keywords.append("small")
            elif t in (3, 4):
                size_keywords.append("medium")
            elif t in (5, 6):
                size_keywords.append("large")
            if t >= 6:
                size_keywords.append("huge")
            if size_keywords:
                parts.append(f"Creature size keywords: {', '.join(size_keywords)}")
        except ValueError:
            pass

    loyalty = card.get("loyalty")
    if loyalty is not None and str(loyalty).strip():
        parts.append(f"Loyalty: {loyalty}")

    produced_mana = [
        str(v).strip() for v in (card.get("produced_mana") or []) if str(v).strip()
    ]
    if produced_mana:
        parts.append(f"Produces mana: {', '.join(produced_mana)}")

    layout = str(card.get("layout") or "").strip()
    if layout:
        parts.append(f"Layout: {layout}")

    rarity = str(card.get("rarity") or "").strip()
    if rarity:
        parts.append(f"Rarity: {rarity}")

    all_parts = card.get("all_parts") or []
    if isinstance(all_parts, list):
        all_part_names = []
        for item in all_parts:
            if isinstance(item, dict):
                n = str(item.get("name") or "").strip()
                if n:
                    all_part_names.append(n)
        if all_part_names:
            parts.append(f"All parts: {', '.join(all_part_names)}")

    faces = card.get("card_faces") or []
    if isinstance(faces, list):
        for i, face in enumerate(faces, start=1):
            if not isinstance(face, dict):
                continue
            face_name = str(face.get("name") or "").strip()
            if face_name:
                parts.append(f"Face {i} Name: {face_name}")
            face_type = str(face.get("type_line") or "").strip()
            if face_type:
                parts.append(f"Face {i} Type: {face_type}")
            face_oracle = str(face.get("oracle_text") or "").strip()
            if face_oracle:
                parts.append(f"Face {i} Oracle: {face_oracle}")
            face_power = face.get("power")
            face_toughness = face.get("toughness")
            if face_power is not None and str(face_power).strip():
                parts.append(f"Face {i} Power: {face_power}")
            if face_toughness is not None and str(face_toughness).strip():
                parts.append(f"Face {i} Toughness: {face_toughness}")
            face_loyalty = face.get("loyalty")
            if face_loyalty is not None and str(face_loyalty).strip():
                parts.append(f"Face {i} Loyalty: {face_loyalty}")

    out = "\n".join(parts).strip()
    return out if out else (name or "(empty)")
