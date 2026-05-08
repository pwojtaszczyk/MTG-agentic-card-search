from __future__ import annotations

MTG_GLOSSARY = """
## MTG glossary
- {W} : White mana symbol
- {U} : Blue mana symbol
- {B} : Black mana symbol
- {R} : Red mana symbol
- {G} : Green mana symbol
- {1} : One colorless mana symbol
- {2} : Two colorless mana symbols
- {X} : Variable colorless mana symbol
- X/Y: Power/Toughness (it's not a fraction)
- */*: Power/Toughness which is undefined, the values depend on the card text rules.


## Fields returned by search tools and database queries which are relevant for game mechanics:
name, mana_cost, cmc, type_line, oracle_text, colors, color_identity, keywords,
produced_mana, power, toughness, loyalty, defense

"""
