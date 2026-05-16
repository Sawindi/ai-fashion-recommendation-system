"""
Outfit generation engine.

1) Generate best outfits for an event
2) Complete an outfit given one item

Uses:
wardrobe_manager database
scoring.py
events_config.py
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional

import numpy as np

from src.scoring import score_outfit
from src.wardrobe_manager import _load_db

# Category groups

TOP_CATS = {"top", "t-shirt", "shirt", "knitwear", "cardigan", "sweatshirt", "vest"}
BOTTOM_CATS = {"trousers", "jeans", "skirt", "shorts"}
OUTERWEAR_CATS = {"blazer", "jacket", "coat"}
DRESS_CATS = {"dress"}

# Helper functions
def _is_top(cat: str) -> bool:
    return cat in TOP_CATS


def _is_bottom(cat: str) -> bool:
    return cat in BOTTOM_CATS


def _is_outerwear(cat: str) -> bool:
    return cat in OUTERWEAR_CATS


def _is_dress(cat: str) -> bool:
    return cat in DRESS_CATS

# Load wardrobe items

def load_items() -> List[Dict]:
    db = _load_db()

    items = []
    for rec in db:   # Loop through each record
        # Load embedding from .npy 
        try:
            emb = np.load(rec["embedding_path"])
        except FileNotFoundError:
            print(f" Missing embedding file, skipping: {rec.get('embedding_path')}")
            continue
        except Exception as e:
            print(f" Could not load embedding, skipping: {rec.get('embedding_path')} | {e}")
            continue

        # Build item dictionary
        items.append({
            "item_id": rec["item_id"],
            "zara_category": rec["zara_category"],
            "yolo_class": rec["yolo_class"],
            "yolo_confidence": rec.get("yolo_confidence", 0.0),
            "color": rec.get("color", ""),
            "embedding": emb,
            "crop_path": rec["crop_path"],
        })

    return items


# Generate outfits
def generate_outfits(event_key: str, top_k: int = 5) -> List[Dict]:
    items = load_items()

    tops = [x for x in items if _is_top(x["zara_category"])]
    bottoms = [x for x in items if _is_bottom(x["zara_category"])]
    dresses = [x for x in items if _is_dress(x["zara_category"])]
    outerwears = [x for x in items if _is_outerwear(x["zara_category"])]

    results = []

    # Dress-based outfits
    for dress in dresses:
        # without outerwear
        breakdown = score_outfit(
            event_key,
            top=None,
            bottom=None,
            dress=dress,
            outerwear=None,
        )

        results.append({
            "items": [dress],
            "score": breakdown,
        })

        # with outerwear
        for outer in outerwears:
            breakdown = score_outfit(
                event_key,
                top=None,
                bottom=None,
                dress=dress,
                outerwear=outer,
            )

            results.append({
                "items": [dress, outer],
                "score": breakdown,
            })

    # Top + Bottom outfits
    for top, bottom in product(tops, bottoms):

        # without outerwear
        breakdown = score_outfit(
            event_key,
            top=top,
            bottom=bottom,
            dress=None,
            outerwear=None,
        )

        results.append({
            "items": [top, bottom],
            "score": breakdown,
        })

        # with outerwear
        for outer in outerwears:
            breakdown = score_outfit(
                event_key,
                top=top,
                bottom=bottom,
                dress=None,
                outerwear=outer,
            )

            results.append({
                "items": [top, bottom, outer],
                "score": breakdown,
            })

    # Sort results by score
    results.sort(key=lambda x: x["score"].total, reverse=True)

    return results[:top_k]

# Complete an outfit
def complete_outfit(
    base_item_id: str,
    event_key: str,
    top_k: int = 5,
) -> List[Dict]:

    items = load_items()
    base = next((x for x in items if x["item_id"] == base_item_id), None)

    if base is None:
        raise ValueError(f"Item not found: {base_item_id}")

    tops = [x for x in items if _is_top(x["zara_category"]) and x["item_id"] != base_item_id]
    bottoms = [x for x in items if _is_bottom(x["zara_category"]) and x["item_id"] != base_item_id]
    outerwears = [x for x in items if _is_outerwear(x["zara_category"]) and x["item_id"] != base_item_id]

    results = []

    # If base is dress
    if _is_dress(base["zara_category"]):
        # Add outerwear options
        breakdown = score_outfit(event_key, None, None, base, None)
        results.append({"items": [base], "score": breakdown})

        for outer in outerwears:
            breakdown = score_outfit(event_key, None, None, base, outer)
            results.append({"items": [base, outer], "score": breakdown})

    # If base is top
    elif _is_top(base["zara_category"]):
        for bottom in bottoms:
            breakdown = score_outfit(event_key, base, bottom, None, None)
            results.append({"items": [base, bottom], "score": breakdown})

            for outer in outerwears:
                breakdown = score_outfit(event_key, base, bottom, None, outer)
                results.append({"items": [base, bottom, outer], "score": breakdown})

    # If base is bottom
    elif _is_bottom(base["zara_category"]):
        for top in tops:
            breakdown = score_outfit(event_key, top, base, None, None)
            results.append({"items": [top, base], "score": breakdown})

            for outer in outerwears:
                breakdown = score_outfit(event_key, top, base, None, outer)
                results.append({"items": [top, base, outer], "score": breakdown})

    results.sort(key=lambda x: x["score"].total, reverse=True)

    return results[:top_k]
