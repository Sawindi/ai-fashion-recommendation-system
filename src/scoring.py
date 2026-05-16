# src/scoring.py
"""
Full outfit scoring (Hybrid: fashion rules + event rules + embeddings) with future ML hook.

Supported outfits:
- Top + Bottom (+ optional Outerwear)
- Dress (+ optional Outerwear)

Item dict expected by this scorer:
{
  "item_id": str,
  "zara_category": str,      
  "yolo_class": str,
  "yolo_confidence": float,
  "color": Optional[str],
  "embedding": np.ndarray,
}

Used by:
- outfit_generator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.events_config import EVENTS, EventProfile


# Category groups
TOP_CATS = {"top", "t-shirt", "shirt", "knitwear", "cardigan", "sweatshirt", "vest"}
BOTTOM_CATS = {"trousers", "jeans", "skirt", "shorts"}
OUTERWEAR_CATS = {"blazer", "jacket", "coat"}
DRESS_CATS = {"dress"}

# Color theory
NEUTRALS = {"black", "white", "grey"}

COMPLEMENTARY = {
    "blue": "orange", "orange": "blue",
    "red": "green", "green": "red",
    "yellow": "purple", "purple": "yellow",
}

ANALOGOUS = {
    "blue": {"green", "purple"},
    "green": {"blue", "yellow"},
    "yellow": {"green", "orange"},
    "orange": {"yellow", "red"},
    "red": {"orange", "purple"},
    "purple": {"red", "blue"},
}


# Formality
FORMALITY_BY_ZARA = {
    # 1 = casual, 5 = very formal
    "t-shirt": 1,
    "top": 2,
    "sweatshirt": 1,
    "knitwear": 2,
    "cardigan": 2,
    "shirt": 4,
    "vest": 2,
    "jeans": 2,
    "trousers": 4,
    "skirt": 3,
    "shorts": 1,
    "dress": 4,
    "blazer": 5,
    "jacket": 3,
    "coat": 4,
    "co-ord set": 3,
}

# YOLO classes
YOLO_GROUPS = {
    "short_sleeve_top": {"top", "t-shirt"},
    "long_sleeve_top": {"shirt", "top", "knitwear", "cardigan"},
    "short_sleeve_outwear": {"jacket"},
    "long_sleeve_outwear": {"blazer", "coat", "jacket"},
    "vest": {"vest", "top"},
    "sling": {"top"},
    "shorts": {"shorts"},
    "trousers": {"trousers", "jeans"},
    "skirt": {"skirt"},
    "short_sleeve_dress": {"dress"},
    "long_sleeve_dress": {"dress"},
    "vest_dress": {"dress"},
    "sling_dress": {"dress"},
}

# ScoreBreakDown dataclass
@dataclass
class ScoreBreakdown:
    total: float
    rating_stars: int
    category_score: float
    event_score: float
    formality_score: float
    color_score: float
    style_score: float
    yolo_consistency_score: float
    learned_score: Optional[float]
    explanation: List[str]

# Helper functions
def _clamp01(x: float) -> float:   # Keep values between 0 and 1 (All scores are designed to stay in normalized range)
    return max(0.0, min(1.0, float(x)))

def _z_in(z: str, group: set[str]) -> bool:   # Checks whether Zara category belongs to a group
    return (z or "").strip().lower() in group

def _cosine(a: np.ndarray, b: np.ndarray) -> float:   # Computes cosine similarity between two embeddings
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# 1) Category validity score (structure correctness)

def category_structure_score(
    top: Optional[Dict],
    bottom: Optional[Dict],
    dress: Optional[Dict],
    outerwear: Optional[Dict],
) -> Tuple[float, List[str]]:
    notes: List[str] = []

    # dress route
    if dress is not None:
        score = 1.0
        if not _z_in(dress.get("zara_category", ""), DRESS_CATS):
            score -= 0.35
            notes.append(f"Dress item category unusual: {dress.get('zara_category')}")
        if outerwear is not None and not _z_in(outerwear.get("zara_category", ""), OUTERWEAR_CATS):
            score -= 0.25
            notes.append(f"Outerwear category unusual: {outerwear.get('zara_category')}")
        return _clamp01(score), (notes or ["Dress outfit structure looks valid."])

    # top + bottom route
    if top is None or bottom is None:
        return 0.0, ["Missing top or bottom → cannot form outfit."]

    score = 1.0
    if not _z_in(top.get("zara_category", ""), TOP_CATS):
        score -= 0.35
        notes.append(f"Top category unusual: {top.get('zara_category')}")
    if not _z_in(bottom.get("zara_category", ""), BOTTOM_CATS):
        score -= 0.35
        notes.append(f"Bottom category unusual: {bottom.get('zara_category')}")
    if outerwear is not None and not _z_in(outerwear.get("zara_category", ""), OUTERWEAR_CATS):
        score -= 0.25
        notes.append(f"Outerwear category unusual: {outerwear.get('zara_category')}")

    return _clamp01(score), (notes or ["Top + bottom outfit structure looks valid."])


# 2) YOLO - Zara consistency (quality control)
def yolo_zara_consistency_score(items: List[Dict]) -> Tuple[float, List[str]]:
    notes: List[str] = []
    if not items:
        return 0.0, ["No items provided."]

    score = 1.0
    for it in items:
        y = (it.get("yolo_class") or "").strip()
        z = (it.get("zara_category") or "").strip().lower()
        conf = float(it.get("yolo_confidence") or 0.0)

        # If YOLO is low-confidence, do not penalize much
        if conf < 0.35:
            notes.append(f"YOLO low confidence for {it.get('item_id','?')} → not penalizing.")
            continue

        allowed = YOLO_GROUPS.get(y)
        if allowed and z not in allowed:
            score -= 0.10
            notes.append(f"YOLO({y}) vs Zara({z}) mismatch (conf={conf:.2f}).")

    return _clamp01(score), (notes or ["YOLO and Zara labels look consistent."])


# 3) Event suitability (required / preferred / forbidden + small colour bonus)

def event_suitability_score(event: EventProfile, items: List[Dict]) -> Tuple[float, List[str]]:
    notes: List[str] = []
    zara_cats = [it.get("zara_category", "").strip().lower() for it in items]
    colors = [it.get("color", "").strip().lower() for it in items if it.get("color")]

    # Smart category matching (group-aware)
    def _has(req: str) -> bool:
        """
        Allows event templates to use broad tokens like..
          - "top"        -> any category in TOP_CATS
          - "bottom"     -> any category in BOTTOM_CATS
          - "outerwear"  -> any category in OUTERWEAR_CATS
        Otherwise matches exact category strings.
        """
        r = (req or "").strip().lower()

        if r == "top":
            return any(c in TOP_CATS for c in zara_cats)
        if r == "bottom":
            return any(c in BOTTOM_CATS for c in zara_cats)
        if r == "outerwear":
            return any(c in OUTERWEAR_CATS for c in zara_cats)

        return r in zara_cats


    # Outfit must satisfy at least one template
    templates = getattr(event, "required_any", [])

    if templates:
        def template_satisfied(template: List[str]) -> bool:
            return all(_has(req) for req in template)

        ok = any(template_satisfied(t) for t in templates)

        if ok:
            req_penalty = 0.0
            notes.append("Outfit satisfies an allowed event template.")
        else:
            req_penalty = 0.25
            notes.append(f"Outfit does not satisfy any allowed template for {event.name}.")
            notes.append(f"Allowed templates: {templates}")
    else:
        req_penalty = 0.0
        notes.append("No required templates defined, skipping template check.")

    # Forbidden categories (exact match)
    forbidden_hit = [f for f in event.forbidden_categories if (f or "").strip().lower() in zara_cats]
    forb_penalty = 0.18 * len(forbidden_hit)
    if forbidden_hit:
        notes.append(f"Forbidden present: {forbidden_hit}")

    # Preferred categories (soft bonus, exact match)
    preferred_hit = [p for p in event.preferred_categories if (p or "").strip().lower() in zara_cats]
    pref_bonus = 0.07 * len(preferred_hit)
    if preferred_hit:
        notes.append(f"Preferred matched: {preferred_hit}")

    # Preferred colours (small bonus)
    colour_bonus = 0.0
    pref_cols = [(c or "").strip().lower() for c in event.preferred_colours]
    if pref_cols and colors:
        if any(c in pref_cols for c in colors):
            colour_bonus = 0.03
            notes.append("Event preferred colour matched.")

  
    # Dress priority (Date Night)
    dress_bonus = 0.0
    if event.name == "date_night" and "dress" in zara_cats:
        dress_bonus = 0.12
        notes.append("Date night bonus: dress outfit prioritised.")

    # Base score with penalties/bonuses
    score = 0.75 - req_penalty - forb_penalty + pref_bonus + colour_bonus + dress_bonus
    return _clamp01(score), notes


# 4) Formality alignment (fashion theory)
def formality_alignment_score(event: EventProfile, items: List[Dict]) -> Tuple[float, List[str]]:
    notes: List[str] = []
    target = int(event.formality_level)

    vals = []
    for it in items:
        z = (it.get("zara_category") or "").lower()
        vals.append(FORMALITY_BY_ZARA.get(z, 3))  # default medium

    if not vals:
        return 0.5, ["No formality info; baseline used."]

    outfit_formality = float(np.mean(vals))
    diff = abs(outfit_formality - target)

    # diff 0 -> 1.0, diff 1 -> 0.8, diff 2 -> 0.6, diff 3 -> 0.4, diff >=4 -> 0.2
    score = max(0.2, 1.0 - 0.2 * diff)
    notes.append(f"Event formality={target}, outfit formality≈{outfit_formality:.2f} (diff={diff:.2f}).")
    return _clamp01(score), notes


# 5) Color harmony (fashion theory)
def color_harmony_score(items: List[Dict]) -> Tuple[float, List[str]]:
    notes: List[str] = []
    cols = [it.get("color", "").strip().lower() for it in items if it.get("color")]

    if len(cols) <= 1:
        return 0.60, ["Not enough colour info -> baseline used."]

    unique = list(dict.fromkeys(cols))
    non_neutral = [c for c in unique if c not in NEUTRALS]

    # Base score boosted by neutrals (neutrals pair well)
    neutral_count = sum(c in NEUTRALS for c in unique)
    score = 0.45 + 0.10 * min(neutral_count, 2)

    # Monochrome
    if len(set(unique)) == 1:
        score = max(score, 0.88)
        notes.append("Monochrome palette.")

    # Complementary
    if len(non_neutral) >= 2:
        a, b = non_neutral[0], non_neutral[1]
        if COMPLEMENTARY.get(a) == b:
            score = max(score, 0.82)
            notes.append(f"Complementary colours: {a}+{b}")

    # Analogous
    if len(non_neutral) >= 2:
        a, b = non_neutral[0], non_neutral[1]
        if b in ANALOGOUS.get(a, set()):
            score = max(score, 0.78)
            notes.append(f"Analogous colours: {a}+{b}")

    return _clamp01(score), (notes or ["Colour harmony baseline applied."])

# 6) Style compatibility (FashionCLIP embeddings)
def style_compatibility_score(
    top: Optional[Dict],
    bottom: Optional[Dict],
    dress: Optional[Dict],
    outerwear: Optional[Dict],
) -> Tuple[float, List[str]]:
    notes: List[str] = []

    def emb(it: Dict) -> np.ndarray:
        e = it.get("embedding")
        if e is None:
            raise ValueError("Missing embedding in item.")
        return e

    sims: List[float] = []

    if dress is not None:
        if outerwear is None:
            return 0.62, ["Dress-only outfit → baseline style score."]
        s = _cosine(emb(dress), emb(outerwear))
        sims = [s]
        notes.append("Style: dress ↔ outerwear similarity.")
    else:
        if top is None or bottom is None:
            return 0.0, ["Missing top/bottom embeddings."]
        sims.append(_cosine(emb(top), emb(bottom)))
        notes.append("Style: top ↔ bottom similarity.")
        if outerwear is not None:
            sims.append(_cosine(emb(top), emb(outerwear)))
            sims.append(_cosine(emb(bottom), emb(outerwear)))
            notes.append("Style: outerwear similarity included.")

    # map cosine [-1,1] -> [0,1]
    sims01 = [(s + 1) / 2 for s in sims]
    return _clamp01(float(np.mean(sims01))), notes


# 7) Future learned compatibility (training later)
def learned_compatibility_score(
    top: Optional[Dict],
    bottom: Optional[Dict],
    dress: Optional[Dict],
    outerwear: Optional[Dict],
) -> Optional[float]:
    """
    Later:
    load models/compatibility_mlp.pt
    compute score in [0,1] for (top,bottom) and (outerwear,top/bottom) pairs
    For now: return None to disable.
    """
    return None


# Final score (weighted blend)
def score_outfit(
    event_key: str,
    top: Optional[Dict],
    bottom: Optional[Dict],
    dress: Optional[Dict],
    outerwear: Optional[Dict],
) -> ScoreBreakdown:
    if event_key not in EVENTS:
        raise KeyError(f"Unknown event '{event_key}'. Available: {list(EVENTS.keys())}")

    event = EVENTS[event_key]
    explanation: List[str] = []

    items = [x for x in [top, bottom, dress, outerwear] if x is not None]

    cat_score, notes = category_structure_score(top, bottom, dress, outerwear)
    explanation += notes

    yolo_score, notes = yolo_zara_consistency_score(items)
    explanation += notes

    evt_score, notes = event_suitability_score(event, items)
    explanation += notes

    form_score, notes = formality_alignment_score(event, items)
    explanation += notes

    col_score, notes = color_harmony_score(items)
    explanation += notes

    style_score, notes = style_compatibility_score(top, bottom, dress, outerwear)
    explanation += notes

    learned = learned_compatibility_score(top, bottom, dress, outerwear)

    # Weighting
    base = (
        0.18 * cat_score +
        0.10 * yolo_score +
        0.25 * evt_score +
        0.17 * form_score +
        0.10 * col_score +
        0.20 * style_score
    )

    if learned is not None:
        total = 0.75 * base + 0.25 * learned
    else:
        total = base

    total = _clamp01(total)

    # Convert total -> stars
    if total >= 0.86:
        stars = 5
    elif total >= 0.70:
        stars = 4
    elif total >= 0.55:
        stars = 3
    elif total >= 0.40:
        stars = 2
    else:
        stars = 1

    return ScoreBreakdown(
        total=total,
        rating_stars=stars,
        category_score=cat_score,
        event_score=evt_score,
        formality_score=form_score,
        color_score=col_score,
        style_score=style_score,
        yolo_consistency_score=yolo_score,
        learned_score=learned,
        explanation=explanation,
    )
