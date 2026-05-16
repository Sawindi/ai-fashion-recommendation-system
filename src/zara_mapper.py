# src/zara_mapper.py
"""
Maps YOLO DeepFashion2 classes to Zara-style categories.

"""

from typing import Dict, List


# Zara Categories 
ZARA_CATEGORIES: List[str] = [
    "blazer",
    "coat",
    "jacket",
    "knitwear",
    "cardigan",
    "shirt",
    "t-shirt",
    "top",
    "jeans",
    "trousers",
    "skirt",
    "shorts",
    "dress",
    "co-ord set",
    "sweatshirt",
]

# YOLO -> Zara Mapping
YOLO_TO_ZARA: Dict[str, List[str]] = {

    "short_sleeve_top": ["top", "t-shirt"],
    "long_sleeve_top": ["shirt", "top", "knitwear"],
    "short_sleeve_outwear": ["jacket"],
    "long_sleeve_outwear": ["blazer", "coat", "jacket"],
    "vest": ["top"],
    "sling": ["top"],
    "shorts": ["shorts"],
    "trousers": ["trousers", "jeans"],
    "skirt": ["skirt"],
    "short_sleeve_dress": ["dress"],
    "long_sleeve_dress": ["dress"],
    "vest_dress": ["dress"],
    "sling_dress": ["dress"],
}


# Suggestion Function

def suggest_zara_categories(yolo_class: str) -> Dict[str, List[str]]:
    if yolo_class not in YOLO_TO_ZARA:
        return {
            "primary": "top",
            "alternatives": []
        }

    suggestions = YOLO_TO_ZARA[yolo_class]

    return {
        "primary": suggestions[0],
        "alternatives": suggestions[1:]
    }
